import torch
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

from gluefactory.models.on3r.dense_tracks.tests import run_valid_lp_edges_test


# ---------------------------
# Simple scoring & gating
# ---------------------------
def build_score_function(Dq, Dr, Dc, alpha=1.0, beta=1.0, gamma=0.2, sigma_q=5.0, sigma_r=5.0):
    #Dc_i = Dc[:, None, :, None]
    #Dc_k = Dc[None, :, None, :]
    return alpha * torch.exp(-Dq / sigma_q) + beta * torch.exp(-Dr / sigma_r) + gamma * Dc / 2

def build_gate(Dq, Dr, thresh_query, thresh_sampson):
    return (Dq < thresh_query) & (Dr < thresh_sampson)

def get_valid_edges_mask(T, N, device):
    ik = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
    return ik[:, :, None, None].expand(T, T, N, N)

def get_topk_mnn(S, top_k=3):
    # Row-wise kth threshold (along last dim), and column-wise kth threshold (along dim=-2)
    kth_row = torch.topk(S, k=min(top_k, S.size(-1)), dim=-1).values[..., -1].unsqueeze(-1)
    kth_col = torch.topk(S, k=min(top_k, S.size(-2)), dim=-2).values[:, :, -1].unsqueeze(-2)
    return (S >= kth_row) & (S >= kth_col)

# ---------------------------
# 1-to-1 constraints (sparse)
# ---------------------------
def build_A_1to1_sub(keys):
    n_putative_edges = keys.shape[0]
    data = np.ones(n_putative_edges, dtype=np.float64)
    cols = np.arange(n_putative_edges, dtype=np.int64)
    rows = np.unique(keys, axis=0, return_inverse=True)[1].astype(np.int64)
    n_unique = int(rows.max()) + 1 if rows.size else 0
    #rows = inv.cpu().numpy()
    A = csr_matrix((data, (rows, cols)), shape=(n_unique, n_putative_edges), dtype=np.float64)
    b = np.ones(n_unique, dtype=np.float64)
    return A, b

def build_A1to1(E):
    keys_R = E[:, (0, 1, 2)]  # (n_putative_edges, 3)
    keys_C = E[:, (0, 1, 3)]  # (n_putative_edges, 3)
    A_r, b_r = build_A_1to1_sub(keys_R)
    A_c, b_c = build_A_1to1_sub(keys_C)
    A_1to1 = vstack([A_r, A_c], format="csr")
    b_1to1 = np.hstack([b_r, b_c])
    return A_1to1, b_1to1

# ---------------------------
# Edge coding & lookup
# ---------------------------
def _pack_edge(i, k, j, l, T, N):
    return (((i * T) + k) * N + j) * N + l  # Maps each edge to a unique code

def _codes_indexer(E, T, N):
    """
    Maps each edge to a unique code (see _pack_edge)
    """
    # Precompute sorted codes for all columns and a lookup that maps codes -> column index
    i_all, k_all, j_all, l_all = E[:,0], E[:,1], E[:,2], E[:,3]
    codes_all = _pack_edge(i_all, k_all, j_all, l_all, T, N)  # |E|
    order = np.argsort(codes_all)
    codes_sorted = codes_all[order]
    return (i_all, k_all, j_all, l_all, codes_sorted, order)

def _lookup_cols(codes_target, codes_sorted, order):
    # Vectorized search: returns (cols, hit_mask)
    pos = np.searchsorted(codes_sorted, codes_target)
    n_putative_edges = codes_sorted.size
    hit = (pos < n_putative_edges) & (codes_sorted[np.minimum(pos, n_putative_edges-1)] == codes_target)
    cols = np.full(codes_target.shape[0], -1, dtype=np.int64)
    cols[hit] = order[pos[hit]]
    return cols, hit

# ---------------------------
# Pair enumeration by shared node
# ---------------------------
def final_build_A_b_delta(e_idx, f_idx, hit, cols_3, n_putative_edges):
    n_hits = e_idx.size  # number of edges with same node
    rows_base = np.arange(n_hits, dtype=np.int64)
    rowsA = np.concatenate([rows_base, rows_base, rows_base[hit]])
    colsA = np.concatenate([e_idx,     f_idx,     cols_3[hit]])
    dataA = np.concatenate([np.ones(n_hits),  np.ones(n_hits), -np.ones(hit.sum())]).astype(np.float64)

    A_d = csr_matrix((dataA, (rowsA, colsA)), shape=(n_hits, n_putative_edges), dtype=np.float64)
    b_d = np.ones(n_hits, dtype=np.float64)
    return A_d, b_d

def correct_order(i, k, j, l):
    swap = i > k
    i_upd = np.where(swap, k, i)
    k_upd = np.where(swap, i, k)
    j_upd = np.where(swap, l, j)
    l_upd = np.where(swap, j, l)
    return i_upd, k_upd, j_upd, l_upd

def get_cols_hit(set1_ind_all, set2_ind_all, node1_ind_all, node2_ind_all, e_idx, f_idx, codes_sorted, order, T, N):
    # Third edge g between (i,j) and (k,l) from the two outgoing edges
    i, j = set1_ind_all[e_idx], node1_ind_all[e_idx]
    k, l = set2_ind_all[f_idx], node2_ind_all[f_idx]
    i_upd, k_upd, j_upd, l_upd = correct_order(i, k, j, l)
    codes3 = _pack_edge(i_upd, k_upd, j_upd, l_upd, T, N)  # We know which node that the extra node hits because of the order
    cols_3, hit = _lookup_cols(codes3, codes_sorted, order)
    return cols_3, hit

def get_C(fixed_node_codes, n_putative_edges, middle):
    """
    Returns:
        C (|E| x |E|): a bool mask which is true when two edges share the same node
    """
    rows = np.unique(fixed_node_codes, return_inverse=True)[1].astype(np.int64)
    cols = np.arange(n_putative_edges, dtype=np.int64)
    ones = np.ones(n_putative_edges, dtype=np.float64)
    n_unique_fixed_nodes = int(rows.max()) + 1
    if middle:
        rows_L = rows[:n_putative_edges]  # row id per edge in B_L
        rows_R = rows[n_putative_edges:]  # row id per edge in B_R
        B_L = csr_matrix((ones, (rows_L, cols)), shape=(n_unique_fixed_nodes, n_putative_edges), dtype=np.float64)
        B_R = csr_matrix((ones, (rows_R, cols)), shape=(n_unique_fixed_nodes, n_putative_edges), dtype=np.float64)
        # ---- Pairs (e,f) that share the same middle: C = B_L^T @ B_R
        C = B_L.T @ B_R  # sparse (n_putative_edges x n_putative_edges)
    else:
        # Build B (each column has exactly one 1)
        B = csr_matrix((ones, (rows, cols)), shape=(n_unique_fixed_nodes, n_putative_edges), dtype=np.float64)
        C = B.T @ B
    return C

def get_edges_that_meet(C, ordered):
    e_idx, f_idx = C.nonzero()
    if not ordered:
        # Ensure consistent ordering (e < f)
        keep = e_idx < f_idx  # drop diagonal and duplicates
        e_idx = e_idx[keep]
        f_idx = f_idx[keep]
    return e_idx, f_idx

def build_A_delta(E, T, N, mode):
    """
    if mode is kl
        Δ1 (middle = (k,l)): for each pair (i,k,j,l) and (k,m,l,n),
        add row: x_{ikjl} + x_{kmln} - [ (i,m,j,n)∈E ] x_{imjn} ≤ 1.
    if mode is ij
        Δ2 (middle = (i,j)): for each pair (i,k,j,l) and (i,m,j,n),
        add row: x_{ikjl} + x_{imjn} - [ (k,m,l,n)∈E ] x_{kmln} ≤ 1.
    if mode is mn
        Δ3 (middle = (m,n)): for each pair (i,m,j,n) and (k,m,l,n),
        add row: x_{imjn} + x_{kmln} - [ (i,k,j,l)∈E ] x_{ikjl} ≤ 1.
    """
    assert mode in ["kl", "ij", "mn"], "mode must be one of 'kl', 'ij', or 'mn'"
    n_putative_edges = E.shape[0]
    if n_putative_edges == 0:
        return csr_matrix((0,0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    i_all, k_all, j_all, l_all, codes_sorted, order = _codes_indexer(E, T, N)

    if mode == "kl":
        # ---- Build middle keys for left and right incidence
        # left middle (k,l), right middle (i,j)
        fixed_node_codes_L = k_all * N + l_all  # (n_putative_edges,)
        fixed_node_codes_R = i_all * N + j_all  # (n_putative_edges,)
        fixed_node_codes = np.concatenate([fixed_node_codes_L, fixed_node_codes_R], axis=0)  # unify row space for middles
    elif mode == "ij":
        fixed_node_codes = i_all * N + j_all  # (n_putative_edges,)
    else:
        fixed_node_codes = k_all * N + l_all  # (n_putative_edges,)

    is_middle = True if mode == "kl" else False
    C = get_C(fixed_node_codes, n_putative_edges, middle=is_middle)
    e_idx, f_idx = get_edges_that_meet(C, ordered=is_middle)
    if e_idx.size == 0:
        return csr_matrix((0, n_putative_edges), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    if mode == "kl":
        cols_3, hit = get_cols_hit(i_all, k_all, j_all, l_all, e_idx, f_idx, codes_sorted, order, T, N)
    elif mode == "ij":
        cols_3, hit = get_cols_hit(k_all, k_all, l_all, l_all, e_idx, f_idx, codes_sorted, order, T, N)
    else:
        cols_3, hit = get_cols_hit(i_all, i_all, j_all, j_all, e_idx, f_idx, codes_sorted, order, T, N)
    A_d, b_d = final_build_A_b_delta(e_idx, f_idx, hit, cols_3, n_putative_edges)
    return A_d, b_d


def build_tracks_from_edges(selected_edges, tuple_length, device):
    dtype = selected_edges.dtype

    # Tracks as a list of tensors to avoid repeated cat; merge by writing in-place.
    tracks = []
    # Map from observation (img, kp) -> track_id
    obs2track = {}

    for two_track in selected_edges:
        i, k, j, l = two_track.tolist()
        a = (int(i), int(j))
        b = (int(k), int(l))
        ta = obs2track.get(a, -1)
        tb = obs2track.get(b, -1)

        if ta == -1 and tb == -1:
            # new track
            track_id = len(tracks)
            tracks.append(torch.full((tuple_length,), -1, dtype=dtype, device=device))
            tracks[track_id][i] = j
            tracks[track_id][k] = l
            obs2track[a] = track_id
            obs2track[b] = track_id
        elif ta != -1 and tb == -1:
            tracks[ta][k] = l
            obs2track[b] = ta
        elif ta == -1 and tb != -1:
            tracks[tb][i] = j
            obs2track[a] = tb
        else:
            # If ta == tb, then (i, j) and (k, l) are already connected in the same track since
            # e.g. ((i, j), (m, n)) and ((k, l), (m, n)) are two two_tracks in selected_edges

            # ta != tb can happen if ((i, j), (m, n)) and ((k, l), (q, r)) from before,
            # and now we get ((i, j), (k, l)),so we must merge the track. Just make sure we have no duplicates in the same image.
            if ta != tb:
                src, dst = max(ta, tb), min(ta, tb)  # merge src into dst, keep dst
                # move all non -1 from src into dst and update mapping
                row_src = tracks[src]
                row_dst = tracks[dst]
                mask = row_src != -1
                assert (mask & (row_dst != -1)).sum() == 0, "Should never have to merge tracks with overlapping observations"
                row_dst[mask] = row_src[mask]
                tracks[dst] = row_dst  # Maybe this is not needed since the row above does this ...
                # update obs2track for moved observations
                for img_idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                    kp = int(row_dst[img_idx].item())
                    obs2track[(img_idx, kp)] = dst
                # invalidate src to save memory later
                tracks[src] = None

    # Stack only the non-empty rows
    tracks = torch.stack([r for r in tracks if r is not None], dim=0) if tracks else \
                        torch.empty((0, tuple_length), dtype=dtype, device=device)
    return tracks


def run_lp(Dq, Dr, Dc, device, thresh_s=0.80, thresh_sampson=5, topk=3, thresh_query=5, test=False):
    # thresh_s = 0.95  # corresponds to t_samp = 7, t_query = 3 and confidence > 0.77

    T, _, N, _ = Dq.shape
    # Build score map
    S = build_score_function(Dq, Dr, Dc)  # (T, T, N, N)
    
    # Build gate map
    G = build_gate(Dq, Dr, thresh_query=thresh_query, thresh_sampson=thresh_sampson)  # (T, T, N, N)

    # Get allowed edges
    valid = get_valid_edges_mask(T, N, device)
    mask_score = S > thresh_s
    mask_pre = valid & G & mask_score

    S_for_topk = S.masked_fill(~mask_pre, float('-inf'))
    mask_topk = get_topk_mnn(S_for_topk, top_k=topk)
    mask_edges = mask_pre & mask_topk  # final E (boolean mask)
    E = torch.argwhere(mask_edges)  # (|E|, 4) with (i, k, j, l)

    # Setup the linprog
    c = -S[mask_edges].detach().cpu().numpy().astype(np.float64)  # (|E|)
    if len(c) == 0:
        tracks_lp = torch.empty((0, T), dtype=torch.int64, device=device)
    else:
        E_np = E.detach().cpu().numpy()

        A_1to1, b_1to1 = build_A1to1(E_np)  # (n_1to1_constraints, |E|), (n_1to1_constraints,)
        A_d1, b_d1 = build_A_delta(E_np, T, N, mode="kl")
        A_d2, b_d2 = build_A_delta(E_np, T, N, mode="ij")
        A_d3, b_d3 = build_A_delta(E_np, T, N, mode="mn")
        A_ub = vstack([A_1to1, A_d1, A_d2, A_d3], format="csr")
        b_ub = np.concatenate([b_1to1, b_d1, b_d2, b_d3], axis=0)

        n_putative_edges = E.size(0)
        bounds = [(0.0, 1.0)] * n_putative_edges
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            exit("Linear Program Failed")

        selected_mask = res.x  > 0.95
        selected_edges = E[selected_mask]
        if test:
            run_valid_lp_edges_test(selected_edges)

        tracks_lp = build_tracks_from_edges(selected_edges, T, device)  # (n_tracks, T)
    return tracks_lp

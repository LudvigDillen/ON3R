
import torch
import numpy as np
from scipy.sparse import csr_matrix

import gluefactory.models.on3r.dense_tracks.lp as lp



def get_invalid_mask_constraint_1to1(i_all, k_all, j_or_l_all, T, N):
    # Group by (i,k,j) and (i,k,l) and flag groups with size > 1
    key = ((i_all * T) + k_all) * (N) + j_or_l_all
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    inconsistent_mask = counts[inv] > 1
    return inconsistent_mask

def get_invalid_assignment_mask(edges: torch.Tensor) -> torch.Tensor:
    """
    Flag edges that participate in any inconsistency:
      - One-to-one violation within a pair (i,k): a fixed (i,k,j) connects to multiple l, or a fixed (i,k,l) to multiple j
      - Triangle/transitivity violation: for any two edges that meet at a node, the closing edge is missing
        (covers Δ1: in×out at (k,l), Δ2: out×out at (i,j), Δ3: in×in at (m,n))

    Args:
        edges: LongTensor [m,4] with rows (i,k,j,l), canonical or not (we canonicalize internally).

    Returns:
        mask: BoolTensor [m], True for edges involved in any inconsistency.
    """
    m = edges.size(0)
    if m == 0:
        return torch.zeros((0,), dtype=torch.bool, device=edges.device)

    E_np = edges.detach().cpu().numpy().astype(np.int64)
    i_all, k_all, j_all, l_all = E_np[:,0], E_np[:,1], E_np[:,2], E_np[:,3]

    # Infer T,N bounds for packing (safe upper bounds)
    T = int(max(i_all.max(), k_all.max()) + 1)
    N = int(max(j_all.max(), l_all.max()) + 1)

    codes_all = lp._pack_edge(i_all, k_all, j_all, l_all, T, N)
    order = np.argsort(codes_all)
    codes_sorted = codes_all[order]
    invalid_mask = np.zeros(m, dtype=bool)

    # ===== 1) One-to-one violations (within each (i,k))
    # row-cap (fixed (i,k,j)) (if i, k, j is the same, l is only allowed to take 1 value)
    invalid_mask_1to1_l = get_invalid_mask_constraint_1to1(i_all, k_all, j_all, T, N)
    invalid_mask |= invalid_mask_1to1_l

    # col-cap (fixed (i,k,l)) (if i, k, j is the same, j is only allowed to take 1 value)
    invalid_mask_1to1_j = get_invalid_mask_constraint_1to1(i_all, k_all, l_all, T, N)
    invalid_mask |= invalid_mask_1to1_j

    # ===== 2) Triangle/transitivity violations (Δ1, Δ2, Δ3) without loops
    # Build incidence matrices once
    # Δ1: middle (k,l): edges ending at (k,l) vs edges starting at (k,l)
    mid_end = k_all * N + l_all          # (k,l)
    mid_start = i_all * N + j_all        # (i,j)

    # rows for unique mids
    mids_all = np.concatenate([mid_end, mid_start], axis=0)
    mids_unique, inv = np.unique(mids_all, return_inverse=True)
    rows_L = inv[:m]         # row-id per column in B_L  (by end)
    rows_R = inv[m:]         # row-id per column in B_R  (by start)
    n_unique_nodes = int(mids_unique.size)  # n unique nodes in selected edges

    ones = np.ones(m, dtype=np.float64)
    cols = np.arange(m, dtype=np.int64)  # One node for appearing (k, l)
    # B is of shape (n_u)
    B_L = csr_matrix((ones, (rows_L, cols)), shape=(n_unique_nodes, m), dtype=np.float64)
    B_R = csr_matrix((ones, (rows_R, cols)), shape=(n_unique_nodes, m), dtype=np.float64)

    # --- Δ1 pairs: (left,end) × (right,start) at same middle
    C1 = B_L.T @ B_R  # (m, m)
    e1, f1 = C1.nonzero()
    # For each pair, compute third edge (i,k=jL? no:) (i from left, m from right; j from left, n from right)
    if e1.size:
        iL, jL = i_all[e1], j_all[e1]
        mR, nR = k_all[f1], l_all[f1]
        swap = iL > mR
        i3 = np.where(swap, mR, iL)
        k3 = np.where(swap, iL, mR)
        j3 = np.where(swap, nR, jL)
        l3 = np.where(swap, jL, nR)
        codes3 = lp._pack_edge(i3, k3, j3, l3, T, N)
        _, hit = lp._lookup_cols(codes3, codes_sorted, order)
        # inconsistency if third edge absent
        bad = ~hit
        if np.any(bad):
            invalid_mask[e1[bad]] = True
            invalid_mask[f1[bad]] = True

    # --- Δ2 pairs: out×out at same (i,j)
    # Reuse B_R (built on start (i,j)). Pairs are from C2 = B_R^T @ B_R
    C2 = B_R.T @ B_R
    e2, f2 = C2.nonzero()
    # remove self and duplicates (keep e2 < f2)
    keep = e2 < f2
    e2, f2 = e2[keep], f2[keep]
    if e2.size:
        # third edge between (k1,l1) and (k2,l2)
        k1, l1 = k_all[e2], l_all[e2]
        k2, l2 = k_all[f2], l_all[f2]
        swap = k1 > k2
        i3 = np.where(swap, k2, k1)
        k3 = np.where(swap, k1, k2)
        j3 = np.where(swap, l2, l1)
        l3 = np.where(swap, l1, l2)
        codes3 = lp._pack_edge(i3, k3, j3, l3, T, N)
        _, hit = lp._lookup_cols(codes3, codes_sorted, order)
        bad = ~hit
        if np.any(bad):
            invalid_mask[e2[bad]] = True
            invalid_mask[f2[bad]] = True

    # --- Δ3 pairs: in×in at same (m,n) i.e., same end (k,l)
    # Reuse B_L (built on end (k,l)). Pairs are from C3 = B_L^T @ B_L
    C3 = B_L.T @ B_L
    e3, f3 = C3.nonzero()
    keep = e3 < f3
    e3, f3 = e3[keep], f3[keep]
    if e3.size:
        # third edge between (i1,j1) and (i2,j2)
        i1_, j1_ = i_all[e3], j_all[e3]
        i2_, j2_ = i_all[f3], j_all[f3]
        swap = i1_ > i2_
        i3 = np.where(swap, i2_, i1_)
        k3 = np.where(swap, i1_, i2_)
        j3 = np.where(swap, j2_, j1_)
        l3 = np.where(swap, j1_, j2_)
        codes3 = lp._pack_edge(i3, k3, j3, l3, T, N)
        _, hit = lp._lookup_cols(codes3, codes_sorted, order)
        bad = ~hit
        if np.any(bad):
            invalid_mask[e3[bad]] = True
            invalid_mask[f3[bad]] = True

    return torch.from_numpy(invalid_mask).to(edges.device)


def run_valid_lp_edges_test(selected_edges):
    mask_bad = get_invalid_assignment_mask(selected_edges)  # selected_edges: LongTensor [m,4]
    num_bad = int(mask_bad.sum().item())
    print("invalid edges:", num_bad)

import numpy as np
import time


def sample_star_topology_tuples(mutual_overlap_mat, k_queries, tuple_length, start_query_ref_threshold=0.05):
    """
    Samples star-topology tuples of images from the mutual overlap matrix.
    Star-topology means that the query images are connected to all sampled reference images, but
    the references are not connected to each other. In practice, perfect star-topology might be hard
    to achieve so, we allow some small overlap between references.

    Args:
        mutual_overlap_mat (np.ndarray): Mutual overlap matrix of shape (N, N) where N is the number of images of the scene.
        k_queries (int): Number of queries to sample.
        tuple_length (int): Number of reference images in each tuple.
    Returns:
        queries (np.ndarray): Indices of sampled query images of shape (k_queries,).
        k_n_ref_tuples (np.ndarray): Indices of sampled reference images for each query of shape
                                     (k_queries, tuple_length).
    """
    def _update_thresholds(ref_threshold, query_ref_threshold, margin):
        ref_threshold += 0.001  # query_ref_threshold
        sampling_success = True
        if ref_threshold >= query_ref_threshold + margin or ref_threshold >= 0.10:
            query_ref_threshold = max(0.8*query_ref_threshold, 0.05)
            if query_ref_threshold <= 0.05:
                margin += 0.01
                print("margin increased to", margin)
            if margin > 0.10:
                sampling_success = False
            ref_threshold = 0.00
        return ref_threshold, query_ref_threshold, margin, sampling_success

    t1 = time.time()
    seen_combos = set()
    k_n_ref_tuples = []
    queries = []
    n_candidates = mutual_overlap_mat.shape[0]  # N
    q_inds = np.arange(n_candidates)

    count = 0
    ref_threshold = 0.00
    query_ref_threshold = start_query_ref_threshold
    margin = 0.00
    found = False
    time_threshold = 360  # seconds
    sampling_success = True
    t2 = time.time()
    while (len(queries) < k_queries) and ((t2 - t1) < time_threshold) and sampling_success:
        t2 = time.time()
        q = np.random.choice(q_inds)
        count += 1

        # potential references = strong overlap with q (excluding q itself)
        refs = np.where(mutual_overlap_mat[q] >= query_ref_threshold)[0]
        refs = refs[refs != q]
        if len(refs) < tuple_length:
            # Not enough references, skip this query
            ref_threshold, query_ref_threshold, margin, sampling_success = _update_thresholds(
                ref_threshold, query_ref_threshold, margin)
            continue

        if tuple_length == 2:
            # Square sub-matrix of pair-wise scores among candidate refs
            sub = mutual_overlap_mat[np.ix_(refs, refs)]
            # Mask out the diagonal, then find the first pair (i<j) whose score ≤ max_rr
            i, j = np.nonzero(np.triu(sub <= ref_threshold + margin, k=1))
            n_candidates = len(i)
            if n_candidates > 0:
                candidate_ind = np.random.choice(np.arange(n_candidates))
                r1, r2 = refs[i[candidate_ind]], refs[j[candidate_ind]]
                # check for duplicate
                combo = (q, r1, r2)
                if combo not in seen_combos:
                    queries.append(q)
                    k_n_ref_tuples.append((r1, r2))
                    seen_combos.add(combo)
                    found = True
        else:
            good_mat = mutual_overlap_mat[np.ix_(refs, refs)] <= ref_threshold + margin
            stack = [([], 0)]                  # (chosen_indices_in_refs, next_start_pos)
            L = len(refs)

            while stack:
                chosen, start = stack.pop()

                # Success --------------------------------------------------------
                if len(chosen) == tuple_length:
                    ref_ids = tuple(refs[i] for i in chosen)
                    combo   = (q, *ref_ids)
                    if combo not in seen_combos:
                        queries.append(q)
                        k_n_ref_tuples.append([refs[i] for i in chosen])
                        seen_combos.add(combo)
                        found = True
                    break

                # Bound 1: not enough elements left ------------------------------
                if len(chosen) + (L - start) < tuple_length or start >= L:
                    continue

                # Branch A: include refs[start] if compatible --------------------
                compatible = True
                for c in chosen:
                    if not good_mat[start, c]:
                        compatible = False
                        break
                if compatible:
                    stack.append((chosen + [start], start + 1))

                # Branch B: skip refs[start] -------------------------------------
                stack.append((chosen, start + 1))

        if found == True:
            ref_threshold = 0.00
            query_ref_threshold = start_query_ref_threshold
            margin = 0.00
            found = False
        else:
            ref_threshold, query_ref_threshold, margin, sampling_success = _update_thresholds(
                ref_threshold, query_ref_threshold, margin)

    if len(queries) < k_queries:
        sampling_success = False
    queries = np.array(queries, dtype=np.int_)  # (k_queries,)
    k_n_ref_tuples = np.array(k_n_ref_tuples, dtype=np.int_)  # (k_queries, T)

    verbose = False
    if verbose and sampling_success:
        # Print some overlap statistics
        query_ref_vecs = mutual_overlap_mat[queries]  # k_queries, N
        overlaps_qr = query_ref_vecs[np.tile(np.arange(k_queries)[:, None], (1, tuple_length)),
                                     k_n_ref_tuples]

        overlaps_rr = []
        for i in range(k_n_ref_tuples.shape[0]):
            submat = mutual_overlap_mat[np.ix_(k_n_ref_tuples[i], k_n_ref_tuples[i])]

            # get the indices of the upper triangle, k=1 excludes the diagonal
            r, c = np.triu_indices_from(submat, k=1)

            # extract the values
            upper_vals = submat[r, c]
            overlaps_rr.append(upper_vals)
        overlaps_rr = np.array(overlaps_rr)  # (k_queries, T*(T-1)/2)

        print(f"\nMean/median overlap query to refs {overlaps_qr.mean():.3f}"
              f" {np.median(overlaps_qr):.3f}")
        print(f"Mean/median overlap between refs {overlaps_rr.mean():.3f}"
              f" {np.median(overlaps_rr):.3f}")
    if not sampling_success:
        print(f"Cannot sample {k_queries} tuples in time threshold {time_threshold} seconds from this scene.")

    return queries, k_n_ref_tuples, sampling_success

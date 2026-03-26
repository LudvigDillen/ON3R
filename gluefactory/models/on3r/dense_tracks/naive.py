import torch


# Naive approach to creating tracks for RoMa
def mnn(dists, thresh=3):
    # Assumes S is (N, N)
    device = dists.device
    a1 = torch.zeros_like(dists, dtype=torch.bool)
    a2 = torch.zeros_like(dists, dtype=torch.bool)
    inds_rows = torch.arange(dists.shape[0], device=device)
    inds_cols = torch.arange(dists.shape[1], device=device)
    a1[inds_rows, dists.argmin(dim=1)] = True
    a2[dists.argmin(dim=0), inds_cols] = True
    assignment_mat = a1 & a2 & (dists < thresh)
    return assignment_mat # (N, N)


def create_1tracks(new_query_inds, tuple_length, tuple_inds):
    device = new_query_inds.device
    n_inds = len(new_query_inds)
    new_tracks = torch.full((n_inds, tuple_length), -1, dtype=torch.long, device=device)
    if isinstance(tuple_inds, int):
        new_tracks[:, tuple_inds] = new_query_inds
    else:
        assert len(tuple_inds) == n_inds, "should not happen"
        new_tracks[:, tuple_inds] = new_query_inds
    return new_tracks  # (n_inds, tuple_length)


def create_naive_tracks(all_query_kpts, all_sampson_errors=None, improved_version=False, px_thresh=5, run_test=False):
    """
    Create naive tracks for the given query keypoints.

    Input:
        all_query_kpts: Tensor (tuple_length, n_samples, 2) containing the kpts for all queries
        all_sampson_errors: Tensor of shape (tuple_length, tuple_length, n_samples, n_samples)
                            containing the Sampson errors for all pairs of corresponding refs.
        improved_version: Whether to use the improved version of the algorithm.
        px_thresh: Pixel threshold.
        run_test: Whether to run a test.
    Output:
        naive_tracks: A tensor of shape (n_samples, tuple_length) containing the created tracks.
        naive_query_kpts_tracks: A tensor of shape (n_samples, tuple_length, 2) containing the
                                 kpts for the created tracks.

    Note: If improved_version is True, all_sampson_errors must be provided.
    """
    if improved_version:
        assert all_sampson_errors is not None, "need them to calculate ref-ref-errors"

    tuple_length, n_samples, _ = all_query_kpts.shape
    device = all_query_kpts.device
    start_query_inds = torch.arange(n_samples, device=device)
    query_inds_wide_accumulated = torch.full((0, tuple_length), -1, device=device)
    new_tracks = create_1tracks(start_query_inds, tuple_length, 0)
    query_inds_wide_accumulated = torch.cat((query_inds_wide_accumulated, new_tracks), dim=0)
    query_kpts_wide_accumulated = all_query_kpts[0]  # (n_unique_kpts_added, 2)

    for i in range(1, tuple_length):
        new_query_kpts = all_query_kpts[i]  # (N, 2)
        query_dists = torch.norm(new_query_kpts[None] - query_kpts_wide_accumulated[:, None], dim=-1)  # (n_unique_kpts_added, N)

        if improved_version: # Sampson_dists
            n_samples_accumuluted = query_inds_wide_accumulated.shape[0]
            start_query_inds_exp = start_query_inds[None, :, None].repeat(
                n_samples_accumuluted, 1, tuple_length)
            query_inds_wide_accumulated_exp = query_inds_wide_accumulated[:, None].repeat(
                1, n_samples, 1)
            tuple_inds_all = torch.arange(tuple_length, device=device)[None, None, :].repeat(
                n_samples_accumuluted, n_samples, 1)
            new_pair_ind = torch.tensor(i, device=device)[None, None, None].repeat(
                n_samples_accumuluted, n_samples, tuple_length)
            # (n_unique_kpts_added, N, tuple_length)
            sampson_errors_full = all_sampson_errors[
                tuple_inds_all, new_pair_ind, query_inds_wide_accumulated_exp, start_query_inds_exp]
            valid = query_inds_wide_accumulated != -1  # (n_unique_kpts_added, tuple_length)
            sampson_errors_mean = (sampson_errors_full * valid[:, None]).sum(dim=-1) / valid.sum(dim=1)[:, None]  # (n_unique_kpts_added, N)
            dists = (query_dists + sampson_errors_mean) / 2
        else:
            dists = query_dists

        assignment_mat = mnn(dists, thresh=px_thresh)  # (n_unique_kpts_added, N)
        # (first col: index in query_keypoints_added, second col: index in new_query_kpts)
        matched_existing_inds, matched_new_inds = assignment_mat.argwhere().unbind(dim=1)  # (M, 2)

        if len(matched_new_inds):
            query_inds_wide_accumulated[matched_existing_inds, i] = matched_new_inds
            query_kpts_matched = query_kpts_wide_accumulated[matched_existing_inds]  # (M, 2)
            new_query_kpts_matched = new_query_kpts[matched_new_inds]  # (M, 2)
            track_lenghts_matched = (query_inds_wide_accumulated[matched_existing_inds] != -1).sum(dim=1)  # (M,)
            w = ((track_lenghts_matched - 1) / track_lenghts_matched)[:, None]  # (M, 1)
            query_kpts_wide_accumulated[matched_existing_inds] = w * query_kpts_matched + (1 - w) * new_query_kpts_matched

        # Now build tracks (either create new or append to existing)
        remaining_inds_mask = ~torch.isin(start_query_inds, matched_new_inds)
        n_new_tracks = remaining_inds_mask.sum()
        assert n_new_tracks + len(matched_new_inds) == n_samples, "should not happen"
        if n_new_tracks:
            inds_to_add = start_query_inds[remaining_inds_mask]  # (n_new_tracks,)
            kpts_to_add = new_query_kpts[remaining_inds_mask]  # (n_new_tracks, 2)
            new_tracks = create_1tracks(inds_to_add, tuple_length, i)
            query_inds_wide_accumulated = torch.cat((query_inds_wide_accumulated, new_tracks), dim=0)
            query_kpts_wide_accumulated = torch.cat((query_kpts_wide_accumulated, kpts_to_add), dim=0)

    track_lengths = (query_inds_wide_accumulated != -1).sum(dim=1)
    two_track_mask = track_lengths >= 2
    naive_tracks = query_inds_wide_accumulated[two_track_mask]  # keep only tracks of length >= 2
    naive_query_kpts_tracks = query_kpts_wide_accumulated[two_track_mask]

    # Assert that no duplicates exist
    if run_test:
        for i in range(tuple_length):
            col = query_inds_wide_accumulated[:, i]
            mask = col != -1
            assert (len(col[mask]) == n_samples), "Some query keypoints are missing from the tracks"
            assert (len(col[mask].unique()) == n_samples), \
                "Duplicates in the same image should not happen"

    return naive_tracks, naive_query_kpts_tracks

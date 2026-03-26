import torch


# Select one query position, Rewarp at query position
def from_normalized_to_pixels(kpts_norm, wh):
    """
    Input:
        kpts_norm: Normalized keypoints (N, 2) in the range ([-1 + 1/w, 1 - 1/w], [-1 + 1/h, 1 - 1/h]) (or subpixel)
        wh: width height to scale to
    
    Output:
        kpts: Upsampled keypoints (N, 2) in pixel coordinates (or subpixel)
    """
    if isinstance(wh, tuple):
        wh = torch.tensor(wh, device=kpts_norm.device, dtype=kpts_norm.dtype)
    wh_ = wh[None]
    kpts = (wh_*kpts_norm + wh_ - 1) / 2
    return kpts


def from_pixels_to_normalized(kpts, wh):
    """
    Input:
        kpts: Keypoints (N, 2) in pixel coordinates (or subpixel)
        wh: width height to scale to

    Output:
        kpts: Normalized keypoints (N, 2) in the range ([-1 + 1/w, 1 - 1/w], [-1 + 1/h, 1 - 1/h])
    """
    if isinstance(wh, tuple):
        wh = torch.tensor(wh, device=kpts.device, dtype=kpts.dtype)
    wh_ = wh[None]
    kpts_norm = (2*kpts + 1 - wh_) / wh_
    return kpts_norm


def round_kpts(kpts_, wh):
    kpts = torch.round(kpts_).long()  # Round to nearest pixel
    kpts[:, 0] = torch.clamp(kpts[:, 0], 0, wh[0] - 1)
    kpts[:, 1] = torch.clamp(kpts[:, 1], 0, wh[1] - 1)
    return kpts


def from_img_res_to_upsample_res(kpts, wh, wh_out_roma):
    if isinstance(wh_out_roma, tuple):
        wh_out_roma = torch.tensor(wh_out_roma, device=kpts.device, dtype=kpts.dtype)
    kpts_ = kpts * wh_out_roma / wh
    return kpts_


def get_query_kpts_to_resample_from_conf_based(tracks, all_certainties, all_matches, wh):
    tuple_length, device = tracks.shape[1], tracks.device

    invalid_elements_mask = tracks == -1  # (n_tracks, T)
    n_tracks = tracks.shape[0]
    img_inds = torch.arange(tuple_length, device=device, dtype=tracks.dtype)
    img_inds_exp = img_inds.unsqueeze(0).expand(n_tracks, -1)
    certainties_in_tracks = all_certainties[img_inds_exp, tracks].clone()  # (n_tracks, tuple_length)
    certainties_in_tracks[invalid_elements_mask] = 0
    ind_max_certainty_in_track = certainties_in_tracks.argmax(dim=1)  # (n_tracks,)
    track_inds = torch.arange(n_tracks, device=device, dtype=tracks.dtype)
    inds_to_rewarp_from = tracks[track_inds, ind_max_certainty_in_track]  # (n_tracks,)
    query_kpt_to_rewarp_from_norm = all_matches[ind_max_certainty_in_track, inds_to_rewarp_from, :2].clone()  # (n_tracks, 2)
    # (n_tracks, 2) (in subpixels)
    query_kpt_to_rewarp_from = from_normalized_to_pixels(query_kpt_to_rewarp_from_norm, wh)  # approx. (-1, 1) to (0, h/w)
    return query_kpt_to_rewarp_from


def resample_matches(tracks, scaler_query, scalers_ref, query_kpt_to_rewarp_from, all_warps, roma_model, wh):
    tuple_length = tracks.shape[1]
    wh_out_roma = roma_model.get_output_resolution()  # (w, h)
    invalid_elements_mask = tracks == -1  # (n_tracks, T)

    # Get coordinates in (-1, 1) that are our new tracks
    improved_naive_query_kpts_tracks_ur = from_img_res_to_upsample_res(query_kpt_to_rewarp_from, wh, wh_out_roma)
    query_kpt_to_rewarp_from = round_kpts(improved_naive_query_kpts_tracks_ur, wh_out_roma)  # (n_tracks, 2)
    new_ref_kpts = all_warps[:, query_kpt_to_rewarp_from[:, 1], query_kpt_to_rewarp_from[:, 0], 2:]  # sample ref kpts
    query_kpts_norm_rewarped_from = from_pixels_to_normalized(query_kpt_to_rewarp_from, wh_out_roma)
    query_kpts_norm_rewarped_from_rep = query_kpts_norm_rewarped_from[None].repeat(tuple_length, 1, 1)
    track_coords = torch.cat((query_kpts_norm_rewarped_from_rep, new_ref_kpts), dim=-1)  # (tuple_length, n_tracks, 4)

    # Get tracks in pixel coordinates and remove invalid elements (some query keypoints do not exist in all images)
    height_q = scaler_query[1]
    width_q = scaler_query[0]
    query_kpts_tracks = torch.zeros_like(track_coords[..., :2])  # (tuple_length, n_tracks, 2)
    ref_kpts_tracks = torch.zeros_like(track_coords[..., 2:])  # (tuple_length, n_tracks, 2)
    for i in range(tuple_length):
        height_r = scalers_ref[i, 1]
        width_r = scalers_ref[i, 0]
        query_kpts_tracks[i], ref_kpts_tracks[i] = roma_model.to_pixel_coordinates(track_coords[i], height_q, width_q, height_r, width_r)
        invalid = invalid_elements_mask[:, i]
        query_kpts_tracks[i, invalid], ref_kpts_tracks[i, invalid] = -1, -1
    one_img_ind_per_track = invalid_elements_mask.float().argmin(dim=1)
    aranged_tracks = torch.arange(tracks.shape[0], device=tracks.device)
    query_kpts_tracks_one = query_kpts_tracks[one_img_ind_per_track, aranged_tracks]
    return query_kpts_tracks_one, ref_kpts_tracks

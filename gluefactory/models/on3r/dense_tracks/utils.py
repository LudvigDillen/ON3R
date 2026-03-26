import torch


def get_ref_kpts_tracks(tracks, all_ref_kpts):
    """
    Input:
        tracks: (n_tracks, T) Inds to keypoints
        all_ref_kpts: (T, num, 2) All reference keypoints for each image in the tuple
    Output:
        ref_kpts_tracks: (T, n_tracks, 2) Reference keypoints for each track
    """
    n_tracks, tuple_length = tracks.shape
    device = tracks.device
    invalid_elements_mask = (tracks == -1)

    img_inds = torch.arange(tuple_length, device=device, dtype=tracks.dtype)
    img_inds_exp = img_inds.unsqueeze(0).expand(n_tracks, -1)
    ref_kpts_tracks = all_ref_kpts[img_inds_exp.T, tracks.T]
    ref_kpts_tracks[invalid_elements_mask.T] = -1  # (T, n_tracks, 2)
    return ref_kpts_tracks

import torch

from gluefactory.geometry.wrappers import Pose, Camera
from gluefactory.geometry.epipolar import T_to_F


def get_sampson_error(x1, x2, F, img_inds_x1, img_inds_x2, squared=False):
    """
    Compute the Sampson error between two sets of keypoints x1, x2 given the fundamental matrix F.
    Args:
        x1: torch.Tensor of shape (N, 2) with image keypoints
        x2: torch.Tensor of shape (N, 2) with image keypoints
        F: torch.Tensor of shape (T, T, 3, 3) with fundamental matrices.
        img_inds_x1: torch.Tensor of shape (N) with the image indices of the keypoints in x1.
        img_inds_x2: torch.Tensor of shape (N) with the image indices of the keypoints in x2.
        squared: bool, if True, the squared Sampson error is returned.
    Returns:
        error: torch.Tensor of shape (N) with the Sampson error for each pair of keypoints.
    """
    N = x1.shape[0]
    dev = x1.device
    dt = x1.dtype
    x1_h = torch.cat([x1, torch.ones((N, 1), device=dev, dtype=dt)], dim=1)
    x2_h = torch.cat([x2, torch.ones((N, 1), device=dev, dtype=dt)], dim=1)

    # Extract the relevant fundamental matrices based on img_inds_x1 and img_inds_x2
    F_selected = F[img_inds_x1, img_inds_x2]  # (N, 3, 3)

    # Compute Fx1 and F^T*x2
    Fx1 = torch.einsum('nij,nj->ni', F_selected, x1_h)  # Shape: (N, 3)
    FTx2 = torch.einsum('nij,ni->nj', F_selected, x2_h)  # Shape: (N, 3)

    # Compute the numerator of the Sampson error
    numerator = torch.einsum('ni,ni->n', x2_h, Fx1)  # Shape: (N,)

    # Compute the denominator of the Sampson error
    a = Fx1[..., 0]**2 + Fx1[..., 1]**2  # Shape: (N,)
    b = FTx2[..., 0]**2 + FTx2[..., 1]**2  # Shape: (N,)

    if squared:
        denominator = a + b
        sampson_error = numerator**2 / denominator.clamp(min=1e-6)  # Avoid division by zero
    else:
        denominator = torch.sqrt(a + b)  # Shape: (B, N, T, K)
        sampson_error = torch.abs(numerator) / denominator.clamp(min=1e-6)  # Avoid division by zero
    return sampson_error

def get_sampson_error_all(kpts, F, squared=False):
    """
    Compute the Sampson error between kpts and kpts given the fundamental matrix F.
    Args:
        kpts: torch.Tensor of shape (T, N, 2) with image keypoints
        F: torch.Tensor of shape (T, T, 3, 3) with fundamental matrices.
        squared: bool, if True, the squared Sampson error is returned.
    Returns:
        error: torch.Tensor of shape (T, T, N, N) with the Sampson error for each pair of keypoints.
    """
    T, N, _ = kpts.shape
    dev = kpts.device
    dt = kpts.dtype
    kpts_h = torch.cat([kpts, torch.ones((T, N, 1), device=dev, dtype=dt)], dim=-1)

    # Extract the relevant fundamental matrices based on img_inds_x1 and img_inds_x2

    # Compute Fx1 and F^T*x2
    Fx1 = torch.einsum('ktij,knj->ktni', F, kpts_h)  # Shape: (T, T, N, 3)
    FTx2 = torch.einsum('ktij,tni->ktnj', F, kpts_h)  # Shape: (T, T, N, 3)

    # Compute the numerator of the Sampson error
    numerator = torch.einsum('tmi,ktni->ktnm', kpts_h, Fx1)  # Shape: (T, T, N, N)

    # Compute the denominator of the Sampson error
    a = Fx1[..., 0]**2 + Fx1[..., 1]**2  # Shape: (T, T, N)
    b = FTx2[..., 0]**2 + FTx2[..., 1]**2  # Shape: (T, T, N)
    denom = a.unsqueeze(-1) + b.unsqueeze(-2)  # Shape: (T, T, N, N)

    if squared:
        sampson_error = numerator**2 / denom.clamp(min=1e-6)  # Avoid division by zero
    else:
        sampson_error = torch.abs(numerator) / denom.sqrt().clamp(min=1e-6)  # Avoid division by zero
    return sampson_error

def check_F_matrices(F):
    """
    Check that the fundamental matrices are all of rank 2 (if not all zeros or identity)
    and that F[i, j] equals the transpose of F[j, i].

    Args:
        F: torch.Tensor of shape (tuple_length, tuple_length, 3, 3)

    Returns:
        valid: A boolean indicating whether all checks passed.
        rank_issues: A list of indices (i, j) where rank check failed.
        symmetry_issues: A list of indices (i, j) where symmetry check failed.
    """
    tuple_length, _, _, _ = F.shape
    rank_issues = []
    symmetry_issues = []

    # Tolerance for checking if a matrix is effectively identity or zero
    tolerance = 1e-6

    for i in range(tuple_length):
        for j in range(tuple_length):
            F_ij = F[i, j]

            # Check if F[i, j] equals the transpose of F[j, i]
            F_ji_transposed = F[j, i].transpose(-1, -2)
            if not torch.allclose(F_ij, F_ji_transposed, atol=tolerance):
                symmetry_issues.append((i, j))

            # Check the rank only if the matrix is not identity or zero
            if not (torch.allclose(F_ij, torch.eye(3, device=F.device), atol=tolerance) or
                    torch.allclose(F_ij, torch.zeros(3, 3, device=F.device), atol=tolerance)):
                if torch.linalg.matrix_rank(F_ij) != 2:
                    rank_issues.append((i, j))

    valid = len(rank_issues) == 0 and len(symmetry_issues) == 0
    return valid, rank_issues, symmetry_issues

def fundamental_from_pose(poses, intrinsics, dev):
    """
    Compute the fundamental matrix from the given poses and intrinsics.

    Args:
        poses of len (T) of type Pose representing the poses (w2cam)
        intrinsics of len (T) of type Camera reprsenting the camera intrinsics and image format.

    Returns:
        F: (T, T, 3, 3) tensor representing the fundamental matrices.
    """
    tuple_length = len(poses)  # Exclude the reference pose
    F = torch.zeros((tuple_length, tuple_length, 3, 3), device=dev)  # fundamental matrix

    for i in range(tuple_length):
        for j in range(tuple_length):
            if i == j:
                F[i, j] = torch.eye(3, device=dev)
            elif i < j:
                T_i2j = poses[j].compose(poses[i].inv())
                F[i, j] = T_to_F(intrinsics[i], intrinsics[j], T_i2j)
                F[j, i] = F[i, j].T
    #a, b, c = check_F_matrices(F)  # It is ok!
    #print(f"Fundamental matrix checks - Rank issues: {b}, Symmetry issues: {c}")
    return F


def get_all_sampson_error_handler(poses, intrinsics, all_ref_kpts, device):
    # Load fundamental matrices
    if isinstance(poses, torch.Tensor):
        poses_cls = [Pose.from_4x4mat(pose) for pose in poses]
    else:
        poses_cls = poses

    if isinstance(intrinsics, torch.Tensor):
        camera_cls = [Camera.from_calibration_matrix(K) for K in intrinsics]
    else:
        camera_cls = intrinsics

    F = fundamental_from_pose(poses_cls, camera_cls, dev=device)  # (T, T, 3, 3)
    all_sampson_errors = get_sampson_error_all(all_ref_kpts, F, squared=False)  # (T, T, N, N)
    return all_sampson_errors


def get_sampson_error_tracks(tracks_, all_sampson_errors):
    """
    Input:
        tracks_: of shape (n_tracks, T): with inds to the corresponding keypoints
        all_sampson_errors: of shape (T, T, num, num): the Sampson errors for all keypoint pairs
    Output:
        sampson_error_tracks: of shape (n_tracks, T, T): the Sampson errors for the keypoints in the tracks
        invalid_sampson_error_tracks: of shape (n_tracks, T, T): a mask indicating invalid Sampson errors
    """
    n_tracks_, tuple_length_ = tracks_.shape
    device = tracks_.device
    invalid_mask = tracks_ == -1

    tuple_inds = torch.arange(tuple_length_, device=device)
    invalid_sampson_error_tracks = invalid_mask[:, None] | invalid_mask[..., None]  # (n_tracks, T, T)
    invalid_sampson_error_tracks[:, tuple_inds, tuple_inds] = True

    # All tensors below have shape (n_tracks, T, T)
    error_inds1 = tracks_[:, :, None].repeat(1, 1, tuple_length_)
    error_inds2 = tracks_[:, None].repeat(1, tuple_length_, 1)
    tuple_inds1 = tuple_inds[None, :, None].repeat(n_tracks_, 1, tuple_length_)
    tuple_inds2 = tuple_inds[None, None].repeat(n_tracks_, tuple_length_, 1)
    sampson_error_tracks = all_sampson_errors[tuple_inds1, tuple_inds2, error_inds1, error_inds2]
    sampson_error_tracks[invalid_sampson_error_tracks] = -1.0  # invalidate errors where either element is invalid
    return sampson_error_tracks, invalid_sampson_error_tracks

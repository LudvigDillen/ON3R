import torch

from gluefactory.geometry.epipolar import T_to_E


def calc_sampson_errors(tracks, pred, data, ret_mode):
    """
    Calculate Sampson errors for each pair of keypoints.

    Args:
    - tracks (torch.Tensor): Shape (B, N_tracks, T), containing track information.
    - pred (dict): Dictionary with prediction data, containing keypoints.
    - data: Additional data required to compute the F matrix.

    Returns:
    - valid_sampson_errors (torch.Tensor): 1D tensor of valid Sampson errors.
    """
    B, N_tracks, tuple_length = tracks.shape
    dev = tracks.device

    x2 = []
    for key in pred.keys():
        x2.append(pred[key]['keypoints1'])
    x2 = torch.cat(x2, dim=1)
    F = get_F_matrices(data)  # B x tuple_length x tuple_length x 3 x 3

    batch_tracks_inds = torch.arange(B, device=dev).view(-1, 1, 1).expand_as(tracks)
    x1_input = x2[batch_tracks_inds, tracks]  # B x N_rows x tuple_length x 2
    x2_input = x1_input.clone()  # B x N_rows x tuple_length x 2
    img_inds_tracks = torch.arange(tuple_length, device=dev)[None, None].expand_as(tracks)

    # (B, N, T, T) with the Sampson error for each pair of keypoints in each track.
    samp_errors = get_sampson_error_multi_view(
        x1_input, x2_input, F, img_inds_tracks, img_inds_tracks)

    mask = tracks == -1
    mask_TT = mask[:, :, :, None] | mask[:, :, None]
    samp_errors[mask_TT] = -1
    inds = torch.arange(tuple_length, device=dev)
    samp_errors[:, :, inds, inds] = -1
    if ret_mode == "only ordered":
        return samp_errors

    # Remove bottom triangle of the matrix to avoid double counting. This also removes the diagonal.
    # Create a lower triangular mask for a single TxT matrix
    mask_single_matrix = torch.tril(torch.ones((tuple_length, tuple_length), dtype=bool))
    # Expand the mask to match the full shape of samp_errors
    mask_bottom_triangle = mask_single_matrix.expand(B, N_tracks, tuple_length, tuple_length)
    se_clone = samp_errors.clone()
    se_clone[mask_bottom_triangle] = -1

    mask_s = se_clone != -1
    valid_sampson_errors = se_clone[mask_s]
    return valid_sampson_errors, samp_errors


def get_F_matrices(data):
    tuple_length = len(data)
    key0 = list(data.keys())[0]
    # B = len(data[key0]['scene'])
    B = data[key0]['T_0to1'].R.shape[0]
    dev = data[key0]['T_0to1'].device
    F = torch.zeros((B, tuple_length, tuple_length, 3, 3), device=dev)  # fundamental matrix

    for i, key in enumerate(data.keys()):
        for j, key_inner in enumerate(data.keys()):
            if i == j:
                F[:, i, j] = torch.eye(3, device=dev)[None].repeat(B, 1, 1)
            elif i < j:
                # NOTE That F_from_data below looks at T between ref and query, and we are
                #      interested in between ref and ref. MATH for pose stuff:
                #          T_ref0_to_ref1 =  T_query_to_ref1 @ T_ref0_to_query
                T_ref_i_to_ref_j = data[key_inner]["T_0to1"] @ data[key]["T_1to0"]
                K_i = data[key]["view1"]["camera"]
                K_j = data[key_inner]["view1"]["camera"]

                F[:, i, j] = F_from_data(T_ref_i_to_ref_j, K_i, K_j)
                F[:, j, i] = F[:, i, j].transpose(1, 2)
    # a, b, c = check_F_matrices(F)  # It is ok!
    return F


def check_F_matrices(F):
    """
    Check that the fundamental matrices are all of rank 2 (if not all zeros or identity)
    and that F[i, j] equals the transpose of F[j, i].

    Args:
        F: torch.Tensor of shape (B, tuple_length, tuple_length, 3, 3)

    Returns:
        valid: A boolean indicating whether all checks passed.
        rank_issues: A list of indices (i, j) where rank check failed.
        symmetry_issues: A list of indices (i, j) where symmetry check failed.
    """
    B, tuple_length, _, _, _ = F.shape
    rank_issues = []
    symmetry_issues = []

    # Tolerance for checking if a matrix is effectively identity or zero
    tolerance = 1e-6

    for b in range(B):
        for i in range(tuple_length):
            for j in range(tuple_length):
                F_ij = F[b, i, j]

                # Check if F[i, j] equals the transpose of F[j, i]
                F_ji_transposed = F[b, j, i].transpose(-1, -2)
                if not torch.allclose(F_ij, F_ji_transposed, atol=tolerance):
                    symmetry_issues.append((b, i, j))

                # Check the rank only if the matrix is not identity or zero
                if not (torch.allclose(F_ij, torch.eye(3, device=F.device), atol=tolerance) or
                        torch.allclose(F_ij, torch.zeros(3, 3, device=F.device), atol=tolerance)):
                    if torch.linalg.matrix_rank(F_ij) != 2:
                        rank_issues.append((b, i, j))

    valid = len(rank_issues) == 0 and len(symmetry_issues) == 0
    return valid, rank_issues, symmetry_issues


def F_from_data(T_ij, K_i, K_j):
    F = (
        K_j.calibration_matrix().inverse().transpose(-1, -2)
        @ T_to_E(T_ij)
        @ K_i.calibration_matrix().inverse()
    )
    return F


def get_query_img_dims(data):
    query_img_dims = []
    for key in data.keys():
        query_img_dims = data[key]['view0']['image_size'] # (B, 2)
        break

    if len(query_img_dims) == 0:
        raise ValueError("Could not find the query norm factor.")  # Should never happen
    return query_img_dims


def get_ref_img_dims(data):
    ref_img_dims = []
    for key in data.keys():
        view1 = data[key]['view1']['image_size'] # (B, 2)
        ref_img_dims.append(view1)

    if len(ref_img_dims) == 0:
        raise ValueError("Could not find the query norm factor.")  # Should never happen
    ref_img_dims = torch.stack(ref_img_dims, dim=1)  # (B, T)
    return ref_img_dims


def get_sampson_error_multi_view(x1, x2, F, img_inds_x1, img_inds_x2, squared=False):
    """
    Compute the Sampson error between two sets of keypoints x1, x2 given the fundamental matrix F.
    Args:
        x1: torch.Tensor of shape (B, N, T, 2) with image keypoints
        x2: torch.Tensor of shape (B, N, K, 2) with image keypoints
        F: torch.Tensor of shape (B, T, T, 3, 3) with fundamental matrices.
        img_inds_x1: torch.Tensor of shape (B, N, T) with the image indices of the keypoints in x1.
        img_inds_x2: torch.Tensor of shape (B, N, K) with the image indices of the keypoints in x2.
        squared: bool, if True, the squared Sampson error is returned.
    Returns:
        error: torch.Tensor of shape (B, N, T, K) with the Sampson error for each pair of keypoints.
    """
    B, N, T, _ = x1.shape
    B, N, K, _ = x2.shape
    dev = x1.device
    dt = x1.dtype
    x1_h = torch.cat([x1, torch.ones((B, N, T, 1), device=dev, dtype=dt)], dim=3)
    x2_h = torch.cat([x2, torch.ones((B, N, K, 1), device=dev, dtype=dt)], dim=3)

    # Broadcast img_inds_x1 and img_inds_x2 for indexing
    inds_x1 = img_inds_x1.unsqueeze(3).expand(B, N, T, K)
    inds_x2 = img_inds_x2.unsqueeze(2).expand(B, N, T, K)

    # Extract the relevant fundamental matrices based on img_inds_x1 and img_inds_x2
    F_selected = F[torch.arange(B, device=dev).view(B, 1, 1, 1), inds_x1, inds_x2]  # (B, N, T, K, 3, 3)

    # Compute Fx1 and F^T*x2
    Fx1 = torch.einsum('bntkij,bntj->bntki', F_selected, x1_h)  # Shape: (B, N, T, K, 3)
    FTx2 = torch.einsum('bntkij,bnki->bntkj', F_selected, x2_h)  # Shape: (B, N, T, K, 3)

    # Compute the numerator of the Sampson error
    numerator = torch.einsum('bnki,bntki->bntk', x2_h, Fx1)  # Shape: (B, N, T, K)

    # Compute the denominator of the Sampson error
    a = Fx1[..., 0]**2 + Fx1[..., 1]**2  # Shape: (B, N, T, K)
    b = FTx2[..., 0]**2 + FTx2[..., 1]**2  # Shape: (B, N, T, K)

    if squared:
        denominator = a + b
        sampson_error = numerator**2 / denominator.clamp(min=1e-6)  # Avoid division by zero
    else:
        denominator = torch.sqrt(a + b)  # Shape: (B, N, T, K)
        sampson_error = torch.abs(numerator) / denominator.clamp(min=1e-6)  # Avoid division by zero
    return sampson_error


def calc_sampson_errors_kpts(x1, x2, data, batch_ind):
    """
    Calculate the Sampson errors for all pair of keypoints.
    Input:
        x1: torch.Tensor of shape (N, 2)
        x2: torch.Tensor of shape (N, 2)
        data: dict with many things
        batch_ind: int
    Output:
        sampson_errors: torch.Tensor of shape (N)
    """
    T_01 = data["T_0to1"]
    K_0 = data["view0"]["camera"]
    K_1 = data["view1"]["camera"]
    F = F_from_data(T_01, K_0, K_1)[batch_ind]
    sampson_errors = get_sampson_error(x1, x2, F)
    return sampson_errors


def get_sampson_error(x1, x2, F):
    """
    Compute the Sampson error between two sets of keypoints x1, x2 given the fundamental matrix F.
    Args:
        x1: torch.Tensor of shape (N, 2) with image keypoints
        x2: torch.Tensor of shape (N, 2) with image keypoints
        F: torch.Tensor of shape (3, 3) with fundamental matrices.
    Returns:
        error: torch.Tensor of shape (N) with the Sampson error for each pair of keypoints.
    """
    N, _ = x1.shape
    dt = x1.dtype
    x1_h = torch.cat([x1, torch.ones((N, 1), device=x1.device, dtype=dt)], dim=1)  # Shape: (N, 3)
    x2_h = torch.cat([x2, torch.ones((N, 1), device=x2.device, dtype=dt)], dim=1)  # Shape: (N, 3)

    # Extract the relevant fundamental matrices based on img_inds_x1 and img_inds_x2

    # Compute Fx1 and F^T*x2
    Fx1 = torch.matmul(F, x1_h.T)  # Shape: (3, 3) x (3, N) -> (3, N)
    FTx2 = torch.matmul(F.T, x2_h.T)  # Shape: (3, 3) x (3, N) -> (3, N)

    # Compute the numerator of the Sampson error
    numerator = torch.einsum('ni,in->n', x2_h, Fx1)  # Shape: (N)

    # Compute the denominator of the Sampson error
    a = Fx1[0]**2 + Fx1[1]**2  # Shape: (N)
    b = FTx2[0]**2 + FTx2[1]**2  # Shape: (N)
    denominator = torch.sqrt(a + b)  # Shape: (N)

    # Compute the Sampson error
    sampson_error = torch.abs(numerator) / denominator.clamp(min=1e-5)  # Avoid div. by zero (N)
    return sampson_error

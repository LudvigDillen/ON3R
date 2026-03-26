import torch
import torch.nn.functional as F
import numpy as np
import poselib


def project_points(X3D, extrinsics, ones_h=None, keep_in_3D=False):
    """
    Project 3D points to 2D using the provided extrinsics.
    Args:
        X3D (torch.Tensor): Output tensor of shape (N, 3) containing 3D points.
        extrinsics (torch.Tensor): Extrinsics matrix of shape (T, 3, 4). (world to camera)
        ones_h (torch.Tensor, optional): Optional tensor of shape (N, 1) for homogeneous coordinates
                                         If None, a tensor of ones will be created.
    Returns:
        torch.Tensor: Projected 2D points of shape (N, T, 3) or (N, T, 2).
    """
    if isinstance(X3D, np.ndarray):
        X3D = torch.tensor(X3D)
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.tensor(extrinsics)

    if ones_h is None:
        ones_h = torch.ones(X3D.shape[0], 1, device=X3D.device, dtype=X3D.dtype)
    X3D_h = torch.cat([X3D, ones_h], dim=-1)  # (N, 4)

    # Project points
    proj_pts = torch.einsum("tij,nj->nti", extrinsics, X3D_h)  # (N, T, 3)
    if keep_in_3D:
        return proj_pts

    normalized_proj_pts = proj_pts[..., :2] / proj_pts[..., 2:]  # (N, T, 2)
    return normalized_proj_pts


def normalized_pixels_to_kpts(kpts, inv_intrinsics, ones_h=None, return_3D=False):
    """
    Moving from pixel space to normalized space using the inverse intrinsics matrix.
    Args:
        kpts (torch.Tensor): Keypoints tensor of shape (N, T, 2).
        inv_intrinsics (torch.Tensor): Inverse intrinsics matrix of shape (T, 3, 3).
        ones_h (torch.Tensor, optional): Optional tensor of shape (N, T, 1) for homog. coordinates.
                                         If None, a tensor of ones will be created.
        return_3D (bool): Whether to return the full 3D coordinates (including z=1).
    Returns:
        torch.Tensor: Normalized keypoints of shape (N, T, 2).
    """
    if ones_h is None:
        N, T = kpts.shape[0], kpts.shape[1]
        ones_h = torch.ones((N, T, 1), device=kpts.device, dtype=kpts.dtype)
    kpts_h = torch.cat([kpts, ones_h], dim=-1)  # (N, T, 3)
    kpts_h_K_norm = torch.einsum('tij,ntj->nti', inv_intrinsics, kpts_h)  # (N, T, 3)
    if return_3D:
        return kpts_h_K_norm  # (N, T, 3)
    return kpts_h_K_norm[..., :2]  # (N, T, 2)


# NOTE: This function is the same as above, but for clarity, we keep them separate.
def normalized_kpts_to_pixels(kpts, intrinsics, ones_h=None):
    """
    Moving from normalized space to pixel space using the intrinsics matrix.
    Args:
        kpts (torch.Tensor): Keypoints tensor of shape (N, T, 2).
        intrinsics (torch.Tensor): Intrinsics matrix of shape (T, 3, 3).
        ones_h (torch.Tensor, optional): Optional tensor of shape (N, T, 1) for homog. coordinates.
                                         If None, a tensor of ones will be created.
    Returns:
        torch.Tensor: Normalized keypoints of shape (N, T, 2).
    """
    if ones_h is None:
        N, T = kpts.shape[0], kpts.shape[1]
        ones_h = torch.ones((N, T, 1), device=kpts.device, dtype=kpts.dtype)
    kpts_h = torch.cat([kpts, ones_h], dim=-1)  # (N, T, 3)
    kpts_h_px = torch.einsum('tij,ntj->nti', intrinsics, kpts_h)  # (N, T, 3)
    return kpts_h_px[..., :2]  # (N, T, 2)


def pixels_from_3D(X3D, extrinsics, intrinsics, ones_h=None):
    """
    Project 3D points to 2D pixels using the provided intrinsics and extrinsics.
    Args:
        X3D (torch.Tensor): Output tensor of shape (N, 3) containing 3D points.
        extrinsics (torch.Tensor): Extrinsics matrix of shape (T, 3, 4). (world to camera)
        intrinsics (torch.Tensor): Intrinsics matrix of shape (T, 3, 3).
        ones_h (torch.Tensor, optional): Optional tensor of shape (N, 1) for homog. coordinates.
                                         If None, a tensor of ones will be created.
    Returns:
        torch.Tensor: Projected 3D in pixels of shape (N, T, 2).
    """
    if isinstance(X3D, np.ndarray):
        X3D = torch.tensor(X3D)
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.tensor(extrinsics)
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.tensor(intrinsics)

    if ones_h is None:
        ones_h = torch.ones(X3D.shape[0], 1, device=X3D.device, dtype=X3D.dtype)
    normalized_2D = project_points(X3D, extrinsics, ones_h=ones_h)                  # (N, T, 2)
    ones_tuple_h = ones_h[:, None, :].repeat(1, extrinsics.shape[0], 1)             # (N, T, 1)
    pixels_2D = normalized_kpts_to_pixels(normalized_2D, intrinsics, ones_tuple_h)  # (N, T, 2)
    return pixels_2D


def bias_loss(bias, extrinsics, r_K_norm, pred_mask, ones_h=None):
    """
    Compute the mean error of the projection of the bias to the 2D image planes.

    Args:
        bias (torch.Tensor): Tensor of shape (S, 3) containing the starting point.
        extrinsics (torch.Tensor): Extrinsics matrix of shape (T, 3, 4). (world to camera)
        intrinsics (torch.Tensor): Intrinsics matrix of shape (T, 3, 3).
        r_K_norm (torch.Tensor): Tensor of shape (N, T, 2) containing the matches.
        pred_mask (torch.Tensor): Tensor of shape (N, T) containing the mask for valid matches.
        ones_h (torch.Tensor, optional): Optional tensor of shape (S, 1) for homog. coordinates.
                                         If None, a tensor of ones will be created.
    Returns:
        torch.Tensor: Mean error of the projection (S).
    """
    if ones_h is None:
        ones_h = torch.ones(bias.shape[0], 1, device=bias.device, dtype=bias.dtype)
    proj_b = project_points(bias, extrinsics, ones_h=ones_h)  # (S, T, 2)
    denom = (pred_mask[..., None].sum(0)).clamp(min=1)     # (T, 1)
    rm = (r_K_norm * pred_mask[..., None]).sum(0) / denom  # (T, 2)
    #errors = (proj_b - rm[None]).norm(dim=-1).mean(dim=1)
    weights = (denom / denom.sum()).swapdims(0, 1)  # (T, 1)
    errors = (weights * (proj_b - rm[None]).norm(dim=-1)).mean(dim=1)
    return errors


def get_poselib_camera_refs(data, batch_ind):
    poselib_cameras = []
    for key in data.keys():
        poselib_cameras.append(data[key]['view1']['camera'][batch_ind].cpu())
    return poselib_cameras


def get_poselib_camera_query(data, batch_ind):
    first_key = list(data.keys())[0]
    assert first_key == 'query_to_ref_0', 'query_to_ref_0 should be the first key'
    poselib_camera_query = data[first_key]['view0']['camera'][batch_ind].cpu()
    return poselib_camera_query


def get_query_camera_estimate(poselib_cam_query, pts_2d, pts_3d, max_reproj_error=16.0, return_info=False):
    """
    Compute the camera pose estimate using poselib.

    Args:
        poselib_cam_query: Poselib camera object for the query image.
        pts_2d: 2D points in pixel coordinates.
        pts_3d: 3D points in world coordinates.
        max_reproj_error: Maximum reprojection error allowed.
        return_info: Whether to return additional information.
    Returns:
        pose: Estimated camera pose.
        info: Additional information (if return_info is True).
    """
    if isinstance(pts_2d, torch.Tensor):
        pts_2d = pts_2d.detach().cpu().numpy()
    if isinstance(pts_3d, torch.Tensor):
        pts_3d = pts_3d.detach().cpu().numpy()
    pose, info = poselib.estimate_absolute_pose(pts_2d, pts_3d, poselib_cam_query.to_cameradict(),
                                                {'max_reproj_error': max_reproj_error}, {})
    if return_info:
        return pose, info
    return pose


def get_absolute_pose_error(pose_est, data, batch_ind, scene=""):
    T_query = get_pose_query_w2cam(data, batch_ind)
    R_gt = T_query[:, :3]
    t_gt = T_query[:, 3]
    t_est_world = -pose_est.R.T @ pose_est.t
    t_gt_world = -R_gt.T @ t_gt
    pose_error_R = get_pose_error_R(pose_est.R, R_gt)
    pose_error_t = get_absolute_pose_error_t(t_est_world, t_gt_world)

    # Compensate for the approximate scale of the scene for MegaDepth
    if scene == "0015":
        pose_error_t *= 16
    elif scene == "0022":
        pose_error_t *= 8

    return pose_error_R, pose_error_t


def get_pose_query_w2cam(data, batch_ind, mode="ndarray", device="cpu"):
    """
    The pose is in the form of a 3x4 matrix. It is world-to-camera.
    """
    first_key = list(data.keys())[0]
    assert first_key == 'query_to_ref_0', 'query_to_ref_0 should be the first key'
    R_query = data[first_key]['view0']['T_w2cam'].R[batch_ind].cpu()
    t_query = data[first_key]['view0']['T_w2cam'].t[batch_ind].cpu()
    T_query = np.concatenate((R_query, t_query[:, None]), axis=1)
    if mode == "tensor":
        return torch.from_numpy(T_query).to(device)
    return T_query


def get_pose_error_R(R_est, R_gt):
    """
    Compute the rotation error between estimated and ground truth rotations.
    Args:
        R_est (torch.Tensor/np.ndarray): Estimated rotation matrix of shape (3, 3).
        R_gt (torch.Tensor/np.ndarray): Ground truth rotation matrix of shape (3, 3).
    """
    if isinstance(R_est, torch.Tensor):
        R_est = R_est.cpu().numpy()
    if isinstance(R_gt, torch.Tensor):
        R_gt = R_gt.cpu().numpy()
    argument = np.clip((np.trace(R_est.T @ R_gt) - 1) / 2, -1, 1)
    r_dist = np.arccos(argument) * (180 / np.pi)
    return r_dist


def get_absolute_pose_error_t(t_est, t_gt):
    """
    Compute the absolute translation error between estimated and ground truth translations.
    Note, input -R.T @ t if the pose is world-to-camera.
    Args:
        t_est (torch.Tensor/np.ndarray): Estimated translation vector (in the wcs) of shape (3,).
        t_gt (torch.Tensor/np.ndarray): Ground truth translation vector (in the wcs) of shape (3,).
    """
    if isinstance(t_est, torch.Tensor):
        t_est = t_est.cpu().numpy()
    if isinstance(t_gt, torch.Tensor):
        t_gt = t_gt.cpu().numpy()
    t_dist = np.linalg.norm(t_est - t_gt)
    return t_dist


# NOTE: This function below is taken from PyTorch3D
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def get_mean_cam_center(extrinsics):
    dt = extrinsics.dtype
    dev = extrinsics.device
    tuple_length = extrinsics.shape[0]
    mean_cam_center = torch.zeros((3, 1), dtype=dt, device=dev)
    for j in range(tuple_length):
        R = extrinsics[j, :, :3]
        t = extrinsics[j, :, 3:]
        cam_center = -R.t() @ t
        mean_cam_center += cam_center / tuple_length
    return mean_cam_center


def get_median_cam_center(extrinsics):
    dt = extrinsics.dtype
    dev = extrinsics.device
    tuple_length = extrinsics.shape[0]
    cam_centers = torch.zeros((tuple_length, 3, 1), dtype=dt, device=dev)
    for j in range(tuple_length):
        R = extrinsics[j, :, :3]
        t = extrinsics[j, :, 3:]
        cam_center = -R.t() @ t
        cam_centers[j] = cam_center
    median_cam_center = cam_centers.quantile(q=0.5, dim=0)
    return median_cam_center


def get_scene_scaler(extrinsics):
    dt = extrinsics.dtype
    dev = extrinsics.device
    tuple_length = extrinsics.shape[0]
    cam_centers = torch.zeros((tuple_length, 3, 1), dtype=dt, device=dev)
    for j in range(tuple_length):
        R = extrinsics[j, :, :3]
        t = extrinsics[j, :, 3:]
        cam_center = -R.t() @ t
        cam_centers[j] = cam_center

    dists = torch.norm(cam_centers[None, :, :, 0] - cam_centers[:, None, :, 0], dim=2)  # (T, T)
    dists_self_inf = dists + torch.eye(tuple_length, dtype=torch.float32, device=dev) * 1e6
    median_min_dist = dists_self_inf.amin(0).quantile(q=0.5, dim=0)
    clipped_median_min_dist = torch.clip(median_min_dist/3, min=0.1, max=25.0).item()
    scene_scaler = 1.0 / clipped_median_min_dist
    print(f"Median min dist for centering: {scene_scaler:.4f}")
    return scene_scaler


def center_extrinsics(extrinsics, mean_cam_center):
    extrinsics_cententered = extrinsics.clone()
    for j in range(extrinsics.shape[0]):
        R = extrinsics[j, :, :3]
        t = extrinsics[j, :, 3:]
        new_t = t + R @ mean_cam_center
        extrinsics_cententered[j, :, 3:] = new_t
    return extrinsics_cententered


def move_pose_back_to_original_cs(poselib_pose_est_query, mean_cam_center_db):
    t_cent = poselib_pose_est_query.t.reshape(3, 1).copy()
    R_cent = poselib_pose_est_query.R.copy()
    t_orig = t_cent - R_cent @ mean_cam_center_db.cpu().numpy()
    poselib_pose_est_query.t = t_orig.reshape(3)
    return poselib_pose_est_query


def scale_back(model):
    pose_sc = model.poselib_pose_est_query
    t_sc = pose_sc.Rt[:, 3]
    R = pose_sc.Rt[:, :3]
    sc = model.sc.item()
    b = model.b.cpu().numpy()
    t = (t_sc + (1 - sc) * R @ b) / sc
    pose = np.hstack((R, t[:, None]))
    model.poselib_pose_est_query.Rt = pose.copy()
    model.extrinsics = model.old_extrinsics.clone().detach()  # reset extrinsics to original (unscaled) ones for BA and plotting
    return model

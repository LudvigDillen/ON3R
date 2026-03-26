import torch
import scipy
import numpy as np
from concave_hull import concave_hull
from matplotlib.path import Path
import re

from gluefactory.settings import DATA_PATH, DATASET
from . import geometry as geo


def compute_keypoint_recall(warped_query_kpts, ref_kpts, pred_mask, scalers_ref=None,
                            thresholds=[5, 10, 25, 50]):
    """
    Compute recall@thresholds px between warped keypoints and reference keypoints.

    Parameters
    ----------
    warped_query_kpts : Tensor
        Shape (N, T, 2). MLP output containing (x, y).
    ref_kpts : Tensor 
        Shape (N, T, 2). Reference keypoints. (not normalized)
    pred_mask : Tensor
        Boolean mask of shape (N, T) that indicates which keypoints to use.
    scalers_ref: Tensor
        (T, 2). Factor to scale the distance back to pixels.
    thresholds : List
        List of thresholds to compute recall@.

    Returns
    -------
    recalls : Tensor
        Shape (N_thresholds). recall@thresholds.
    """
    if scalers_ref is not None:
        warped_query_kpts = warped_query_kpts*scalers_ref[None]  # (N, T, 2)
    dists = torch.norm((warped_query_kpts - ref_kpts)[pred_mask], p=2, dim=1)  # (N_active_samples)
    thresholds = torch.tensor(thresholds, device=dists.device, dtype=dists.dtype)  # N_thresholds
    recalls = (dists[None] < thresholds[:, None]).float().mean(1)
    return recalls


def get_keypoint_matches(pred, matches_as_tracks, batch_ind, ratio_train):
    """
    Get training and validation keypoints.

    Parameters
    ----------
    pred : Dict
        Dictionary containing the predictions.
    matches_as_tracks : Tensor
        Shape (K, T+1). Matches in track format.
    batch_ind : int
        Batch index.
    ratio_train : float
        Ratio of keypoints to use for training.

    Returns
    -------
    query_kpts_train : Tensor
        Shape (K'', 2). Training query keypoints.
    query_kpts_val : Tensor
        Shape (K''', 2). Validation query keypoints.
    ref_kpts_train : Tensor
        Shape (K'', T, 2). Training reference keypoints.
    ref_kpts_val : Tensor
        Shape (K''', T, 2). Validation reference keypoints.
    """

    dev = matches_as_tracks.device
    dt = pred['query_to_ref_0']['keypoints0'][batch_ind].dtype
    n_query_kpts, tuple_plus_1 = matches_as_tracks.shape  # K, T+1
    tuple_length = tuple_plus_1 - 1  # T

    valid_tracks_mask = (matches_as_tracks != -1).sum(1) > 1  # K
    valid_query_kpts = torch.arange(n_query_kpts, device=dev)[valid_tracks_mask]  # K'
    n_valid_tracks = valid_query_kpts.size(0)

    perm = torch.randperm(n_valid_tracks, device=dev)
    k = int(ratio_train * n_valid_tracks)
    train_samples = perm[:k]  # K''
    val_samples = perm[k:]  # K'''

    ref_kpts = torch.zeros((n_query_kpts, tuple_length, 2), device=dev, dtype=dt)
    for key_ind, key in enumerate(pred.keys()):
        if key_ind == 0:  # it's the same query kpts for all keys => we can just use the first key
            query_kpts = pred['query_to_ref_0']['keypoints0'][batch_ind]  # (K, 2)
        ref_kpts[:, key_ind] = pred[key]['keypoints1'][batch_ind]  # (K, 2)

    matches_as_tracks_kpts = torch.full((n_query_kpts, tuple_length+1, 2), -1,
                                        dtype=ref_kpts.dtype, device=dev)
    tracks_mask = matches_as_tracks != -1  # (K, T+1)
    matches_as_tracks_kpts[:, 0] = query_kpts

    indices = matches_as_tracks[:, 1:].unsqueeze(-1) # (K, T) -> (K, T, 1)
    indices = indices.expand(-1, -1, ref_kpts.size(-1))  # (K, T, 1) -> (K, T, 2)
    indices_ = indices.clone()
    indices_[indices == -1] = 0  # cannot handle -1 in gather
    matches_as_tracks_kpts[:, 1:] = torch.gather(ref_kpts, dim=0, index=indices_)

    # Set invalid kpts to -1
    matches_as_tracks_kpts[~tracks_mask] = -1

    valid_tracks = matches_as_tracks_kpts[valid_tracks_mask]
    train_kpts = valid_tracks[train_samples]  # (K'', T+1, 2)
    val_kpts = valid_tracks[val_samples]  # (K''', T+1, 2)

    query_kpts_train = train_kpts[:, 0]   # (K'', 2)
    query_kpts_val   = val_kpts[:, 0]     # (K''', 2)
    ref_kpts_train   = train_kpts[:, 1:]  # (K'', T, 2)
    ref_kpts_val     = val_kpts[:, 1:]    # (K''', T, 2)

    return query_kpts_train, query_kpts_val, ref_kpts_train, ref_kpts_val


def step_and_break_check(output, model, old_xyz, old_recall, epoch, scheduler, lr_end,
                         ref_kpts_train, step_check_interval=50):
    break_check = False
    thresholds = [1, 2, 5, 10, 25, 50]
    if old_xyz is not None:
        if epoch % step_check_interval == 0:
            xyz = output[:, :3]  # (N, 3)

            lr_last = scheduler.get_last_lr()[0]
            warpedkpts = geo.pixels_from_3D(xyz, model.extrinsics, model.intrinsics, model.ones_h)
            recall = compute_keypoint_recall(warpedkpts, ref_kpts_train, model.pred_mask,
                                             scalers_ref=None, thresholds=thresholds).mean()
            if epoch > 100 and recall < old_recall:
                if lr_last < lr_end:
                    break_check = True
                scheduler.step()
            old_xyz = xyz
            old_recall = recall
    else:
        old_xyz = output[:, :3]  # (N, 3) (old_warped_kpts is actually the xyz)
        warpedkpts = geo.pixels_from_3D(old_xyz, model.extrinsics, model.intrinsics, model.ones_h)
        old_recall = compute_keypoint_recall(warpedkpts, ref_kpts_train, model.pred_mask,
                                             scalers_ref=None, thresholds=thresholds).mean()
    return old_xyz, break_check, scheduler, old_recall


def residuals_to_confidence(residuals, min_residual=0.010, max_residual=20.0,
                            image_norm=1200, mode='division', norm_method="percentile"):
    if mode == 'division':
        confidence =  min_residual/torch.clamp(residuals, min=min_residual, max=1.0)
    elif mode == 'linear':
        scale_factor = image_norm / torch.sqrt(torch.tensor(2, dtype=residuals.dtype,
                                                            device=residuals.device))
        confidence = torch.clamp(1.0 - scale_factor/max_residual * residuals, min=0, max=1)
    else:
        raise ValueError(f"Mode {mode} not recognized.")

    confidence_norm = confidences_to_01(confidence, mode=norm_method)
    return confidence_norm


def confidences_to_01(confidences, mode="percentile"):
    N = confidences.size(0)
    if N == 0:
        return confidences

    if mode == "min-max":
        c_min, c_max = confidences.min(), confidences.max()
        if c_min == c_max:
            confidences_01 = torch.full_like(confidences, 0.5)
        else:
            confidences_01 = (confidences - c_min) / (c_max - c_min)
    elif mode == "percentile":
        # Define lower and upper percentiles
        p_low = 0.025
        p_high = 0.975

        # Compute the corresponding quantiles
        lower_bound = torch.quantile(confidences, p_low)
        upper_bound = torch.quantile(confidences, p_high)
        if lower_bound == upper_bound:
            confidences_01 = torch.full_like(confidences, 0.5)
        else:
            # Clip the data
            confidence_clipped = torch.clamp(confidences, min=lower_bound, max=upper_bound)
            confidences_01 = (confidence_clipped - lower_bound) / (upper_bound - lower_bound)
    else:
        raise ValueError(f"Invalid normalization method: {mode}")

    confidences_01 = torch.clamp(confidences_01, min=0.0, max=1.0)
    return confidences_01


def bilinear_interpolation(image, pts_2d):
    """
    Interpolate RGB `image` at floating-point 2D coordinates `pts_2d` using.
    image shape: (H, W, 3)
    pts_2d shape: (N, 2)
    Returns:
        interpolated_colors: (N, 3) array
    """
    # Prepare output array
    N = pts_2d.shape[0]
    interpolated_colors = np.empty((N, 3), dtype=image.dtype)

    # SciPy’s map_coordinates wants coordinates in the form [row_coords, col_coords]
    # i.e., y first, then x
    y = pts_2d[:, 1]
    x = pts_2d[:, 0]

    for c in range(3):
        interpolated_colors[:, c] = scipy.ndimage.map_coordinates(
            image[..., c],
            [y, x],         # [row-coords, col-coords]
            order=1,        # 1 => bilinear interpolation
            mode='nearest'  # How to handle boundaries
        )

    return interpolated_colors


def get_concave_hull_2d(pts_2d: np.ndarray) -> np.ndarray:
    """
    Compute the concave hull of a set of 2D points.
    Parameters
    ----------
    pts_2d : np.ndarray
        Array of shape (N, 2) containing the 2D points.
    Returns
    -------
    xy_concave_hull : np.ndarray
        Array of shape (M, 2) containing the points inside the concave hull.
    """
    poly_verts = concave_hull(pts_2d)

    # 2. build a Path object for inclusion tests
    path = Path(poly_verts)

    # # 3. integer grid in the bounding box
    xmin, ymin = np.floor(poly_verts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(poly_verts.max(axis=0)).astype(int)

    # # build all integer (x,y) in that box
    X, Y = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    grid = np.vstack((X.ravel(), Y.ravel())).T

    # # 4. test which grid points lie inside the polygon
    mask = path.contains_points(grid)
    xy_concave_hull = grid[mask]
    return xy_concave_hull


def print_residuals_in_gt_cam(data, batch_ind, model, X3D, q_kpts):
    cam_pose_query_gt = geo.get_pose_query_w2cam(data, batch_ind, mode="tensor", device=model.device)
    proj_pts = geo.project_points(X3D, cam_pose_query_gt[None], model.ones_h)[:, 0]  # (N, 2)
    inv_K = model.inv_intrinsics_query[None]
    normalized_q_kpts = geo.normalized_pixels_to_kpts(q_kpts[:, None], inv_K)[:, 0]  # (N, 2)
    #residuals_scaled = model.cauchy_loss(proj_pts, normalized_q_kpts)  # (N)
    #print(f"\nLoss GT cam:      {residuals_scaled.mean().item():.1e}")
    residuals_gt_cam = torch.linalg.norm((proj_pts - normalized_q_kpts), dim=1)
    print(f"Residuals GT cam: {residuals_gt_cam.mean().item():.1e}")
    return None


def grad_norm(loss_term, params, norm_type=2):
    grads = torch.autograd.grad(
        outputs=loss_term,
        inputs=[p for p in params if p.requires_grad],
        retain_graph=True,          # so we can call it twice
        allow_unused=True           # if some params are frozen / not used
    )
    # Flatten all gradients into a single vector ‖g‖₂
    return torch.linalg.vector_norm(
        torch.stack([g.norm(norm_type) for g in grads if g is not None]),
        ord=norm_type
    )

def get_img_path(batch_ind, data_inner, view):
    parent_dir = get_parent_dir()
    image_name = data_inner[f"view{view}"]["name"][batch_ind]
    scene = data_inner["scene"][batch_ind]
    if DATASET == "megadepth":
        image_path = parent_dir + scene + "/images/" + image_name
    elif DATASET == "scannet1500":
        image_path = parent_dir + scene + "/color/" + image_name
    elif DATASET == "cambridge":
        seq_numbers = re.findall(r'-seq(\d+)-', data_inner["name"][batch_ind])
        assert len(seq_numbers) == 2, f"Expected exactly two seq numbers"
        assert seq_numbers[0] == seq_numbers[1], f"Expected same seq number for query and ref"
        seq = "seq" + seq_numbers[0]
        image_path = parent_dir + scene + "/" + seq + "/" + image_name
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")
    return image_path


def get_parent_dir():
    if DATASET == "megadepth":
        # For MegaDepth, we need to load the images from the parent directory
        # and the scene name is part of the image name.
        parent_dir = str(DATA_PATH) + "/megadepth/Undistorted_SfM/"
    elif DATASET == "scannet1500" or DATASET == "cambridge":
        # For ScanNet1500, we need to load the images from the parent directory
        # and the scene name is part of the image name.
        parent_dir = str(DATA_PATH) + f"/{DATASET}/"
    else:
        raise ValueError(f"Unsupported dataset: {DATASET}")
    return parent_dir


def pred_to_query_kpts(pred, batch_ind):
    return pred['query_to_ref_0']['keypoints0'][batch_ind].cpu().numpy()  # (K, 2)


def pred_to_ref_kpts(pred, batch_ind):
    return np.stack([pred[key]['keypoints1'][batch_ind].cpu() for key in pred.keys()])  # (T, K, 2)


def pred_to_kpts(pred, batch_ind):
    query_kpts = pred_to_query_kpts(pred, batch_ind)  # (K, 2)
    ref_kpts = pred_to_ref_kpts(pred, batch_ind)      # (T, K, 2)
    return query_kpts, ref_kpts

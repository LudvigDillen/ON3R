import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata

from . import utils as on3ru
from ...visualization import viz2d
from . import general_utils as guti
from . import geometry as geo
from . import rerun_tools as rrt
from gluefactory.geometry.wrappers import Camera


def plot_proj_preds(img0, img1, query_kpts, ref_kpts, pred_ref_kpts, mode=None):
    """
    Plot query keypoints on image 0 and reference and predicted keypoints on image 1.
    Also draws lines between reference and predicted keypoints.

    Parameters:
    data (dict): Dictionary containing image data.
    query_kpts (Tensor): Query keypoints, shape (N, 2).
    ref_kpts (Tensor): Reference keypoints, shape (N, 2).
    pred_ref_kpts (Tensor): Predicted reference keypoints, shape (N, 2).
    mode (str): Suptitle for the plot. (training or validation)
    """
    if isinstance(query_kpts, torch.Tensor):
        query_kpts = query_kpts.detach().cpu().numpy()
    else:
        raise ValueError("query_kpts must be a torch.Tensor")
    if isinstance(ref_kpts, torch.Tensor):
        ref_kpts = ref_kpts.detach().cpu().numpy()
    else:
        raise ValueError("ref_kpts must be a torch.Tensor")
    if isinstance(pred_ref_kpts, torch.Tensor):
        pred_ref_kpts = pred_ref_kpts.detach().cpu().numpy()
    else:
        raise ValueError("pred_ref_kpts must be a torch.Tensor")

    # Normalize query keypoints for color mapping
    n_pts = len(query_kpts)
    colors = []
    if n_pts > 0:
        normalizer0 = query_kpts[:, 0].max() - query_kpts[:, 0].min()
        x_normalized = (query_kpts[:, 0] - query_kpts[:, 0].min()) / normalizer0
        normalizer1 = query_kpts[:, 1].max() - query_kpts[:, 1].min()
        y_normalized = (query_kpts[:, 1] - query_kpts[:, 1].min()) / normalizer1
        colors = np.stack([x_normalized, y_normalized, 0.5 * (x_normalized + y_normalized)], axis=1)

    # Plot image 0
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img0)
    if n_pts == 0:
        # Create an empty scatter plot with the desired label
        plt.scatter([], [], color='red', s=20, marker='o', label='Query Keypoints')
    for j in range(n_pts):
        plt.scatter(query_kpts[j, 0], query_kpts[j, 1], color=colors[j],
                    s=20, marker='o', label='Query Keypoints' if j == 0 else "")
    plt.title("Image 0: Query Keypoints")
    plt.legend(loc="upper right", fontsize=8)

    # Plot image 1
    plt.subplot(1, 2, 2)
    plt.imshow(img1)
    if n_pts == 0:
        # Create an empty scatter plot with the desired label
        plt.scatter([], [], color='blue', s=15, marker='s', label='Reference Keypoints')
        plt.scatter([], [], color='green', s=15, marker='x', label='Predicted Warps')
    for j in range(n_pts):
        # Draw line between reference and predicted keypoints
        plt.plot([ref_kpts[j, 0], pred_ref_kpts[j, 0]],
                 [ref_kpts[j, 1], pred_ref_kpts[j, 1]],
                 color=colors[j], linestyle='-', linewidth=1)

        # Plot reference keypoints
        plt.scatter(ref_kpts[j, 0], ref_kpts[j, 1], color=colors[j],
                    s=15, marker='s', label='Reference Keypoints' if j == 0 else "")

        # Plot predicted keypoints
        plt.scatter(pred_ref_kpts[j, 0], pred_ref_kpts[j, 1], color=colors[j],
                    s=15, marker='x', label='Predicted Warps' if j == 0 else "")

    plt.title("Image 1: Reference and Predicted Keypoints")
    if mode is not None:
        plt.suptitle(mode)
    plt.legend(loc="upper right", fontsize=8)
    plt.show()


def plot_2d_preds(
    data, model, query_kpts, ref_kpts, scaler_query, batch_ind, mode="train"):
    """
    This function consits of two plotting functions.
    Plot preds: Plotting predicted warp (projection of 3D points) on images vs ground truth.

    Parameters:
        data         : dictionary with the image data.
        model        : the network
        query_kpts   : query keypoints, shape (N, 2).
        ref_kpts     : reference keypoints, shape (N, 2).
        scaler_query : (width, height) for image 0.
        batch_ind    : batch index to use.
        mode         : mode for the plot. (train or val)
    Returns:
        None
    """
    assert mode in ["train", "val"], f"Invalid mode: {mode}. Use 'train' or 'val'."
    title_str = "Training Set" if mode == "train" else "Validation Set"

    # Normalize
    q_norm = query_kpts / scaler_query[None]  # (N, 2)
    # Do inference
    output = model(q_norm)  # (N, 2) -> (N, 4)
    # Extract output
    xyz = output[..., :3]  # (N, 3)
    pixels = geo.pixels_from_3D(xyz, model.extrinsics, model.intrinsics)  #(N, T, 2)

    pred_mask  = model.pred_mask  # (N, T)

    for i, key in enumerate(data.keys()):
        # Extract images for plotting
        img0 = data[key]['view0']['image'][batch_ind].permute(1, 2, 0).cpu().numpy()
        img1 = data[key]['view1']['image'][batch_ind].permute(1, 2, 0).cpu().numpy()

        pred_mask_img_i = pred_mask[:, i]
        query_kpts_i    = query_kpts[pred_mask_img_i]   # (N, 2)
        ref_kpts_i      = ref_kpts[pred_mask_img_i, i]  # (N, 2)
        pred_ref_kpts_i = pixels[pred_mask_img_i, i]    # (N, 2)

        # Plotting keypoints on images
        plot_proj_preds(img0, img1, query_kpts_i, ref_kpts_i, pred_ref_kpts_i, mode=title_str)


def warp_images(data, model, batch_ind, query_kpts_train, scaler_query, scalers_ref,
                interpolate=False, norm_method="percentile", noise_std=0.050, samples=int(1e6),
                plot_blended_image=True, model_type="on3r", use_confidence=False):
    """
    Warps image 0 into image 1's coordinate system.
    
    Parameters:
      data         : dictionary with the image data.
      model        : the warping network.
      batch_ind    : batch index to use.
      scaler_query : (width, height) for image 0.
      scalers_ref  : tensor of (width, height) for reference images.
      interpolate  : whether to interpolate the images.
      norm_method  : method to normalize the confidence values.
      noise_std    : standard deviation of the noise to add to the query keypoints.
      samples      : number of samples to use for the dense grid.
      plot_blended_image : whether to plot the blended image.
      model_type   : type of model used for warping. (e.g., "on3r").
    
    Note:
      - The network outputs normalized coordinates (and a confidence value in channel 3).
      - Currently, we use a nearest neighbor assignment and simple accumulation.
      - TODO: Switch to bilinear interpolation for subpixel accuracy.
      - TODO: Improve interpolation (e.g., using more advanced inpainting).
    """
    assert model_type in ["on3r"], f"Invalid model type: {model_type}. Use 'on3r'."
    if plot_blended_image:
        assert use_confidence, "Blended image plotting requires confidence values."
    width_query, height_query = scaler_query

    q_train_norm = query_kpts_train / scaler_query[None]  # (n_train, 2)
    repeats = int(torch.ceil(torch.tensor(samples / q_train_norm.shape[0])))
    q_train_extra = q_train_norm.repeat(repeats, 1)[:samples]  # (samples, 2)
    # NOTE: it is not a dense grid, it is a randomized grid.

    # Create a mask for values out of the [-2.5, 2.5] range
    noise = torch.randn_like(q_train_extra)
    mask = (noise < -2.5) | (noise > 2.5)

    # While there are out-of-range values, re-sample only those
    while mask.any():
        noise[mask] = torch.randn_like(noise[mask])
        mask = (noise < -2.5) | (noise > 2.5)

    dense_grid_img0 = q_train_extra + noise * noise_std  # (samples, 2)

    dense_grid0_pxs = np.round((dense_grid_img0*scaler_query[None]).cpu().numpy()).astype(np.int32)
    dense_grid0_pxs[:, 0] = np.clip(dense_grid0_pxs[:, 0], 0, width_query.cpu() - 1)
    dense_grid0_pxs[:, 1] = np.clip(dense_grid0_pxs[:, 1], 0, height_query.cpu() - 1)
    col_grid = dense_grid0_pxs[:, 0]
    row_grid = dense_grid0_pxs[:, 1]
    # Same query image for all views
    img0 = data['query_to_ref_0']['view0']['image'][batch_ind].permute(1, 2, 0).cpu().numpy()
    colors_img0 = img0[row_grid, col_grid]  # shape: (n_pts, 3)

    # Run the model on the grid.
    # Output shape: (n_pts, T * 3) which we reshape to (n_pts, T, 3)
    with torch.no_grad():
        output0 = model(dense_grid_img0)  # (N, 2) -> (N, 4)
        xyz     = output0[:, :3]  # (N, T, 3)
        pred_px = geo.pixels_from_3D(xyz, model.extrinsics, model.intrinsics)  # (N, T, 2)

    for i, key in enumerate(data.keys()):
        # Extract images for plotting
        # NOTE: img0 will be (height, width, channels) and img1 will be (height, width, channels)
        img1 = data[key]['view1']['image'][batch_ind].permute(1, 2, 0).cpu().numpy()

        # Get the reference image dimensions (format: (width, height))
        sr = scalers_ref[i]
        width_ref, height_ref = sr

        # Normalize confidence to [0, 1]
        if use_confidence:
            confidence = output0[:, 3]
            confidence_norm = on3ru.confidences_to_01(confidence, mode=norm_method)

        xy_i = pred_px[:, i]  # shape: (n_pts, 2) in normalized space
        # Convert to integer coordinates with nearest neighbor
        xy_i_int = np.round(xy_i.cpu().numpy()).astype(np.int32)
        # Clip coordinates to ensure they are within image bounds.
        col_coords = np.clip(xy_i_int[:, 0], 0, sr[0].cpu() - 1)
        row_coords = np.clip(xy_i_int[:, 1], 0, sr[1].cpu() - 1)

        # # Create a new image for img1 with a default white background.
        warped_img = np.ones_like(img1, dtype=np.float32)
        warped_img[:, width_ref:, :] = 0
        warped_img[height_ref:, :, :] = 0

        # NOTE: Using NumPy's advanced indexing, if multiple source pixels map to the same target,
        # the last assignment in the ordering will prevail.

        warped_img[row_coords, col_coords] = colors_img0
        if use_confidence:
            alpha_values = np.zeros_like(img1[:, :, 0], dtype=np.float32)
            alpha_values[row_coords, col_coords] = confidence_norm.cpu().numpy()

        if interpolate:
            mask_assigned = np.zeros(img1.shape[:2], dtype=bool)   # mask to track assigned pixels
            mask_assigned[row_coords, col_coords] = True
            # The images may be padded with zeros to the right or bottom.
            # Create masks for the valid pixel regions.
            mask1 = np.zeros_like(img1[:, :, 0], dtype=bool)
            mask1[:height_ref, :width_ref] = True

            # Identify holes: pixels inside the valid region (mask1) with no mapping.
            to_fill_in = (~mask_assigned) & mask1  # height_ref x width_ref
            holes_y, holes_x = np.where(to_fill_in)
            # Identify known pixels.
            known_y, known_x = np.where(mask_assigned)
            known_points = np.column_stack((known_y, known_x))
            # Only perform interpolation if there are known points and holes.
            if known_points.size > 0 and holes_y.size > 0:
                # Interpolate each color channel separately.q
                for c in range(3):
                    known_vals = warped_img[mask_assigned, c]
                    # Using 'nearest' interpolation; change method if needed.
                    interp_vals = griddata(known_points,
                                            known_vals,
                                            np.column_stack((holes_y, holes_x)),
                                            method='linear', fill_value=1)
                    warped_img[holes_y, holes_x, c] = interp_vals
                if use_confidence:
                    known_conf = alpha_values[mask_assigned]
                    interp_vals_conf = griddata(known_points,
                                                known_conf,
                                                np.column_stack((holes_y, holes_x)),
                                                method='linear', fill_value=0)
                    alpha_values[holes_y, holes_x] = interp_vals_conf
            suptitle = "Interpolated Warped Image 0 → Image 1 coords"
        else:
            suptitle = "Non-Interpolated Warped Image 0 → Image 1 coords"

        if use_confidence:
            alpha_values = np.clip(alpha_values, a_min=0.0, a_max=1.0)
        img0         = np.clip(img0, a_min=0.0, a_max=1.0)
        img1         = np.clip(img1, a_min=0.0, a_max=1.0)
        warped_img   = np.clip(warped_img, a_min=0.0, a_max=1.0)

        plt.figure(figsize=(12, 6))
        plt.suptitle(suptitle)

        if use_confidence:
            plt.subplot(2, 2, 1)
            plt.imshow(img0)
            plt.title("Image 0")

            plt.subplot(2, 2, 2)
            plt.imshow(img1)
            plt.title("Image 1")

            plt.subplot(2, 2, 3)
            plt.imshow(warped_img)
            plt.title("Warped Image 0 → Image 1 coords")

            plt.subplot(2, 2, 4)
            warped_img_alpha = np.dstack([warped_img, alpha_values])
            plt.imshow(warped_img_alpha)
            plt.title("Warped Image 0 → Image 1 coords (Confidence Weighted)")
        else:
            plt.subplot(1, 3, 1)
            plt.imshow(img0)
            plt.title("Image 0")

            plt.subplot(1, 3, 2)
            plt.imshow(img1)
            plt.title("Image 1")

            plt.subplot(1, 3, 3)
            plt.imshow(warped_img)
            plt.title("Warped Image 0 → Image 1 coords")

        plt.tight_layout()
        plt.show()

        if plot_blended_image:
            #plt.figure(figsize=(12, 6))
            #plt.title("Confidence Blended Warped Image 0 → Image 1 coords")
            #plt.imshow(warped_img_alpha, alpha=1.0)
            #img1_blended = np.dstack([img1, 1-alpha_values])
            #plt.imshow(img1_blended, alpha=1.0)

            plt.figure(figsize=(12, 6))
            blended_image = warped_img*alpha_values[..., None] + img1*(1 - alpha_values[..., None])
            plt.imshow(blended_image)
            plt.title("Confidence Blended Warped Image 0 → Image 1 coords")
            plt.show()


def plot_matches(pred, data, batch_ind, ref_ind, matches_as_tracks, max_sampson_error=10.0):
    """
    Plot matches as colors them by the Sampson error.

    pred : dictionary containing the predicted keypoints.
    data : dictionary containing the image data.
    batch_ind : batch index to use.
    ref_ind : index of the reference view.
    matches_as_tracks : tensor of matches as tracks. (K, T+1) in local inds.
    """
    kps0 = pred['keypoints0'][batch_ind]
    kps1 = pred['keypoints1'][batch_ind]

    queries = matches_as_tracks[:, 0]
    refs    = matches_as_tracks[:, 1+ref_ind]
    valid   = refs != -1
    queries_valid = queries[valid]
    refs_valid    = refs[valid]
    m_kpts_query = kps0[queries_valid]
    m_kpts_ref = kps1[refs_valid]
    plot_kpts_as_matches(data, batch_ind, m_kpts_query, m_kpts_ref, max_sampson_error)


def plot_kpts_as_matches(data, batch_ind, kpts0, kpts1, max_sampson_error=10.0, opacities=None):
    """
    Plot keypoints as matches and color them by the Sampson error.

    data : dictionary containing the image data.
    batch_ind : batch index to use.
    kpts0 : tensor of keypoints in view 0. (N, 2)
    kpts1 : tensor of keypoints in view 1. (N, 2)
    max_sampson_error : maximum Sampson error to use for color mapping.
    opacities : tensor of opacities for the matches. (N,) or None.
    """
    img0 = data['view0']['image'][batch_ind].permute(1, 2, 0).cpu().numpy()
    img1 = data['view1']['image'][batch_ind].permute(1, 2, 0).cpu().numpy()
    viz2d.plot_images([img0, img1])

    sampson_errors = guti.calc_sampson_errors_kpts(kpts0, kpts1, data, batch_ind)
    sampson_errors_clamped = torch.clamp(sampson_errors, max=max_sampson_error)
    se_in = torch.abs((sampson_errors_clamped / max_sampson_error - 1))
    colors = viz2d.cm_RdGn(se_in.cpu()).tolist()
    alphas = 1.0 if opacities is None else opacities.cpu().tolist()

    viz2d.plot_matches(kpts0.cpu(), kpts1.cpu(), color=colors, a=alphas, ps=3, lw=0.2)
    plt.show()


def plot_3d_scene(model, query_kpts, ref_kpts, poselib_cam_query, xyz, data=None, batch_ind=None,
                  img_query=None, imgs_refs=None):
    if model.poselib_pose_est_query is not None:
        cam_pose_est_query = model.poselib_pose_est_query.Rt.astype(np.float32)
    elif data is None:  # If no data is provided, we have not gt pose, but we want to plt something.
        cam_pose_est_query = geo.get_query_camera_estimate(
            poselib_cam_query, query_kpts, xyz, max_reproj_error=16.0)  # w2cam
    else:
        cam_pose_est_query = None

    T = model.tuple_length
    ref_kpts_used = [ref_kpts[:, i][model.pred_mask[:, i]].detach().cpu().numpy() for i in range(T)]
    if data is not None:
        rrt.plot_3d_scene(xyz, query_kpts, model.extrinsics, cam_pose_est_query, data, batch_ind,
                          ref_kpts_used, plot_projected_3d_points=True, pred_mask=model.pred_mask,
                          new_cam_center_db=model.new_cam_center_db, sc=model.sc.item(),
                          b=model.b.cpu().numpy())
        # on3ru.print_residuals_in_gt_cam(data, batch_ind, model, xyz, query_kpts)
    else:
        poslib_cam_intrinsics_refs = []
        for i in range(T):
            poslib_cam_intrinsics_refs.append(Camera.from_calibration_matrix(
                model.intrinsics[i].cpu().numpy()))
        poselib_cam_query_ = poselib_cam_query.to(torch.float32)
        rrt.plot_3d_scene_no_data(
            xyz, query_kpts, model.extrinsics, cam_pose_est_query, ref_kpts_used,
            plot_projected_3d_points=True, pred_mask=model.pred_mask, img_query=img_query,
            imgs_refs=imgs_refs, poselib_cam_intrinsics_q=poselib_cam_query_,
            poselib_cam_intrinsics_refs=poslib_cam_intrinsics_refs)
    return None

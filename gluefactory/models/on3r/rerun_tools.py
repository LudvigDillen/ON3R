import numpy as np
import torch
from typing import Optional
import rerun as rr

from . import utils as on3ru
from . import geometry as geo


def log_3d_points(pts_3d: np.ndarray, parent_path: str = "world",
                  colors: Optional[np.ndarray] = None, filt: bool = False,
                  title: Optional[str] = "3d_points", radii: Optional[np.ndarray] = None) -> None:
    """
    Log a set of 3D points into Rerun under `parent_path`.
    """
    # pts_3d: shape (N, 3), in your world coordinate frame
    colors = colors if colors is not None else np.array([200, 50, 50])
    # Make radii depend on the number of points
    if filt:
        reasonable_dist = np.quantile(pts_3d, q=0.8) - np.quantile(pts_3d, q=0.2)
        lower_limit = np.quantile(pts_3d, q=0.02)
        upper_limit = np.quantile(pts_3d, q=0.98)
        lower_limit_soft = lower_limit - reasonable_dist
        upper_limit_soft = upper_limit + reasonable_dist
        mask = (pts_3d >= lower_limit_soft).all(axis=1) & (pts_3d <= upper_limit_soft).all(axis=1)
        pts_3d = pts_3d[mask]
        # n_removed  = mask.shape[0] - pts_3d.shape[0]
        # print(f"Rerun: Filtering 3D points: removed {n_removed}")
    n_pts = pts_3d.shape[0]
    radii = np.clip(0.05 - 0.0005*(n_pts**(1/3)), 0.003, 0.04) if radii is None else radii
    title = title if title is not None else "3d_points"
    rr.log(
        f"{parent_path}/{title}",
        rr.Points3D(pts_3d, colors=colors, radii=radii)
    )


def log_keypoints(
    pts_2d: np.ndarray,
    parent_path: str,
    extra_string: str = "",
    colors: Optional[np.ndarray] = None
):
    """
    Overlay 2D keypoints on an image in the world.
    """
    if colors is None:
        colors = np.array([200, 50, 50])
    # We assume pts_2d is shape (N, 2) in image pixel coordinates:
    rr.log(
        f"{parent_path}/image/keypoints{extra_string}",
        rr.Points2D(pts_2d, colors=colors, radii=4.0)
    )


def visualize_image(image: np.ndarray, name: str, opacity: Optional[float] = None) -> None:
    """
    Visualize image in Rerun.

    @param image The image.
    @param name Image name.
    @param opacity Image opacity (0-1).
    """
    image_clipped = np.clip(image, 0, 1)
    rr.log(f'{name}/image', rr.Image(image_clipped, opacity=opacity))


def visualize_camera(cam, R, t, name) -> None:
    """
    Visualize camera in Rerun.

    @note Does not take lens distorion into account.

    @param cam The camera.
    @param R World-to-camera rotation matrix.
    @param t World-to-camera translation vector.
    @param name Camera name.
    """
    rr.log(name, rr.Transform3D(translation=t, mat3x3=R, from_parent=True))
    rr.log(
        name,
        rr.Pinhole(
            focal_length=[cam.f[0], cam.f[1]],
            principal_point=cam.c,
            width=cam.size[0],
            height=cam.size[1],
        ),
    )


def log_2d_segments(cam_name, segments, colors=None):
    colors = colors if colors is not None else np.array([50, 50, 200])
    rr.log(
        f"{cam_name}/image/residuals",
        rr.LineStrips2D(segments, colors=colors, radii=1)
    )


def log_projected_3d_points(
    pts_3d, pts_2d, T_w2cam, poselib_intrinsics, cam_name, pred_mask=None):
    """
    pts_3d: shape (N, 3) in world coordinates
    pts_2d: shape (M, 2) in image coordinates (pixel coordinates)
    T_w2cam: shape (3, 4) in world2camera
    poselib_intrinsics: Camera (poselib format)
    cam_name: Camera name in Rerun
    pred_mask: Optional, shape (N,) boolean mask indicating which 3D points are visible in the image.
    """
    # NOTE: We do not visualize type 0 points here. But we could have if we wanted to.
    K = poselib_intrinsics.calibration_matrix()  # (3, 3)
    proj_pts = geo.pixels_from_3D(pts_3d, T_w2cam[None], K[None])[:, 0]  # (N, 2)
    if pred_mask is not None:
        proj_pts = proj_pts[pred_mask.to(proj_pts.device)]  # [N, 2] -> [M, 2]
    proj_pts = proj_pts.cpu().numpy()

    log_keypoints(proj_pts, cam_name, extra_string="_proj", colors=np.array([50, 200, 50]))
    log_keypoints(pts_2d, cam_name)

    # Build (M, 2, 2) -> M segments, each with two 2D points [start, end]
    segments = np.stack([proj_pts, pts_2d], axis=1)
    log_2d_segments(cam_name, segments)


def plot_scene(pts_3d, pts_2d, img_query, imgs_refs, camera_poses_refs, poselib_cam_intrinsics_q,
               poselib_cam_intrinsics_refs, cam_pose_query_est=None, cam_pose_query_gt=None,
               pts_2d_refs=None, plot_projected_3d_points=False, pred_mask=None, xyz_plot=None):
    """
    pts_3d: shape (N, 3) in world coordinates
    pts_2d: shape (N, 2) in image coordinates (pixel coordinates)
    img_query: shape (H, W, 3) in RGB
    imgs_refs: list of images of shape (H_i, W_i, 3) in RGB. Len of list is T.
    camera_poses_refs: shape (T, 3, 4) in world2camera (T number of camera poses)
    poselib_cam_intrinsics_q: Camera (poselib format)
    poselib_cam_intrinsics_refs: Camera (poselib format)
    cam_pose_query_est: None or shape (3, 4) (world2camera)
    cam_pose_query_gt: None or shape (3, 4) (world2camera)
    pts_2d_refs: Optional, list of ref kpts per image of shape (M_i, 2)
                 in image coordinates (pixel coordinates)
    plot_projected_3d_points: If True, log projected 3D points.
    pred_mask: Optional, shape (N, T) boolean mask indicating which 3D points are
               visible in each reference image.
    xyz_plot: Optional, shape (M, 3) additional 3D points to plot (e.g. query keypoints
              lifted to 3D)
    """

    # 1) Initialize Rerun
    rr.init("A Scene", spawn=True)  # spawn=True will open the Rerun Viewer automatically.
    # Connect to the Rerun TCP server using the default address and
    # port: localhost:9876
    #rr.connect_tcp()
    #rr.serve_web(open_browser=True, ws_port=4321)

    # 2) Log the 3D points
    # Get color of 3D points
    #gamma = 0.7
    #img_query_gamma = np.clip(np.power(img_query, gamma), 0.0, 1.0)
    pts_3d_colors = on3ru.bilinear_interpolation(img_query, pts_2d)
    log_3d_points(pts_3d, parent_path="world", colors=pts_3d_colors, filt=True)
    if xyz_plot is not None:
        xyz_plot_cs = cam_pose_query_est[:3, :3] @ xyz_plot.T + cam_pose_query_est[:3, 3:4]
        radii = 0.05
        xyz_plot_cs[2] -= 1.5*radii  # Convert to camera coordinate system for visualization
        xyz_plot_closer = (cam_pose_query_est[:3, :3].T @ (xyz_plot_cs - cam_pose_query_est[:3, 3:4])).T
        log_3d_points(xyz_plot_closer, parent_path="world", colors=np.array([0, 255, 0]), filt=True, title="query_kpts", radii=radii)

    # 3) Log query camera and image
    name_q_gt = "world/camera_query_gt"
    name_q_est = "world/camera_query_estimated"

    if cam_pose_query_est is not None:
        rot_in_est, t_in_est = cam_pose_query_est[:, :3], cam_pose_query_est[:, 3]
        visualize_camera(poselib_cam_intrinsics_q, rot_in_est, t_in_est, name_q_est)
        visualize_image(img_query, name_q_est)

    if cam_pose_query_gt is not None:
        rot_in_gt, t_in_gt = cam_pose_query_gt[:, :3], cam_pose_query_gt[:, 3]
        visualize_camera(poselib_cam_intrinsics_q, rot_in_gt, t_in_gt, name_q_gt)
        visualize_image(img_query, name_q_gt)

    # 4) Log reference cameras and images
    for i, img_ref in enumerate(imgs_refs):
        camera_name = f"world/camera_db_{i}"
        pose_ri = camera_poses_refs[i]
        visualize_image(img_ref, camera_name)
        visualize_camera(poselib_cam_intrinsics_refs[i], pose_ri[:, :3], pose_ri[:, 3], camera_name)

        if pts_2d_refs is not None:
            if plot_projected_3d_points and pred_mask is not None:
                # Log projected 3D points
                log_projected_3d_points(
                    pts_3d, pts_2d_refs[i], camera_poses_refs[i], poselib_cam_intrinsics_refs[i], camera_name,
                    pred_mask=pred_mask[:, i])
            else:
                log_keypoints(pts_2d_refs[i], camera_name)

    # 5) Log keypoints
    if cam_pose_query_gt is not None:
        if plot_projected_3d_points:
            log_projected_3d_points(
                pts_3d, pts_2d, cam_pose_query_gt, poselib_cam_intrinsics_q, name_q_gt)
        else:
            log_keypoints(pts_2d, name_q_gt)

    if cam_pose_query_est is not None:
        if plot_projected_3d_points:
            log_projected_3d_points(
                pts_3d, pts_2d, cam_pose_query_est, poselib_cam_intrinsics_q, name_q_est)
        else:
            log_keypoints(pts_2d, name_q_est)
    pass


def plot_3d_scene(pts_3d, pts_2d, camera_poses_refs, cam_pose_query_est, data, batch_ind, pts_2d_refs=None,
                  plot_projected_3d_points=False, pred_mask=None, new_cam_center_db=None, xyz_plot=None,
                  sc=None, b=None):
    """
    pts_3d: shape (N, 3) in world coordinates
    pts_2d: shape (N, 2) in image coordinates (pixel coordinates)
    camera_poses_refs: shape (T, 3, 4) in world2camera (T number of camera poses)
    cam_pose_query_est: None or shape (3, 4) (world2camera)
    data: dict
    batch_ind: int
    """
    if isinstance(pts_3d, torch.Tensor):
        pts_3d = pts_3d.detach().cpu().numpy()
    if isinstance(pts_2d, torch.Tensor):
        pts_2d = pts_2d.detach().cpu().numpy()
    if isinstance(camera_poses_refs, torch.Tensor):
        camera_poses_refs = camera_poses_refs.detach().cpu().numpy()
    if isinstance(cam_pose_query_est, torch.Tensor):
        cam_pose_query_est = cam_pose_query_est.detach().cpu().numpy()

    # Gather images
    first_key = list(data.keys())[0]
    assert first_key == 'query_to_ref_0', 'query_to_ref_0 should be the first key'

    wh = data[first_key]['view0']['image_size'][batch_ind]  # (2,)
    img_query = data[first_key]['view0']['image'][batch_ind][:, :wh[1], :wh[0]]
    img_query = img_query.permute(1, 2, 0).cpu().numpy()
    imgs_refs = []
    for key in data.keys():
        wh = data[key]['view1']['image_size'][batch_ind]  # (2,)
        img_ref = data[key]['view1']['image'][batch_ind][:, :wh[1], :wh[0]]
        imgs_refs.append(img_ref.permute(1, 2, 0).cpu().numpy())

    # Gather intrinsics
    poselib_cam_intrinsics_q = geo.get_poselib_camera_query(data, batch_ind)
    poselib_cam_intrinsics_refs = geo.get_poselib_camera_refs(data, batch_ind)

    # Gather query camera pose ground truth
    cam_pose_query_gt = geo.get_pose_query_w2cam(data, batch_ind)
    if new_cam_center_db is not None:
        cam_pose_query_gt_new_cs = geo.center_extrinsics(
            torch.tensor(cam_pose_query_gt[None]), new_cam_center_db.cpu())[0].numpy()
    else:
        cam_pose_query_gt_new_cs = cam_pose_query_gt

    q_gt_pose_upd = cam_pose_query_gt_new_cs.copy()
    if sc is not None and b is not None:
        t = cam_pose_query_gt_new_cs[:3, 3]
        R = cam_pose_query_gt_new_cs[:3, :3]
        t_upd = sc * t - (1 - sc) * R@b
        q_gt_pose_upd = cam_pose_query_gt_new_cs.copy()
        q_gt_pose_upd[:3, 3] = t_upd

    plot_scene(pts_3d, pts_2d, img_query, imgs_refs, camera_poses_refs, poselib_cam_intrinsics_q,
               poselib_cam_intrinsics_refs, cam_pose_query_est, q_gt_pose_upd,
               pts_2d_refs, plot_projected_3d_points, pred_mask, xyz_plot=xyz_plot)
    pass


def plot_3d_scene_no_data(pts_3d, pts_2d, camera_poses_refs, cam_pose_query_est, pts_2d_refs=None,
                          plot_projected_3d_points=False, pred_mask=None, img_query=None, imgs_refs=None,
                          poselib_cam_intrinsics_q=None, poselib_cam_intrinsics_refs=None):
    """
    pts_3d: shape (N, 3) in world coordinates
    pts_2d: shape (N, 2) in image coordinates (pixel coordinates)
    camera_poses_refs: shape (T, 3, 4) in world2camera (T number of camera poses)
    cam_pose_query_est: None or shape (3, 4) (world2camera)
    data: dict
    batch_ind: int
    """
    if isinstance(pts_3d, torch.Tensor):
        pts_3d = pts_3d.detach().cpu().numpy()
    if isinstance(pts_2d, torch.Tensor):
        pts_2d = pts_2d.detach().cpu().numpy()
    if isinstance(camera_poses_refs, torch.Tensor):
        camera_poses_refs = camera_poses_refs.detach().cpu().numpy()
    if isinstance(cam_pose_query_est, torch.Tensor):
        cam_pose_query_est = cam_pose_query_est.detach().cpu().numpy()
    if img_query is not None and isinstance(img_query, torch.Tensor):
        img_query = img_query.cpu().numpy()
    if imgs_refs is not None:
        for i in range(len(imgs_refs)):
            if isinstance(imgs_refs[i], torch.Tensor):
                imgs_refs[i] = imgs_refs[i].cpu().numpy()

    # Gather query camera pose ground truth
    cam_pose_query_gt = None
    plot_scene(pts_3d, pts_2d, img_query, imgs_refs, camera_poses_refs, poselib_cam_intrinsics_q,
               poselib_cam_intrinsics_refs, cam_pose_query_est, cam_pose_query_gt,
               pts_2d_refs, plot_projected_3d_points, pred_mask)
    pass

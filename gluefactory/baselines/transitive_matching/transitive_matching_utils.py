import numpy as np
import poselib

from locba.src.pybind_ba import single_cam_pylocba


def assemble_transitive_matching_data(data, matches, batch_ind):
    tuple_length = len(data.keys())
    matches_q2ref = matches.swapaxes(0, 1).cpu().numpy()  # (T, K)

    K_q = data['query_to_ref_0']['view0']['camera'].calibration_matrix()[batch_ind].cpu().numpy()  # (3, 3)
    camera_q = {
        'model': 'PINHOLE',
        'width': data['query_to_ref_0']['view0']['image_size'][batch_ind][0].cpu().numpy(),
        'height': data['query_to_ref_0']['view0']['image_size'][batch_ind][1].cpu().numpy(),
        'params': np.array([K_q[0, 0], K_q[1, 1], K_q[0, 2], K_q[1, 2]])
    }

    K_ref = [
        data['query_to_ref_' + str(j)]['view1']['camera'].calibration_matrix()[batch_ind].cpu().numpy() for j in range(tuple_length)
    ]
    cameras_ref = [
        {
            'model': 'PINHOLE',
            'width': data['query_to_ref_' + str(j)]['view1']['image_size'][batch_ind][0].cpu().numpy(),
            'height': data['query_to_ref_' + str(j)]['view1']['image_size'][batch_ind][1].cpu().numpy(),
            'params': np.array([K_ref[j][0, 0], K_ref[j][1, 1], K_ref[j][0, 2], K_ref[j][1, 2]])
        } for j in range(tuple_length)
    ]

    Rt_ref = [
        data['query_to_ref_' + str(j)]['view1']['T_w2cam'][batch_ind].cpu().numpy() for j in range(tuple_length)
    ]


    poses_ref = [
        to_qt(p[0], p[1]) for p in Rt_ref
    ]
    return matches_q2ref, cameras_ref, poses_ref, camera_q


def to_qt(R,t):
    p = poselib.CameraPose()
    p.R = R
    p.t = t
    return [p.q, p.t]


def run_ba(rec, kpts_query, kpts_ref, matches_q2ref, camera_q, pose_q, cameras_ref, poses_ref):
    """
    Perform bundle adjustment using pylocba for a single camera.
    Args:
        rec: COLMAP reconstruction object
        kpts_query: np.ndarray of shape (K, 2) keypoints of the query image (np.float32)
        kpts_ref: np.ndarray of shape (T, K, 2) keypoints of the reference images (np.float32)
        matches_q2ref: np.ndarray of shape (T, K) matches between query and ref images (int64)
        cameras_ref: list of camera parameters for reference images (len T of list)
        poses_ref: list of camera poses for reference images (len T of list)
            poses_ref[i] = [q, t] where q is the quaternion and t is the translation vector
        camera_q: camera parameters for the query image
            (model, width, height, params) e.g.
            camera_q = {'model': 'PINHOLE', 'width': 640, 'height': 480,
                        'params': np.array([fx, fy, cx, cy])}
    Returns:
        result: result of the bundle adjustment (dict with keys "points_3d", "pose_query")
    """
    xyz = np.array([p.xyz for p in rec.points3D.values()])  # (N, 3)
    tuple_length = kpts_ref.shape[0]

    ref_kpts_in_model = np.full((xyz.shape[0], tuple_length, 2), -1)
    query_kpts_in_model = np.full((xyz.shape[0], 2), -1)
    max_ind = 2**64-1
    keys_3D = list(rec.points3D.keys())

    for k, m in enumerate(matches_q2ref):
        im = rec.images[k+1]
        if im.num_points3D == 0:  # is not triangulated (triangulated = 0/0 which gives error)
            continue

        have_match = m >= 0
        ind = np.where(have_match)[0]
        for ind_q in ind:
            ind_r = m[ind_q]

            pt3d_id = im.points2D[ind_r].point3D_id
            if pt3d_id < 0 or pt3d_id == max_ind:
                continue
            assert pt3d_id in keys_3D, f"pt3d_id: {pt3d_id} not in keys_3D"
            ind_2d_pt = keys_3D.index(pt3d_id) if pt3d_id in keys_3D else -1
            if ind_2d_pt == -1:
                print("error!!")
            query_kpts_in_model[ind_2d_pt] = kpts_query[ind_q]
            ref_kpts_in_model[ind_2d_pt, k] = kpts_ref[k, ind_r]
    track_mask = ref_kpts_in_model[:, :, 0] != -1  # (N, T)
    # (T, 3, 3)
    q_intrinsics = np.array([[camera_q["params"][0], 0, camera_q["params"][2]],
                             [0, camera_q["params"][1], camera_q["params"][3]],
                             [0, 0, 1]])
    r_intrinsics = np.stack([np.array([[camera_ref["params"][0], 0, camera_ref["params"][2]],
                            [0, camera_ref["params"][1], camera_ref["params"][3]],
                            [0, 0, 1]]) for camera_ref in cameras_ref])
    query_kpts_h = np.concatenate((query_kpts_in_model,
                                   np.ones((query_kpts_in_model.shape[0], 1))), axis=1)  # (K, 3)
    ref_kpts_h = np.concatenate((
        ref_kpts_in_model,
        np.ones((ref_kpts_in_model.shape[0], tuple_length, 1))),
        axis=2)  # (N, T, 3)

    # Some 2d keypoints are not used
    q_train_norm = np.einsum('ij,nj->ni', np.linalg.inv(q_intrinsics), query_kpts_h)[:, :2]  #(N, 2)
    # (N, T, 2)
    r_train_norm = np.einsum('tij,ntj->nti', np.linalg.inv(r_intrinsics), ref_kpts_h)[..., :2]
    q_cam = pose_q.Rt  # (3, 4)

    r_cams = []
    for pose_ref in poses_ref:
        pose_r = poselib.CameraPose()
        pose_r.q = pose_ref[0]
        pose_r.t = pose_ref[1]
        r_cams.append(pose_r.Rt)
    r_cams = np.stack(r_cams)  # (T, 3, 4)

    tol_reproj = 1.0  # px
    tol_reproj_norm = tol_reproj / q_intrinsics[0, 0]  # scaling of Cauchy loss

    result = single_cam_pylocba(
        xyz, q_train_norm, r_train_norm, track_mask, q_cam, r_cams, tol_reproj_norm,
    )
    return result

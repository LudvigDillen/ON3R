import numpy as np
import poselib
import torch
import sys
from pathlib import Path

from ..transitive_matching import transitive_matching_utils as tmu
_RELOC3R_REPO = Path(__file__).resolve().parents[3] / "reloc3r"
if str(_RELOC3R_REPO) not in sys.path:
    sys.path.insert(0, str(_RELOC3R_REPO))

from reloc3r.reloc3r_visloc import Reloc3rVisloc


def absolute_pose_from_motion_averaging(data, kpts_q, kpts_ref, matches, batch_ind):
    matches_q2ref, cameras_ref, abs_poses_ref, camera_q = \
        tmu.assemble_transitive_matching_data(data, matches, batch_ind)

    n_refs = len(cameras_ref)
    n_kpts = kpts_q.shape[0]
    poses_q2d = []
    poses_db = []
    n_rel_poses_estimated = 0

    for i in range(n_refs):
        matches_q2_ref_i = matches_q2ref[i]
        mask = matches_q2_ref_i >= 0  # Only consider valid matches
        inds_q = torch.arange(n_kpts)[mask]
        inds_ref_i = matches_q2_ref_i[mask]

        xq = kpts_q[inds_q]  # (K, 2)
        xr_i = kpts_ref[i, inds_ref_i]  # (K, 2)

        if xq.shape[0] < 3 or xr_i.shape[0] < 3:
            continue

        # Estimate relative pose
        # NOTE: I assume that the relative pose is from query to reference, how to be sure?
        pose_q2d_, info = poselib.estimate_relative_pose(xq, xr_i, camera_q, cameras_ref[i],
                                                         {'max_reproj_error': 16.0}, {})
        if info["num_inliers"] < 3:
            continue
        n_rel_poses_estimated += 1

        pose_q2d = np.eye(4, dtype=pose_q2d_.Rt.dtype)
        # TODO: the problem is that pose_q2d can be the identity which is
        # problematic since it will lead to the centers being at the same place
        # I guess it is the identity when there are no matches
        pose_q2d[:3] = pose_q2d_.Rt
        poses_q2d.append(pose_q2d)

        # Expected absolute pose should be camera to world, so we need to invert the database pose
        p = poselib.CameraPose()
        p.q = abs_poses_ref[i][0]
        p.t = abs_poses_ref[i][1]
        p_inv = poselib.CameraPose()
        p_inv.R = p.R.T
        p_inv.t = -p.R.T @ p.t
        abs_pose_cam2w = np.eye(4, dtype=p_inv.Rt.dtype)
        abs_pose_cam2w[:3] = p_inv.Rt
        poses_db.append(abs_pose_cam2w)

    # Perform motion averaging using Reloc3r code. All database poses are known, we have relative
    # poses from query to all database, now we do motion averaging to get the abs pose of the query.
    p = poselib.CameraPose()
    if n_rel_poses_estimated == 0:
        pass  # Return identity pose if no relative poses were estimated
    elif n_rel_poses_estimated == 1:  # Only one relative pose, use it directly
        # (q2w) = (cam2w = poses_db[0]), (poses_q2d[0] q2cam)
        absolute_pose_q2w = poses_db[0]@poses_q2d[0]
        p.Rt = np.linalg.inv(absolute_pose_q2w)[:3]
    else:
        reloc3r_visloc = Reloc3rVisloc()
        absolute_pose_q2w = reloc3r_visloc.motion_averaging(poses_db, poses_q2d)
        p.Rt = np.linalg.inv(absolute_pose_q2w)[:3]
    return p

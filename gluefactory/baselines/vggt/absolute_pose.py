import torch
import numpy as np
import poselib
import sys
from pathlib import Path
from functools import lru_cache
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

_RELOC3R_REPO = Path(__file__).resolve().parents[3] / "reloc3r"
if str(_RELOC3R_REPO) not in sys.path:
    sys.path.insert(0, str(_RELOC3R_REPO))

from reloc3r.reloc3r_visloc import Reloc3rVisloc
from ..transitive_matching.transitive_matching_utils import to_qt
from gluefactory.models.on3r.utils import get_img_path


@lru_cache(maxsize=1)
def get_model(device):
    """Return the same VGGT model instance every time."""
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()                    # helpful if you only infer
    return model


def predict_relative_poses(model, image_paths, device):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    images = load_and_preprocess_images(image_paths).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsics, _ = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:], build_intrinsics=False)

    # NOTE: The extrinsics are to cam0 from cam1, cam2, ..., camN (i.e. database to query).
    extrinsics_np = extrinsics.squeeze(0).cpu().numpy()  # (N, 3, 4) s.t. N is number of cams  (remove batch dim)
    poses_w2cam = [np.eye(4, dtype=extrinsics_np.dtype) for _ in range(len(extrinsics_np))]
    for i in range(len(extrinsics_np)):
        poses_w2cam[i][:3] = extrinsics_np[i]  # (3, 4) to (4, 4) by adding last row [0, 0, 0, 1]
    poses_q2d = []
    pose_w2q = poses_w2cam[0]
    for i in range(len(poses_w2cam)):
        # q -> w -> cam_i
        poses_q2d.append(poses_w2cam[i] @ np.linalg.inv(pose_w2q))

    return poses_q2d[1:]  # Exclude the first pose which is the query pose (identity)


def predict_absolute_pose_runner(model, data, tuple_length, batch_ind, device):
    image_paths = []
    q_img_pth = get_img_path(batch_ind, data["query_to_ref_0"], 0)
    image_paths.append(q_img_pth)

    for i in range(tuple_length):
        r_i_img_path = get_img_path(batch_ind, data[f"query_to_ref_{i}"], 1)
        image_paths.append(r_i_img_path)
    poses_q2d = predict_relative_poses(model, image_paths, device)

    Rt_ref = [
        data['query_to_ref_' + str(j)]['view1']['T_w2cam'][batch_ind].cpu().numpy() for j in range(tuple_length)
    ]

    abs_poses_ref = [
        to_qt(p[0], p[1]) for p in Rt_ref
    ]

    poses_db = []  # List of (4, 4) absolute poses that transform points from camera to world
    for i in range(tuple_length):
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

    reloc3r_visloc = Reloc3rVisloc()
    absolute_pose_q2w = reloc3r_visloc.motion_averaging(poses_db, poses_q2d)
    p = poselib.CameraPose()
    p.Rt = np.linalg.inv(absolute_pose_q2w)[:3]
    return p

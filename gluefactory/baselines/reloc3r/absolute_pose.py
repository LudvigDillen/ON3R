import numpy as np
from functools import lru_cache
import poselib
import sys
from pathlib import Path
_RELOC3R_REPO = Path(__file__).resolve().parents[3] / "reloc3r"
if str(_RELOC3R_REPO) not in sys.path:
    sys.path.insert(0, str(_RELOC3R_REPO))

from reloc3r.utils.image import load_images, check_images_shape_format
from reloc3r.reloc3r_relpose import setup_reloc3r_relpose_model, inference_relpose
from reloc3r.reloc3r_visloc import Reloc3rVisloc
from reloc3r.utils.device import to_numpy
from ..transitive_matching.transitive_matching_utils import to_qt
from gluefactory.models.on3r.utils import get_img_path


@lru_cache(maxsize=1)
def get_model(img_reso, device):
    """Return the same reloc3r model instance every time."""
    model = setup_reloc3r_relpose_model(model_args=img_reso, device=device)
    model.eval()                      # helpful if you only infer
    return model


def predict_relative_poses(model,image_paths, device, img_reso):
    query_path = image_paths[0]
    ref_paths = image_paths[1:]
    poses_q2d = []
    for ref_path in ref_paths:
        images = load_images([ref_path, query_path], size=int(img_reso))
        images = check_images_shape_format(images, device)

        # relpose
        for img in (images[0], images[1]):
            height, width = img['true_shape'][0]

            if width < height:
                # rectify portrait to landscape
                # print(f"Changed shape {img['img'].shape}")
                assert img['img'].shape == (1, 3, height, width)
                img['img'] = img['img'].swapaxes(2, 3)
                # print(f"to {img['img'].shape}")
        # print('Running relative pose estimation...')
        batch = [images[0], images[1]]
        pose_q2r = to_numpy(inference_relpose(batch, model, device)[0])
        pose_q2r[0:3,3] = pose_q2r[0:3,3] / np.linalg.norm(pose_q2r[0:3,3])  # normalize the scale to 1 meter
        poses_q2d.append(pose_q2r)

    return poses_q2d


def predict_absolute_pose_runner(model, data, tuple_length, batch_ind, device, img_reso):
    image_paths = []
    q_img_pth = get_img_path(batch_ind, data["query_to_ref_0"], 0)
    image_paths.append(q_img_pth)

    for i in range(tuple_length):
        r_i_img_path = get_img_path(batch_ind, data[f"query_to_ref_{i}"], 1)
        image_paths.append(r_i_img_path)
    poses_q2d = predict_relative_poses(model, image_paths, device, img_reso)

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

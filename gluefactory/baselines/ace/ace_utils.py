from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from functools import lru_cache
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

# Put this near the top of your module (once per process)
try:
    mp.set_sharing_strategy('file_system')   # reduces FD pressure on Linux
except RuntimeError:
    pass


@lru_cache(maxsize=1)
def cached_encoder(encoder_path: str):
    state = torch.load(encoder_path, map_location="cpu")
    # If you have a factory for just the encoder, use that here.
    # We reconstruct via Regressor then keep only .encoder if needed:
    # enc = build_ace_encoder_from_state(state)
    # For backward-compat, we keep the split loader below and stitch later.
    return state  # we cache the state dict; materialize to device once per regressor


@lru_cache(maxsize=1)
def cached_head_encoder(head_path: str):
    state = torch.load(head_path, map_location="cpu")
    return state


def _Rt_cam2w(T_w2cam):
    """Accepts something with .R (3x3) and .t (3,) already world->cam."""
    # ACE writes in the paper:
    # "We define the camera pose as the rigid body transformation that maps coordinates in
    # camera space ei to coordinates in scene space yi, therefore yi = hei."
    # Hence, we need to invert T_w2cam to get T_cam2w.
    R = np.asarray(T_w2cam.R.cpu(), dtype=np.float32)
    t = np.asarray(T_w2cam.t.cpu(), dtype=np.float32).reshape(3, 1)
    last_row = np.array([[0, 0, 0, 1]], dtype=np.float32)
    posew2c = np.vstack([np.hstack([R, t]), last_row])  # 4x4
    pose = np.linalg.inv(posew2c)
    return pose


def ace_adjusted_K(K):
    # ace only supports one focal length, so we average fx and fy
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    f = (fx + fy) / 2.0
    K_ace = np.array([[f, 0, cx],
                      [0, f, cy],
                      [0, 0, 1]], dtype=np.float32)
    return K_ace


def _save_rgb(tensor_hwc_uint8, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_hwc_uint8).save(path)


def make_options(**kwargs):
    # Mirror your argparse fields so TrainerACE can consume it without changes.
    # Defaults align with your CLI script.
    defaults = dict(
        scene=None,
        output_map_file=None,
        encoder_path=Path(__file__).parents[3] / "ace/ace_encoder_pretrained.pt",
        num_head_blocks=1,
        learning_rate_min=0.0005,
        learning_rate_max=0.005,
        training_buffer_size=8_000_000,
        samples_per_image=1024,
        batch_size=5120,
        epochs=16,
        repro_loss_hard_clamp=1000,
        repro_loss_soft_clamp=50,
        repro_loss_soft_clamp_min=1,
        use_half=True,
        use_homogeneous=True,
        use_aug=True,
        aug_rotation=15,
        aug_scale=1.5,
        image_resolution=480,
        repro_loss_type="dyntanh",
        repro_loss_schedule="circle",
        depth_min=0.1,
        depth_target=10.0,
        depth_max=1000.0,
        num_clusters=None,
        cluster_idx=None,
        render_visualization=False,
        render_target_path=Path("renderings"),
        render_flipped_portrait=False,
        render_map_error_threshold=10,
        render_map_depth_filter=10,
        render_camera_z_offset=4,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def export_tuple_as_dsac_scene(scene_root: Path, data: dict, batch_i: int):
    """
    Creates a minimal DSAC* scene folder from your tuple batch.
    - data['query_to_ref_0']['view0'] is the query
    - data[f'query_to_ref_{j}']['view1'] are the refs (j=0..T-1)
    """
    scene_root = Path(scene_root)
    train_rgb = scene_root / "train" / "rgb"
    test_rgb  = scene_root / "test"  / "rgb"
    train_poses = scene_root / "train" / "poses"
    test_poses  = scene_root / "test"  / "poses"
    train_calibrations = scene_root / "train" / "calibration"
    test_calibrations  = scene_root / "test"  / "calibration"
    (scene_root / "train").mkdir(parents=True, exist_ok=True)
    (scene_root / "test").mkdir(parents=True, exist_ok=True)
    (train_rgb).mkdir(parents=True, exist_ok=True)
    (test_rgb).mkdir(parents=True, exist_ok=True)
    (train_poses).mkdir(parents=True, exist_ok=True)
    (test_poses).mkdir(parents=True, exist_ok=True)
    (train_calibrations).mkdir(parents=True, exist_ok=True)
    (test_calibrations).mkdir(parents=True, exist_ok=True)

    # 0) Query first (index 0)
    v0 = data['query_to_ref_0']['view0']
    img_q_padded = (v0['image'][batch_i].clamp(0, 1).mul(255).byte().permute(1,2,0).cpu().numpy())
    wq, hq = v0['image_size'][batch_i]
    img_q = img_q_padded[:hq, :wq, :]
    _save_rgb(img_q, test_rgb / f"frame_{0:05d}.png")
    K_q = ace_adjusted_K(v0["camera"][batch_i].calibration_matrix().cpu().numpy())
    Rt_q = _Rt_cam2w(v0["T_w2cam"][batch_i])
    with open(test_calibrations / f"calibration_{0:05d}.txt", "w") as fki, \
         open(test_poses / f"poses_{0:05d}.txt", "w") as fp:
        np.savetxt(fki, K_q, fmt="%.18e")
        np.savetxt(fp, Rt_q, fmt="%.18e")

    # --- 2) Refs -> train ---
    img_idx = 0
    while f"query_to_ref_{img_idx}" in data:
        v1 = data[f"query_to_ref_{img_idx}"]["view1"]
        img_r_padded = (
            v1["image"][batch_i]
            .clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        )
        wr, hr = v1['image_size'][batch_i]
        img_r = img_r_padded[:hr, :wr, :]
        _save_rgb(img_r, train_rgb / f"frame_{img_idx:05d}.png")

        K_r = ace_adjusted_K(v1["camera"][batch_i].calibration_matrix().cpu().numpy())
        Rt_r = _Rt_cam2w(v1["T_w2cam"][batch_i])

        img_idx += 1

        with open(train_calibrations / f"calibration_{img_idx:05d}.txt", "w") as fki, \
             open(train_poses / f"poses_{img_idx:05d}.txt", "w") as fp:
            np.savetxt(fki, K_r, fmt="%.18e")
            np.savetxt(fp, Rt_r, fmt="%.18e")

    return scene_root

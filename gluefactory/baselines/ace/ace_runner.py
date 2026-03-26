from __future__ import annotations

from pathlib import Path
import numpy as np
import math, time
import logging
from typing import Optional, Dict, Any, Tuple, List
import gc

import cv2
import torch
from torch.cuda.amp import autocast
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import sys
from pathlib import Path as _Path
_ace_dir = str(_Path(__file__).resolve().parents[3] / "ace")
if _ace_dir not in sys.path:
    sys.path.insert(0, _ace_dir)

from ace.dataset import CamLocDataset

import dsacstar
from ace.ace_network import Regressor
from ace.ace_trainer import TrainerACE

from .ace_utils import (export_tuple_as_dsac_scene, cached_encoder,
                        cached_head_encoder, make_options)


_logger = logging.getLogger(__name__)
# Put this near the top of your module (once per process)
try:
    mp.set_sharing_strategy('file_system')   # reduces FD pressure on Linux
except RuntimeError:
    pass


def run_ace(data, batch_i, workdir=Path(__file__).parents[3] / "ace_scenes"):
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    try:
        with open(workdir / "scene_id.txt", "r+") as f:
            scene_id = int(f.read().strip())  # read current ID
            f.seek(0)                         # move cursor to start
            f.write(str(scene_id + 1))        # increment for next time
            f.truncate()                      # remove any leftover content
    except FileNotFoundError:
        with open(workdir / "scene_id.txt", "w") as tmp:
            tmp.write(str(2))
            tmp.truncate()
        scene_id = 1

    scene_name = f"ace_scene_{scene_id}"
    scene_dir = workdir / scene_name
    scene = export_tuple_as_dsac_scene(scene_dir, data, batch_i)

    out_map = scene_dir / "ace_head.pt"
    opts = make_options(
        scene=Path(scene),
        output_map_file=Path(out_map),
        epochs=16,
        training_buffer_size=200000,
        samples_per_image=50000,
        batch_size=5120,
    )
    trainer = TrainerACE(opts)
    trainer.num_data_loader_workers = 0
    trainer.train()
    del trainer
    gc.collect()

    rot_error_deg, trans_error_m = run_test_for_scene(
        scene=scene,
        head_network_path=out_map,             # ← no disk I/O
        encoder_path=Path(__file__).parents[3] / "ace/ace_encoder_pretrained.pt",
        device="cuda",
        scene_name=scene_name,
        output_dir=scene_dir,
    )

    return rot_error_deg, trans_error_m


@torch.no_grad()
def run_test_for_scene(
    scene: Path,
    # One of these two is required:
    head_network_path,

    # Reuse/caching knobs:
    device: str = "cuda",
    encoder_path: Optional[Path] = None,           # if provided, reuse cached encoder state

    # DSAC / eval params:
    image_resolution: int = 480,
    hypotheses: int = 64,
    threshold: float = 10.0,
    inlieralpha: float = 100.0,
    maxpixelerror: float = 100.0,
    session="",
    scene_name: str = "",
    output_dir: Optional[Path] = None,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    device = torch.device(device)
    scene = Path(scene)

    # Dataloader cache (fast)
    testset = CamLocDataset(
        scene / "test",
        mode=0,
        image_height=image_resolution,
    )
    # num_workers tuned—adjust if you see dataloader overheads
    testset_loader = DataLoader(testset, shuffle=False, num_workers=0, persistent_workers=False,
                                pin_memory=False)

    total_frames_before = len(testset_loader.dataset)

    # Build / reuse regressor
    encoder_state_dict = cached_encoder(str(encoder_path))
    head_state_dict = cached_head_encoder(str(head_network_path))

    # Create regressor from split states
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

    network = network.to(device).eval()


    # This will contain aggregate scene stats (median translation/rotation errors, and avg processing time per frame).
    test_log_file = output_dir / f'test_{scene_name}_{session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    # This will contain each frame's pose (stored as quaternion + translation) and errors.
    pose_log_file = output_dir / f'poses_{scene_name}_{session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest.
    avg_batch_time = 0
    num_batches = 0

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    # Percentage of frames predicted within certain thresholds from their GT pose.
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # Testing loop.
    for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
        batch_start_time = time.time()

        image_B1HW = image_B1HW.to(device, non_blocking=True)

        # Predict scene coordinates.
        with autocast(enabled=True):
            scene_coordinates_B3HW = network(image_B1HW)

        # We need them on the CPU to run RANSAC.
        scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

        # Each frame is processed independently.
        for (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path) in zip(
            scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames):

            # Extract focal length and principal point from the intrinsics matrix.
            focal_length = intrinsics_33[0, 0].item()
            ppX = intrinsics_33[0, 2].item()
            ppY = intrinsics_33[1, 2].item()
            # We support a single focal length.
            assert torch.allclose(intrinsics_33[0, 0], intrinsics_33[1, 1])

            # Remove path from file name
            frame_name = Path(frame_path).name

            # Allocate output variable.
            out_pose = torch.zeros((4, 4))

            # Compute the pose via RANSAC.
            inlier_count = dsacstar.forward_rgb(
                scene_coordinates_3HW.unsqueeze(0),
                out_pose,
                hypotheses,
                threshold,
                focal_length,
                ppX,
                ppY,
                inlieralpha,
                maxpixelerror,
                network.OUTPUT_SUBSAMPLE,
            )

            # Calculate translation error.
            t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

            # Rotation error.
            gt_R = gt_pose_44[0:3, 0:3].numpy()
            out_R = out_pose[0:3, 0:3].numpy()

            r_err = np.matmul(out_R, np.transpose(gt_R))
            # Compute angle-axis representation.
            r_err = cv2.Rodrigues(r_err)[0]
            # Extract the angle.
            r_err = np.linalg.norm(r_err) * 180 / math.pi

            _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

            # Save the errors.
            rErrs.append(r_err)
            tErrs.append(t_err * 100)

            # Check various thresholds.
            if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                pct10_5 += 1
            if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                pct5 += 1
            if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                pct2 += 1
            if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                pct1 += 1

            # Write estimated pose to pose file (inverse).
            out_pose = out_pose.inverse()

            # Translation.
            t = out_pose[0:3, 3]

            # Rotation to axis angle.
            rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
            angle = np.linalg.norm(rot)
            axis = rot / angle

            # Axis angle to quaternion.
            q_w = math.cos(angle * 0.5)
            q_xyz = math.sin(angle * 0.5) * axis

            # Write to output file. All in a single line.
            pose_log.write(f"{frame_name} "
                            f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                            f"{t[0]} {t[1]} {t[2]} "
                            f"{r_err} {t_err} {inlier_count}\n")

        avg_batch_time += time.time() - batch_start_time
        num_batches += 1

    total_frames = len(rErrs)
    assert total_frames == total_frames_before, "Frame count mismatch"

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]
    assert len(tErrs) == 1, "We only localize one image"
    rot_error_deg = median_rErr
    trans_error_m = median_tErr / 100.0

    # Compute average time.
    avg_time = avg_batch_time / num_batches

    # Compute final metrics.
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    # Write to the test log file as well.
    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")

    test_log.close()
    pose_log.close()
    return rot_error_deg, trans_error_m

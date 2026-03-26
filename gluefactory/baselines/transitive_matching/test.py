import numpy as np
import pickle
from .transitive_matching import *
import poselib
import argparse
import time

from gluefactory.utils.misc import start_debug
from ...models.on3r import geometry as geo


"""
Run with
python -m gluefactory.baselines.transitive_matching.test (--debug)
"""
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
args = parser.parse_intermixed_args()

# Start the debugger if the debug argument was given
if args.debug:
    start_debug()

with open("assets/data_for_colmap.pkl", "rb") as f:
    loaded_data = pickle.load(f)

# dict_keys(['matches_for_query_kpts', 'kpts_query', 'kpts_db',
# 'pose_w2cam_query', 'pose_w2cam_db', 'intrinsics_query', 
# 'intrinsics_db', 'image_dims_width_height_query', 'image_dims_width_height_db'])

# Load data and mangle it a bit
kpts_q = loaded_data["kpts_query"]
kpts_ref = [
    loaded_data['kpts_db']['query_to_ref_' + str(i)] for i in range(5)
]
matches_q2ref = [
    loaded_data['matches_for_query_kpts']['query_to_ref_' + str(i)] for i in range(5)
]
K_q = loaded_data['intrinsics_query']
K_ref = [
    loaded_data['intrinsics_db']['query_to_ref_' + str(i)] for i in range(5)
]
camera_q = {
    'model': 'PINHOLE',
    'width': loaded_data['image_dims_width_height_query'].cpu().numpy()[0],
    'height': loaded_data['image_dims_width_height_query'].cpu().numpy()[1],
    'params': np.array([K_q[0, 0], K_q[1, 1], K_q[0, 2], K_q[1, 2]])
}
cameras_ref = [
    {
        'model': 'PINHOLE',
        'width': loaded_data['image_dims_width_height_db']['query_to_ref_' + str(i)].cpu().numpy()[0],
        'height': loaded_data['image_dims_width_height_db']['query_to_ref_' + str(i)].cpu().numpy()[1],
        'params': np.array([K_ref[i][0, 0], K_ref[i][1, 1], K_ref[i][0, 2], K_ref[i][1, 2]])
    } for i in range(5)
]

Rt_q = loaded_data['pose_w2cam_query']
Rt_ref = [
    loaded_data['pose_w2cam_db']['query_to_ref_' + str(i)] for i in range(5)
]
def to_qt(R,t):
    p = poselib.CameraPose()
    p.R = R
    p.t = t
    return [p.q, p.t]
poses_ref = [
    to_qt(p[:,:3], p[:,3]) for p in Rt_ref
]

# Reformat input
kpts_ref = np.stack(kpts_ref)  # (T, K, 2)
matches_q2ref = np.stack(matches_q2ref)  # (T, K)
t1 = time.time()
pose = transitive_matching(kpts_q, kpts_ref, matches_q2ref, cameras_ref, poses_ref, camera_q)
t2 = time.time()
print(f"Time taken: {t2-t1:.2f} seconds")

print("Pose estimated:")
print(pose.R, pose.t)

print("GT pose:")
print(Rt_q[:,:3], Rt_q[:,3])

t_est_world = -pose.R.T @ pose.t
t_gt_world = -Rt_q[:,:3].T @ Rt_q[:,3]
pose_error_R = geo.get_pose_error_R(pose.R, Rt_q[:,:3])
pose_error_t = geo.get_absolute_pose_error_t(t_est_world, t_gt_world)

print(f"Pose error [R]: {pose_error_R:.2f} degrees")
print(f"Pose error [t]: {100*pose_error_t:.2f} cm")

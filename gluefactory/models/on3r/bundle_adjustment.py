import torch
from locba.src.pybind_ba import single_cam_pylocba


def pylocba_ba(model, xyz, q_K_norm, r_K_norm):
    result = single_cam_pylocba(
        xyz.cpu().detach().numpy(),
        q_K_norm.cpu().numpy(),
        r_K_norm.cpu().numpy(),
        model.pred_mask.cpu().numpy(),
        model.poselib_pose_est_query.Rt,
        model.extrinsics.cpu().numpy(),
        model.tol_reproj_norm.cpu().numpy(),
    )
    xyz = torch.tensor(result['points_3d'], dtype=model.dtype, device=model.device)
    model.poselib_pose_est_query.Rt = result['pose_query']
    return xyz

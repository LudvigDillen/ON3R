import torch

from . import utils as on3ru
from . import geometry as geo
from . import bundle_adjustment as ba
from . import plotting as on3r_plotting


def training_loop(
    model, optimizer, scheduler, q_norm, r_loss, ref_kpts, lr_end=1e-5,
    performance_mode="low_error", query_kpts=None, data=None, batch_ind=None,
    poselib_cam_query=None, img_query=None, imgs_refs=None, plot=False):
    old_xyz, break_check, old_recall = None, False, 0

    if model.use_moge2_depth:
        for epoch in range(1, 100):
            model.current_epoch = epoch
            model.train()
            optimizer.zero_grad()

            output = model(q_norm)  # (N, 2) -> (N, 3)
            loss = model.loss(output, r_loss, epoch=epoch, depth_only=True)
            error_est = model.intrinsics_query[0, 0] * model.s * torch.sqrt(torch.exp(loss.detach() / (model.s * model.moge2_depth_loss_scaler)) - 1.0)
            # print(f"Estimated depth error: {error_est:.2f} px at epoch {epoch}")
            loss.backward()
            optimizer.step()
            if error_est < 100.0:
                break
            if (not epoch % 20 or epoch == 1) and plot:
                print("Epoch: ", epoch)
                xyz = output[..., :3]  # (N, 3)
                on3r_plotting.plot_3d_scene(model, query_kpts, ref_kpts, poselib_cam_query, xyz,
                                            data, batch_ind, img_query, imgs_refs)

    for epoch in range(1, model.max_epochs+1):
        model.current_epoch = epoch
        model.train()
        optimizer.zero_grad()

        output = model(q_norm)  # (N, 2) -> (N, 3)
        loss = model.loss(output, r_loss, epoch=epoch, depth_only=False)

        loss.backward()
        optimizer.step()

        if not epoch % 10:
            # NOTE: Set to lower than 7 if more accuracy is needed. But this is faster. Or higher if less accuracy is ok.
            if model.residual_metric < 7.0 / model.intrinsics_query[0, 0].item():
                # print("Early stopping at epoch ", epoch)
                break
            # lr = optimizer.param_groups[0]['lr']
            # res = model.residual_metric * model.intrinsics_query[0, 0].item()
            # print(f"Epoch {epoch}: Residual metric (train): {res:.2f} px at lr={lr:.2e}")
            pass

        if (not epoch % 5 or epoch == 1) and plot:
            # print("Epoch: ", epoch)
            xyz = output[..., :3]  # (n_train, 3)
            on3r_plotting.plot_3d_scene(model, query_kpts, ref_kpts, poselib_cam_query, xyz,
                                        data, batch_ind, img_query, imgs_refs)

        if performance_mode == "low_error":
            scheduler.step()
            if epoch == 50:
                manual_lr = 1e-4  # for example
                for g in optimizer.param_groups:
                    g['lr'] = manual_lr
                #print(f"Manually set lr to {manual_lr:.2e} at epoch {epoch}")
        else:
            if epoch == 50:
                scheduler.step()
            else:
                old_xyz, break_check, scheduler, old_recall = on3ru.step_and_break_check(
                    output, model, old_xyz, old_recall, epoch, scheduler, lr_end, ref_kpts,
                    step_check_interval=50)

        if break_check:
            break
    return None


def train(
    model, query_kpts, ref_kpts, scaler_query, poselib_cam_query, data=None, batch_ind=None,
    performance_mode="speed", lr_start=0.0003, img_query=None, imgs_refs=None, plot=False
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)
    if performance_mode == "low_error":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(model.max_epochs/6), gamma=0.3)
    elif performance_mode == "speed":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.30)
    else:
        raise ValueError(f"Unknown performance mode: {performance_mode}")
    lr_end = 1e-5
    # TODO: All scaling should be dividing by scaling scaler - 1, not scaler only.
    q_norm = query_kpts / scaler_query[None]  # (N, 2)
    q_K_norm = geo.normalized_pixels_to_kpts(
        query_kpts[:, None], model.inv_intrinsics_query[None]).squeeze(dim=1)  # (N, 2)
    r_K_norm = geo.normalized_pixels_to_kpts(ref_kpts, model.inv_intrinsics)  # (N, T, 2)
    r_loss = r_K_norm[model.pred_mask]  # (n_errors, 2)

    training_loop(
        model, optimizer, scheduler, q_norm, r_loss, ref_kpts, lr_end, performance_mode,
        query_kpts, data, batch_ind, poselib_cam_query, img_query=img_query, imgs_refs=imgs_refs,
        plot=plot
    )

    # Do pose estimation with the intitial model
    output = model(q_norm).detach()
    xyz = output[..., :3]  # (N, 3)

    model.poselib_pose_est_query = geo.get_query_camera_estimate(
        poselib_cam_query, query_kpts, xyz, max_reproj_error=16.0)  # w2cam

    if model.bundle_adjust:
        xyz = ba.pylocba_ba(model, xyz, q_K_norm, r_K_norm)

    if plot:
        on3r_plotting.plot_3d_scene(
            model, query_kpts, ref_kpts, poselib_cam_query, xyz, data, batch_ind, img_query,
            imgs_refs)

    return None

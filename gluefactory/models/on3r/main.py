import sys
import torch
import time

from . import general_utils as guti
from . import plotting
from . import utils as on3ru
from . import model as on3r_model
from . import train as on3r_train
from .stats import append_sample_to_last_chunk, update_last_line_time, read_last_line_time
from . import geometry as geo
from ...baselines.transitive_matching.transitive_matching import transitive_matching_runner
from . import moge2


@torch.enable_grad()
def estimate_abs_pose_on3r(
    conf, query_kpts, ref_kpts, scaler_query, scalers_ref, intrinsics_query, intrinsics, extrinsics,
    img_q, img_refs, qcam, data=None, batch_ind=100000, new_cam_center_db=None):
    """
    Estimate the absolute query pose with the ON3R pipeline for one sample.

    Args:
        conf: Runtime/config object. Uses `conf.on3r.*` fields for optimization,
            network, and pipeline settings.
        query_kpts (torch.Tensor): Query keypoints in pixel coordinates, shape
            `(N, 2)`.
        ref_kpts (torch.Tensor): Matched reference keypoints in pixel coordinates,
            shape `(N, T, 2)`. Invalid entries are expected to be `-1`.
        scaler_query (torch.Tensor): Query image `(width, height)` used for
            normalization, shape `(2,)`.
        scalers_ref (torch.Tensor): Reference image `(width, height)` per view,
            shape `(T, 2)`.
        intrinsics_query (torch.Tensor): Query camera intrinsics, shape `(3, 3)`.
        intrinsics (torch.Tensor): Reference camera intrinsics, shape `(T, 3, 3)`.
        extrinsics (torch.Tensor): Reference world-to-camera extrinsics, shape
            `(T, 3, 4)`.
        img_q (torch.Tensor): Query image tensor in `CHW` format.
        img_refs (list[torch.Tensor]): Reference image tensors in `CHW` format,
            length `T`.
        qcam: Poselib camera object for the query image.
        data (dict | None): Optional batch dictionary for visualization/debug
            utilities during training.
        batch_ind (int): Batch index propagated to logging/visualization helpers.
        new_cam_center_db (torch.Tensor | None): If using normalized poses, the
            camera center used for normalization. Shape `(3,)`. (for plotting)

    Returns:
        tuple: `(pose_est_query, model)` where `pose_est_query` is the poselib
        absolute pose estimate (world-to-camera) and `model` is the trained
        `model.ON3R` instance.

    Raises:
        SystemExit: If `query_kpts` is empty or `None`.
    """
    # If no valid matches, skip
    if query_kpts is None or len(query_kpts) == 0:
        sys.exit(f"[Warning] No valid training samples. Should return identity pose.")

    torch.manual_seed(0)  # For reproducibility
    dt = query_kpts.dtype
    dev = query_kpts.device

    # PARAMS
    max_epochs = conf.on3r.hyperparameters.max_epochs

    moge2_model = moge2.get_model(dev) if conf.on3r.hyperparameters.use_moge2_depth else None
    pred_mask = ref_kpts[..., 0] != -1  # (N, T)

    # Get depths maps from MoGe-2
    if conf.on3r.hyperparameters.use_moge2_depth:
        depths_refs_hw = [moge2.get_depth_map(img_ref, moge2_model, f=intrinsics[i, 0, 0]) for i, img_ref in enumerate(img_refs)]
        kpt_d_r_unfilt = torch.stack([moge2.sample_depth_map(depths_ref_hw, ref_kpts[:, j])
                                      for j, depths_ref_hw in enumerate(depths_refs_hw)], dim=1)
        kpt_depths_refs = kpt_d_r_unfilt[pred_mask]  # (n_error_refs)
    else:
        kpt_depths_refs = None
        print(f"[Warning] Using dummy depths, not using MoGe-2 depth maps.")

    model = on3r_model.ON3R(
        pred_mask, intrinsics_query=intrinsics_query,
        intrinsics=intrinsics, extrinsics=extrinsics, ref_img_dims=scalers_ref,
        ref_kpts=ref_kpts, max_epochs=max_epochs,
        hidden_dim=conf.on3r.network.hidden_dims,
        positional_encoding_frequencies=conf.on3r.hyperparameters.positional_encoding_frequencies,
        n_layers=conf.on3r.network.n_layers, dev=dev, dt=dt,
        bundle_adjust=conf.on3r.pipeline.bundle_adjust,
        cauchy_scaler=conf.on3r.hyperparameters.cauchy_scaler,
        kpt_depths_refs=kpt_depths_refs,
        moge2_depth_loss_scaler=conf.on3r.hyperparameters.moge2_depth_loss_scaler,
        s_min_depth=conf.on3r.hyperparameters.s_min_depth,
        s_scaler_depth=conf.on3r.hyperparameters.s_scaler_depth,
        s_max=conf.on3r.hyperparameters.s_max,
        query_kpts=query_kpts, new_cam_center_db=new_cam_center_db,
        bias_scale=conf.on3r.hyperparameters.normalize_poses.bias_scale,
    )
    on3r_model.custom_layernorm_init(model)

    img_query = img_q.permute(1, 2, 0)
    img_refs_ = [img_ref.permute(1, 2, 0) for img_ref in img_refs]
    on3r_train.train(
        model, query_kpts, ref_kpts, scaler_query, qcam, data=data, batch_ind=batch_ind,
        performance_mode=conf.on3r.hyperparameters.performance_mode,
        lr_start=conf.on3r.hyperparameters.lr_start, img_query=img_query, imgs_refs=img_refs_,
        plot=conf.on3r.plot
    )
    if conf.on3r.hyperparameters.normalize_poses.bias_scale > 0:
        model = geo.scale_back(model)

    return model.poselib_pose_est_query, model


@torch.enable_grad()
def estimate_abs_pose_on3r_b(pred, conf, data):
    """
    pred (dict): Contains the predicted keypoints and descriptors.
    conf (dict): Contains the configuration parameters.
    data (dict): Contains the image data.
    """
    t1_extra = time.time()
    torch.manual_seed(0)  # For reproducibility
    p_keys = list(pred.keys())
    b, n_kpts, _ = pred[p_keys[0]]['keypoints0'].shape
    tuple_length = len(pred)
    dt = pred[p_keys[0]]['keypoints0'].dtype
    dev = pred[p_keys[0]]['keypoints0'].device

    matches = torch.full((b, n_kpts, tuple_length), -2, dtype=torch.long, device=dev)
    for i, key in enumerate(pred.keys()):
        matches[:, :, i] = pred[key]["matches0"]

    query_kpt_inds = torch.arange(n_kpts, device=dev, dtype=matches.dtype)

    # Get image dimensions for normalization
    query_img_dims = guti.get_query_img_dims(data)  # (b, 2)    (xy max dims)
    ref_img_dims   = guti.get_ref_img_dims(data)    # (b, T, 2) (xy max dims)

    # Get intrinsics
    intrinsics_query = torch.zeros((b, 3, 3), device=dev, dtype=dt)
    for i, key in enumerate(data.keys()):
        data_key = data[key]
        intrinsics_query[:, :, :] = data_key['view0']['camera'].calibration_matrix()

    intrinsics = torch.zeros((b, tuple_length, 3, 3), device=dev, dtype=dt)
    extrinsics = torch.zeros((b, tuple_length, 3, 4), device=dev, dtype=dt)
    for i, key in enumerate(data.keys()):
        data_key = data[key]
        extrinsics[:, i, :, :3] = data_key['view1']['T_w2cam'].R
        extrinsics[:, i, :, 3] = data_key['view1']['T_w2cam'].t
        intrinsics[:, i, :, :] = data_key['view1']['camera'].calibration_matrix()
    t2_extra = time.time()

    t0 = time.time()
    for i in range(b):
        t3_extra = time.time()
        matches_as_tracks = torch.cat((query_kpt_inds[:, None], matches[i]), dim=1)
        query_kpts, _, ref_kpts, _ = on3ru.get_keypoint_matches(
            pred, matches_as_tracks, i, 1.0)
        
        scene = data['query_to_ref_0']['view0']['scene'][i]
        t4_extra = time.time()

        scaler_query    = query_img_dims[i]
        scalers_ref     = ref_img_dims[i]  # (T, 2)

        img_q = data['query_to_ref_0']['view0']['image'][i][:, :scaler_query[1], :scaler_query[0]]
        img_refs = []
        for j in range(tuple_length):
            img_ref = data[f'query_to_ref_{j}']['view1']['image'][i][:, :scalers_ref[j, 1], :scalers_ref[j, 0]]
            img_refs.append(img_ref)

        t_start_roma = time.time()
        if conf.on3r.pipeline.roma.use:
            from .roma import get_roma_matches
            query_kpts, ref_kpts = get_roma_matches(
                scaler_query, scalers_ref, dev, tuple_length, data, i,
                num=conf.on3r.pipeline.roma.num_samples, build_tracks=conf.on3r.pipeline.roma.build_tracks,
                tracks_method=conf.on3r.pipeline.roma.tracks_method, test=conf.on3r.pipeline.roma.run_tests,
                resample=conf.on3r.pipeline.roma.resample, plot=conf.on3r.pipeline.roma.plot
            )
        pred_mask = ref_kpts[..., 0] != -1  # (N, T)
        time_roma = time.time() - t_start_roma if conf.on3r.pipeline.roma.use else 0.0
        qcam = geo.get_poselib_camera_query(data, i)

        if conf.on3r.hyperparameters.normalize_poses.mean:
            new_cam_center_db = geo.get_mean_cam_center(extrinsics[i])
            extrinsics_ = geo.center_extrinsics(extrinsics[i], new_cam_center_db)
        elif conf.on3r.hyperparameters.normalize_poses.median:
            new_cam_center_db = geo.get_median_cam_center(extrinsics[i])
            extrinsics_ = geo.center_extrinsics(extrinsics[i], new_cam_center_db)
        else:
            new_cam_center_db = None
            extrinsics_ = extrinsics[i].clone()
        t1 = time.time()

        pose_on3r, model = estimate_abs_pose_on3r(
            conf, query_kpts, ref_kpts, scaler_query, scalers_ref, intrinsics_query[i],
            intrinsics[i], extrinsics_, img_q, img_refs, qcam, data, i, new_cam_center_db
        )

        # Move the pose back to original coordinate system
        if conf.on3r.hyperparameters.normalize_poses.mean or conf.on3r.hyperparameters.normalize_poses.median:
            pose_on3r = geo.move_pose_back_to_original_cs(pose_on3r, new_cam_center_db)

        t2 = time.time()
        # Saving timings ours
        # NOTE: These timings are not valid when roma is run since we need to exclude the time
        #       taken to compute the lightglue matches, but this is not that much in comparison.
        extra_time = (t2_extra - t1_extra) + (t4_extra - t3_extra) + (t2 - t1) + time_roma
        out_file = conf.on3r.experiment.timings.out_folder + '/ours_timings.txt'
        update_last_line_time(out_file, extra_time)

        error_R, error_t = geo.get_absolute_pose_error(pose_on3r, data, i, scene)
        print(f"Pose error: {error_R:.2f} degrees, {error_t:.2f} m")
        if conf.on3r.run_baselines:
            from ...baselines.relative_pose.relative_pose import absolute_pose_from_motion_averaging
            from ...baselines.vggt.absolute_pose import predict_absolute_pose_runner as vggt_baseline
            from ...baselines.vggt.absolute_pose import get_model as get_vggt_model
            from ...baselines.reloc3r.absolute_pose import get_model as get_reloc3r_model
            from ...baselines.reloc3r.absolute_pose import predict_absolute_pose_runner as reloc3r_baseline
            from ...baselines.ace.ace_runner import run_ace

            # Run transitive matching baseline
            t1_tm = time.time()
            kpts_q, kpts_ref = on3ru.pred_to_kpts(pred, batch_ind=i)

            n_tracks = ref_kpts.shape[0]
            matches_qr = torch.arange(n_tracks)[:, None].repeat(1, tuple_length)
            matches_qr[~pred_mask] = -1
            qkt = query_kpts.cpu().numpy()
            rkt = ref_kpts.swapaxes(0, 1).cpu().numpy()

            pose_transitive, n_3d_pts_tm = transitive_matching_runner(
                data, qkt, rkt, matches_qr, i, mode=conf.on3r.baselines.transitive_matching.mode,
                ba=conf.on3r.pipeline.bundle_adjust)
            t2_tm = time.time()
            out_file = conf.on3r.experiment.timings.out_folder + '/transitive_matching_timings.txt'
            extra_time = t2_tm - t1_tm
            update_last_line_time(out_file, extra_time)

            tm_error_R, tm_error_t = geo.get_absolute_pose_error(pose_transitive, data, i, scene)

            t1_ma = time.time()
            # TODO: there "should" be no difference, but since the ordering of elements is different
            #       the results are slightly different. So, when re-running all experiments, only use
            #       the upper version for all experiments (had to use it for roma).
            if conf.on3r.pipeline.roma.use:
                pose_motion_aver = absolute_pose_from_motion_averaging(data, qkt, rkt, matches_qr, i)
            else:
                pose_motion_aver = absolute_pose_from_motion_averaging(data, kpts_q, kpts_ref, matches[i], i)

            t2_ma = time.time()
            out_file = conf.on3r.experiment.timings.out_folder + '/motion_averaging_timings.txt'
            extra_time = t2_ma - t1_ma
            update_last_line_time(out_file, extra_time)
            ma_error_R, ma_error_t = geo.get_absolute_pose_error(pose_motion_aver, data, i, scene)

            vggt_model = get_vggt_model(dev)
            t1_vggt = time.time()
            pose_vggt = vggt_baseline(vggt_model, data, tuple_length, i, dev)
            t2_vggt = time.time()
            vggt_error_R, vggt_error_t = geo.get_absolute_pose_error(pose_vggt, data, i, scene)

            reloc3r_model = get_reloc3r_model(img_reso='512', device=dev)
            t1_reloc3r = time.time()
            pose_reloc3r = reloc3r_baseline(reloc3r_model, data, tuple_length, i, dev, img_reso='512')
            t2_reloc3r = time.time()
            reloc3r_error_R, reloc3r_error_t = geo.get_absolute_pose_error(pose_reloc3r, data, i, scene)

            t1_ace = time.time()
            ace_error_R, ace_error_t = run_ace(data, batch_i=i)
            t2_ace = time.time()

            print(f"Transitive matching error: {tm_error_R:.2f} degrees, {tm_error_t:.2f} m")
            print(f"Motion averaging error: {ma_error_R:.2f} degrees, {ma_error_t:.2f} m")
            print(f"VGGT error: {vggt_error_R:.2f} degrees, {vggt_error_t:.2f} m")
            print(f"Reloc3r error: {reloc3r_error_R:.2f} degrees, {reloc3r_error_t:.2f} m")
            print(f"ACE error: {ace_error_R:.2f} degrees, {ace_error_t:.2f} m")

        # PLOTTING
        plot_matches = False
        if plot_matches:
            for ref_ind, key in enumerate(pred.keys()):
                plotting.plot_matches(pred[key], data[key], i, ref_ind, matches_as_tracks, 10.0)

        # Plotting keypoints on images
        plot_2d_preds = False
        if plot_2d_preds:
            plotting.plot_2d_preds(
                data, model, query_kpts, ref_kpts, scaler_query, i, mode="train")

        warp_images = False
        if warp_images:
            plotting.warp_images(
                data, model, i, query_kpts, scaler_query, scalers_ref, interpolate=True,
                noise_std=0.050, samples=int(1e6), model_type="on3r", plot_blended_image=False)

        # Open the file for writing
        if conf.on3r.tuning_stats:
            # NOTE: use batch size 1 when running official timings
            n_3d_pts = query_kpts.shape[0]
            time_file = conf.on3r.experiment.timings.out_folder + '/ours_timings.txt'
            time_ours = read_last_line_time(time_file)
            append_sample_to_last_chunk(conf.on3r.stats_file, None, time_ours, error_R, error_t,
                                        n_3d_pts)
            if conf.on3r.run_baselines:
                time_file = conf.on3r.experiment.timings.out_folder + '/transitive_matching_timings.txt'
                time_trans = read_last_line_time(time_file)
                append_sample_to_last_chunk(conf.on3r.stats_file_transitive, None, time_trans,
                                            tm_error_R, tm_error_t, n_3d_pts_tm)
                time_file = conf.on3r.experiment.timings.out_folder + '/motion_averaging_timings.txt'
                time_ma = read_last_line_time(time_file)
                append_sample_to_last_chunk(conf.on3r.stats_file_motion_averaging, None, time_ma,
                                            ma_error_R, ma_error_t, 0)
                append_sample_to_last_chunk(conf.on3r.stats_file_vggt, None, t2_vggt-t1_vggt,
                                            vggt_error_R, vggt_error_t, 0)
                append_sample_to_last_chunk(conf.on3r.stats_file_reloc3r, None, t2_reloc3r-t1_reloc3r,
                                            reloc3r_error_R, reloc3r_error_t, 0)
                append_sample_to_last_chunk(conf.on3r.stats_file_ace, None, t2_ace-t1_ace,
                                            ace_error_R, ace_error_t, 0)
        pass

    print(f"Batch Done with total time: {time.time()-t0:.2f}")
    return None

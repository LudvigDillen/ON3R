from gluefactory.settings import DATASET
import sys
from pathlib import Path

_ROMA_REPO = Path(__file__).resolve().parents[3] / "RoMa"
if str(_ROMA_REPO) not in sys.path:
    sys.path.insert(0, str(_ROMA_REPO))

from romatch import roma_indoor, roma_outdoor
import torch
from functools import lru_cache
import time

from . import utils as on3ru
from .dense_tracks import plots as pl
from .dense_tracks.matching import get_match_data
from .dense_tracks.geometry import get_all_sampson_error_handler, get_sampson_error_tracks
from .dense_tracks.lp import run_lp
from .dense_tracks.resample import get_query_kpts_to_resample_from_conf_based, resample_matches
from .dense_tracks.utils import get_ref_kpts_tracks
from .dense_tracks.naive import create_naive_tracks

@lru_cache(maxsize=1)
def get_model(device, indoor):
    """Return the same roma model instance every time."""
    if indoor:
        model = roma_indoor(device=device)
    else:
        model = roma_outdoor(device=device)
    model.eval()
    return model


def get_roma_matches(scaler_query, scalers_ref, dev, tuple_length, data, batch_ind, num=1000,
                     build_tracks=True, tracks_method="good_naive", test=False, resample=False,
                     plot=False):
    assert resample is False, "I have not handled the update of ref_kpts_tracks to all_ref_kpts"
    assert tracks_method in ["naive", "good_naive", "lp"], "Unknown tracks_method"
    indoor = True if DATASET == "scannet1500" else False
    q_pth = on3ru.get_img_path(batch_ind, data['query_to_ref_0'], 0)
    r_pths = [on3ru.get_img_path(batch_ind, data[f'query_to_ref_{j}'], 1) for j in range(tuple_length)]
    img_paths = [q_pth] + r_pths
    roma_model = get_model(dev, indoor)

    all_warps, all_matches, all_certainties, all_query_kpts, all_ref_kpts = get_match_data(
        scaler_query, scalers_ref, roma_model, img_paths, dev, num=num)

    if build_tracks:
        query_kpts, ref_kpts = update_kpts_with_tracks(
            all_query_kpts, data, tuple_length, tracks_method, all_certainties, all_ref_kpts,
            dev, resample, plot, test, all_matches, scaler_query, scalers_ref, all_warps,
            roma_model, img_paths, num, batch_ind)
    else:
        query_kpts = all_query_kpts.reshape(-1, 2)
        ref_kpts = torch.full((tuple_length*num, tuple_length, 2), -1.0, device=dev, dtype=all_ref_kpts.dtype)  # (tuple_length*num, tuple_length, 2)
        for j in range(tuple_length):
            ref_kpts[j*num:(j+1)*num, j, :] = all_ref_kpts[j]

    return query_kpts, ref_kpts


def update_kpts_with_tracks(
    all_query_kpts, data, tuple_length, tracks_method, all_certainties, all_ref_kpts, dev, resample,
    plot, test, all_matches, scaler_query, scalers_ref, all_warps, roma_model, img_paths, num,
    batch_ind):
    query_dists = torch.norm((all_query_kpts[None, :, None] - all_query_kpts[:, None, :, None]), dim=-1)
    poses = [data['query_to_ref_0']['view0']['T_w2cam'][batch_ind]]
    poses += [data[f'query_to_ref_{i}']['view1']['T_w2cam'][batch_ind] for i in range(tuple_length)]
    intrinsics = [data['query_to_ref_0']['view0']['camera'][batch_ind]]
    intrinsics += [data[f'query_to_ref_{i}']['view1']['camera'][batch_ind] for i in range(tuple_length)]

    # Run linear program
    if tracks_method == "lp":
        Dc_i = all_certainties[:, None, :, None]
        Dc_k = all_certainties[None, :, None, :]
        confidence_combo = (Dc_k + Dc_i)/2
        all_sampson_errors = get_all_sampson_error_handler(poses[1:], intrinsics[1:], all_ref_kpts, dev)

        tracks = run_lp(query_dists, all_sampson_errors, confidence_combo, dev, thresh_s=0.95,
                        thresh_sampson=5, topk=3, thresh_query=5, test=test)  # (n_edges, 4)
        query_kpts_tracks = get_query_kpts_to_resample_from_conf_based(tracks, all_certainties, all_matches, scaler_query)  # (n_tracks, 2)
        if resample:
            query_kpts_tracks, ref_kpts_tracks = resample_matches(
                tracks, scaler_query, scalers_ref, query_kpts_tracks, all_warps, roma_model, scaler_query)
            if plot:
                sampson_error_tracks, _ = get_sampson_error_tracks(tracks, all_sampson_errors)
                resampled_sampson_error_tracks, invalid_sampson_error_tracks = pl.get_resampled_sampson_errors(ref_kpts_tracks, tracks, poses[1:], intrinsics[1:])
                pl.plot_sampson_errors_list_ecdf_old([sampson_error_tracks, resampled_sampson_error_tracks], invalid_sampson_error_tracks, mode="", logscale=True, labels=["Original", "Resampled"])
        else:
            ref_kpts_tracks = get_ref_kpts_tracks(tracks, all_ref_kpts)
        if plot:
            pl.plot_two_tracks(img_paths, query_kpts_tracks, ref_kpts_tracks, tracks, q=0, ii=0, jj=1)

    # Run Naive Version
    elif tracks_method == "naive":
        tracks, query_kpts_tracks = create_naive_tracks(
            all_query_kpts, all_sampson_errors=None, improved_version=False, px_thresh=5, run_test=test)
    else:
        all_sampson_errors = get_all_sampson_error_handler(poses[1:], intrinsics[1:], all_ref_kpts, dev)
        tracks, query_kpts_tracks = create_naive_tracks(
            all_query_kpts, all_sampson_errors, improved_version=True, px_thresh=5, run_test=test)

    if plot:
        all_sampson_errors = get_all_sampson_error_handler(poses[1:], intrinsics[1:], all_ref_kpts, dev)
        pl.plot_sampson_errors_ecdf(tracks, all_sampson_errors, mode=tracks_method)

    print(f"Number of tracks ({tracks_method}): {tracks.shape[0]}")

    # Setup with inds
    all_query_kpts_ = all_query_kpts.clone()  # (tuple_length, num, 2)
    n_tracks = tracks.shape[0]
    tuple_inds = torch.arange(tuple_length, device=tracks.device)[:, None].repeat(1, n_tracks)  # (tuple_length, n_tracks)
    tracks_mask = (tracks != -1).T  # (tuple_length, n_tracks)
    tracks_mask_only_first = torch.zeros_like(tracks_mask)
    tracks_inds = torch.arange(n_tracks, device=tracks.device)

    # Assign new query kpts to first image index in track
    img_inds_first_in_track = tracks_mask.float().argmax(dim=0)  # n_tracks
    tracks_mask_only_first[img_inds_first_in_track, tracks_inds] = True  # (tuple_length, n_tracks)
    tuple_inds_in_use, tracks_in_use = tuple_inds[tracks_mask_only_first], tracks.T[tracks_mask_only_first]
    all_query_kpts_[tuple_inds_in_use, tracks_in_use] = query_kpts_tracks

    # Assign -1 to all other image indices in track
    tracks_mask_not_first = tracks_mask & ~tracks_mask_only_first
    tuple_inds_in_use_not_first = tuple_inds[tracks_mask_not_first]
    tracks_in_use_not_first = tracks.T[tracks_mask_not_first]
    all_query_kpts_[tuple_inds_in_use_not_first, tracks_in_use_not_first] = -1  # (tuple_length, n_tracks, 2)

    # Remove duplicates and reshape
    duplicate_mask = (all_query_kpts_  == -1).all(dim=2)  # (tuple_length, n_tracks)
    duplicate_mask_res = duplicate_mask.reshape(-1)
    query_kpts = all_query_kpts_.reshape(-1, 2)[~duplicate_mask_res]  # (n_unique_kpts_added, 2)

    ref_kpts_full = torch.full((tuple_length*num, tuple_length, 2), -1.0, device=dev, dtype=all_ref_kpts.dtype)  # (tuple_length*num, tuple_length, 2)
    for j in range(tuple_length):
        ref_kpts_full[j*num:(j+1)*num, j, :] = all_ref_kpts[j]

    tracks_global = tracks.clone()
    for j in range(tuple_length):
        tracks_global[tracks[:, j] != -1, j] += j * num
    row_inds = tracks_global[tracks_mask_not_first.T]
    img_inds = tuple_inds.T[tracks_mask_not_first.T]
    kpts_to_insert = ref_kpts_full[row_inds, img_inds]

    rows_to_insert_on_ = tracks_global[tracks_mask_only_first.T]
    repeats = tracks_mask.sum(dim=0) - 1
    rows_to_insert_on = rows_to_insert_on_.repeat_interleave(repeats)
    ref_kpts_full[rows_to_insert_on, tuple_inds_in_use_not_first] = kpts_to_insert
    ref_kpts = ref_kpts_full[~duplicate_mask_res]
    return query_kpts, ref_kpts


def get_roma_matches_compare(scaler_query, scalers_ref, dev, tuple_length, data, i, num=1000,
                             build_tracks=True, tracks_method="good_naive", test=False,
                             resample=False, plot=False):
    assert resample is False, "I have not handled the update of ref_kpts_tracks to all_ref_kpts"
    assert tracks_method in ["naive", "good_naive", "lp"], "Unknown tracks_method"
    indoor = True if DATASET == "scannet1500" else False
    q_pth = on3ru.get_img_path(i, data['query_to_ref_0'], 0)
    r_pths = [on3ru.get_img_path(i, data[f'query_to_ref_{j}'], 1) for j in range(tuple_length)]
    img_paths = [q_pth] + r_pths
    roma_model = get_model(dev, indoor)

    t1 = time.time()
    all_warps, all_matches, all_certainties, all_query_kpts, all_ref_kpts = get_match_data(
        scaler_query, scalers_ref, roma_model, img_paths, dev, num=num)
    t2 = time.time()

    if build_tracks:
        query_dists = torch.norm((all_query_kpts[None, :, None] - all_query_kpts[:, None, :, None]), dim=-1)
        poses = [data['query_to_ref_0']['view0']['T_w2cam']]
        poses += [data[f'query_to_ref_{i}']['view1']['T_w2cam'] for i in range(tuple_length)]
        intrinsics = [data['query_to_ref_0']['view0']['camera']]
        intrinsics += [data[f'query_to_ref_{i}']['view1']['camera'] for i in range(tuple_length)]

        all_sampson_errors = get_all_sampson_error_handler(poses[1:], intrinsics[1:], all_ref_kpts, dev)
        Dc_i = all_certainties[:, None, :, None]
        Dc_k = all_certainties[None, :, None, :]
        confidence_combo = (Dc_k + Dc_i)/2
        t3 = time.time()

        # Run linear program
        tracks_lp = run_lp(query_dists, all_sampson_errors, confidence_combo, dev, thresh_s=0.95,
                        thresh_sampson=5, topk=3, thresh_query=5, test=test)  # (n_edges, 4)
        query_kpts_tracks = get_query_kpts_to_resample_from_conf_based(tracks_lp, all_certainties, all_matches, scaler_query)  # (n_tracks, 2)
        if resample:
            query_kpts_tracks, ref_kpts_tracks = resample_matches(
                tracks_lp, scaler_query, scalers_ref, query_kpts_tracks, all_warps, roma_model, scaler_query)
            if plot:
                sampson_error_tracks, _ = get_sampson_error_tracks(tracks_lp, all_sampson_errors)
                resampled_sampson_error_tracks, invalid_sampson_error_tracks = pl.get_resampled_sampson_errors(ref_kpts_tracks, tracks_lp, poses[1:], intrinsics[1:])
                pl.plot_sampson_errors_list_ecdf_old([sampson_error_tracks, resampled_sampson_error_tracks], invalid_sampson_error_tracks, mode="", logscale=True, labels=["Original", "Resampled"])
        else:
            ref_kpts_tracks = get_ref_kpts_tracks(tracks_lp, all_ref_kpts)
        if plot:
            pl.plot_two_tracks(img_paths, query_kpts_tracks, ref_kpts_tracks, tracks_lp, q=0, ii=0, jj=1)
        t4 = time.time()

        height_q = scaler_query[1]
        width_q = scaler_query[0]

        # Run Naive Version
        naive_tracks, naive_query_kpts_tracks = create_naive_tracks(
            all_query_kpts, all_sampson_errors=None, improved_version=False, px_thresh=5, run_test=test)
        t5 = time.time()
        good_naive_tracks, good_naive_query_kpts_tracks = create_naive_tracks(
            all_query_kpts, all_sampson_errors, improved_version=True, px_thresh=5, run_test=test)
        t6 = time.time()

        if plot:
            #pl.plot_sampson_errors_ecdf(tracks_lp, all_sampson_errors, mode=tracks_method)
            #pl.plot_sampson_errors_ecdf(tracks_lp, all_sampson_errors, mode="Linear Programming")
            #pl.plot_sampson_errors_ecdf(naive_tracks, all_sampson_errors, mode="Naive MNN, only q_dists")
            #pl.plot_sampson_errors_ecdf(naive_tracks_v2, all_sampson_errors, mode="Naive MNN, q_dists and r_dists")
            pl.plot_sampson_errors_list_ecdf([tracks_lp, naive_tracks, good_naive_tracks], all_sampson_errors, mode="", logscale=True, labels=["LP", "Naive", "Good Naive"])

        print(f"Number of tracks (LP): {tracks_lp.shape[0]}")
        print(f"Number of tracks (Naive): {naive_tracks.shape[0]}")
        print(f"Number of tracks (Improved Naive): {good_naive_tracks.shape[0]}")

        print(f"Total Time: {t6 - t1:.3f}")
        print(f"  Matching time: {t2 - t1:.3f}")
        print(f"  Dists computation time: {t3 - t2:.3f}")
        print(f"  Linear program time: {t4 - t3:.3f}")
        print(f"  Naive version time: {t5 - t4:.3f}")
        print(f"  Improved naive version time: {t6 - t5:.3f}")

        if tracks_method == "lp":
            tracks = tracks_lp
            query_kpts_tracks = query_kpts_tracks
        elif tracks_method == "naive":
            tracks = naive_tracks
            query_kpts_tracks = naive_query_kpts_tracks
        elif tracks_method == "good_naive":
            tracks = good_naive_tracks
            query_kpts_tracks = good_naive_query_kpts_tracks

        # Setup with inds
        all_query_kpts_ = all_query_kpts.clone()  # (tuple_length, num, 2)
        n_tracks = tracks.shape[0]
        tuple_inds = torch.arange(tuple_length, device=tracks.device)[:, None].repeat(1, n_tracks)  # (tuple_length, n_tracks)
        tracks_mask = (tracks != -1).T  # (tuple_length, n_tracks)
        tracks_mask_only_first = torch.zeros_like(tracks_mask)
        tracks_inds = torch.arange(n_tracks, device=tracks.device)

        # Assign new query kpts to first image index in track
        img_inds_first_in_track = tracks_mask.float().argmax(dim=0)  # n_tracks
        tracks_mask_only_first[img_inds_first_in_track, tracks_inds] = True  # (tuple_length, n_tracks)
        tuple_inds_in_use, tracks_in_use = tuple_inds[tracks_mask_only_first], tracks.T[tracks_mask_only_first]
        all_query_kpts_[tuple_inds_in_use, tracks_in_use] = query_kpts_tracks

        # Assign -1 to all other image indices in track
        tracks_mask_not_first = tracks_mask & ~tracks_mask_only_first
        tuple_inds_in_use_not_first = tuple_inds[tracks_mask_not_first]
        tracks_in_use_not_first = tracks.T[tracks_mask_not_first]
        all_query_kpts_[tuple_inds_in_use_not_first, tracks_in_use_not_first] = -1  # (tuple_length, n_tracks, 2)

        # Remove duplicates and reshape
        duplicate_mask = (all_query_kpts_  == -1).all(dim=2)  # (tuple_length, n_tracks)
        duplicate_mask_res = duplicate_mask.reshape(-1)
        query_kpts = all_query_kpts_.reshape(-1, 2)[~duplicate_mask_res]  # (n_unique_kpts_added, 2)

        ref_kpts_full = torch.full((tuple_length*num, tuple_length, 2), -1.0, device=dev, dtype=all_ref_kpts.dtype)  # (tuple_length*num, tuple_length, 2)
        for j in range(tuple_length):
            ref_kpts_full[j*num:(j+1)*num, j, :] = all_ref_kpts[j]

        tracks_global = tracks.clone()
        for j in range(tuple_length):
            tracks_global[tracks[:, j] != -1, j] += j * num
        # duplicate_rows = ref_kpts_full[duplicate_mask_res].clone()  # (_, tuple_length, 2)
        duplicate_rows = ref_kpts_full[tracks_global[tracks_mask_not_first.T]]

        kpts_to_insert = duplicate_rows[tracks_mask_not_first.T].clone()
        rows_to_insert_on = tracks[tracks_mask_only_first.T] + img_inds_first_in_track*num
        ref_kpts_full[rows_to_insert_on, tuple_inds_in_use_not_first] = kpts_to_insert
        ref_kpts = ref_kpts_full[~duplicate_mask_res]
    else:
        # TODO: Check that this is correct
        query_kpts = all_query_kpts.reshape(-1, 2)
        ref_kpts = torch.full((tuple_length*num, tuple_length, 2), -1.0, device=dev, dtype=all_ref_kpts.dtype)  # (tuple_length*num, tuple_length, 2)
        for j in range(tuple_length):
            ref_kpts[j*num:(j+1)*num, j, :] = all_ref_kpts[j]

    return query_kpts, ref_kpts

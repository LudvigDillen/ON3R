import torch
import cv2
import matplotlib.pyplot as plt

from gluefactory.models.on3r.dense_tracks.geometry import get_sampson_error_all
from gluefactory.visualization import viz2d
from gluefactory.geometry.wrappers import Pose, Camera
from gluefactory.models.on3r.dense_tracks.geometry import fundamental_from_pose
from gluefactory.models.on3r.dense_tracks.geometry import get_sampson_error_tracks


def plot_images(img_paths):
    imgs = [cv2.imread(p) for p in img_paths]
    fig, axs = plt.subplots(1, len(imgs), figsize=(20, 10))
    for i, img in enumerate(imgs):
        axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[i].axis("off")
    plt.show()

def plot_sampson_errors_ecdf(tracks, all_sampson_errors, mode="", logscale=False):
    sampson_error_tracks, invalid_mask = get_sampson_error_tracks(tracks, all_sampson_errors)

    valid_errors = sampson_error_tracks[~invalid_mask].clone()
    if isinstance(valid_errors, torch.Tensor):
        valid_errors = valid_errors.cpu().numpy()

    valid_errors[valid_errors <= 0] = 1e-6
    plt.ecdf(valid_errors)
    if logscale:
        plt.xscale("log")
    plt.xlabel("Sampson Error")
    plt.ylabel("Probability")
    title_str = "ECDF of Sampson Errors"
    if mode:
        title_str += f" ({mode})"
    plt.title(title_str)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.show()


def plot_sampson_errors_list_ecdf_old(sampson_errors_list, invalid_mask, labels=None, mode="", logscale=False):
    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(sampson_errors_list))]

    for sampson_errors, label in zip(sampson_errors_list, labels):
        valid_errors = sampson_errors[~invalid_mask].clone()
        if isinstance(valid_errors, torch.Tensor):
            valid_errors = valid_errors.cpu().numpy()

        valid_errors[valid_errors <= 0] = 1e-6
        plt.ecdf(valid_errors, label=label)

    if logscale:
        plt.xscale("log")
    plt.xlabel("Sampson Error")
    plt.ylabel("Probability")
    title_str = "ECDF of Sampson Errors"
    if mode:
        title_str += f" ({mode})"
    plt.title(title_str)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.legend()
    plt.show()


def plot_sampson_errors_list_ecdf(tracks_list, all_sampson_errors, labels=None, mode="", logscale=False):
    if labels is None:
        labels = [f"Set {i+1}" for i in range(len(tracks_list))]

    for tracks, label in zip(tracks_list, labels):
        sampson_error_tracks, invalid_mask = get_sampson_error_tracks(tracks, all_sampson_errors)
        n_tracks = len(tracks)
        avr_track_length = (tracks != -1).sum() / n_tracks if n_tracks > 0 else 0

        valid_errors = sampson_error_tracks[~invalid_mask].clone()
        if isinstance(valid_errors, torch.Tensor):
            valid_errors = valid_errors.cpu().numpy()

        valid_errors[valid_errors <= 0] = 1e-6
        if valid_errors.size > 0:
            plt.ecdf(valid_errors, label=f"{label} ({len(tracks)} tracks, {avr_track_length:.2f} avg. length)")

    if logscale:
        plt.xscale("log")
    plt.xlabel("Sampson Error")
    plt.ylabel("Probability")
    tuple_length, _, num, _ = all_sampson_errors.shape
    title_str = f"ECDF of Sampson Errors with {num} matches and {tuple_length}tuples"
    if mode:
        title_str += f" ({mode})"
    plt.title(title_str)
    plt.grid(True, which="both", axis="both", alpha=0.3)
    plt.legend()
    plt.show()



def get_resampled_sampson_errors(ref_kpts_tracks, tracks, poses, intrinsics):
    tuple_length, n_tracks, _ = ref_kpts_tracks.shape
    device = ref_kpts_tracks.device
    poses_cls = [Pose.from_4x4mat(pose) for pose in poses]
    camera_cls = [Camera.from_calibration_matrix(K) for K in intrinsics]  # Exclude the reference pose
    F = fundamental_from_pose(poses_cls, camera_cls, dev=device)  # (T, T, 3, 3)

    tracks_inds = torch.arange(n_tracks, device=ref_kpts_tracks.device)
    tuple_inds = torch.arange(tuple_length, device=ref_kpts_tracks.device)
    resampled_sampson_error_tracks_ = get_sampson_error_all(ref_kpts_tracks, F, squared=False)[:, :, tracks_inds, tracks_inds]
    resampled_sampson_error_tracks = resampled_sampson_error_tracks_.swapaxes(1, 2).swapaxes(0, 1)
    invalid_mask = tracks == -1
    resampled_invalid_sampson_error_tracks = invalid_mask[:, None] | invalid_mask[..., None]  # (n_tracks, T, T)
    resampled_invalid_sampson_error_tracks[:, tuple_inds, tuple_inds] = True
    resampled_sampson_error_tracks[resampled_invalid_sampson_error_tracks] = -1.0  # invalidate errors where either element is invalid
    return resampled_sampson_error_tracks, resampled_invalid_sampson_error_tracks

def plot_two_tracks(img_paths, query_kpts, ref_kpts_tracks, tracks, q=0, ii=0, jj=1):
    imgs = [cv2.imread(p) for p in img_paths]
    img1 = cv2.cvtColor(imgs[q], cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    img2 = cv2.cvtColor(imgs[ii+1], cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    img3 = cv2.cvtColor(imgs[jj+1], cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

    two_tracks = ((tracks[:, (ii, jj)] != -1).sum(dim=1) == 2).argwhere()[:, 0]

    kpts_query = query_kpts[two_tracks][:10]
    kpts_ref1 = ref_kpts_tracks[ii][two_tracks][:10]
    kpts_ref2 = ref_kpts_tracks[jj][two_tracks][:10]

    viz2d.plot_images([img1, img2])
    viz2d.plot_matches(kpts_query.cpu(), kpts_ref1.cpu(), color="lime", ps=1, lw=1.5, a=1.0)
    viz2d.plot_images([img3, img2])
    viz2d.plot_matches(kpts_ref2.cpu(), kpts_ref1.cpu(), color="lime", ps=1, lw=1.5, a=1.0)
    plt.show()

import argparse
from pathlib import Path
import time
import h5py
import numpy as np
import torch
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
from gluefactory.utils.misc import start_debug


def sanity_plot(overlap_matrices, folder):
    # ---------------------------------------------------------------
    # Visual sanity‑check:  grid of image pairs + score histogram
    # ---------------------------------------------------------------
    # ---------- 1. flatten the dict ----------
    all_pairs = []        # (scene, fi, fj, score)
    for scene, d in overlap_matrices.items():
        for (fi, fj), s in d.items():
            all_pairs.append((scene, fi, fj, s))
    scores = [s.cpu() for *_, s in all_pairs]

    # ---------- 2. pick representative pairs ----------
    def sample_pairs(lo, hi, k=5):
        cand = [(sc, i, j, s) for sc,i,j,s in all_pairs if lo <= s < hi and i != j]
        return random.sample(cand, min(k, len(cand)))

    high  = sample_pairs(0.70, 1.01)          # 5 “easy” pairs
    mid   = sample_pairs(0.30, 0.70)          # 5 medium‑overlap
    low   = sample_pairs(0.00, 0.30)          # 5 almost no overlap
    grid  = high + mid + low                  # 15 pairs total

    # ---------- 3. visualise ----------
    ncols, nrows = 3, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    for ax, (scene, i, j, s) in zip(axes.flat, grid):
        root = Path(folder) / scene / "color"
        img_i = Image.open(root / f"{i}.jpg")
        img_j = Image.open(root / f"{j}.jpg")
        # side‑by‑side concat for quick viewing
        pair = Image.new("RGB", (img_i.width + img_j.width, img_i.height))
        pair.paste(img_i, (0, 0))
        pair.paste(img_j, (img_i.width, 0))
        ax.imshow(pair)
        ax.axis("off")  
        ax.set_title(f"{scene} • {i} ↔ {j}\nscore = {s:.3f}", fontsize=9)
    fig.suptitle("ScanNet-1500 overlap sanity-check", fontsize=14)
    plt.tight_layout(); plt.show()

    # ---------- 4. score histogram ----------
    plt.figure(figsize=(6,3))
    plt.hist(scores, bins=50)
    plt.xlabel("overlap score"); plt.ylabel("#pairs")
    plt.title("Distribution of computed overlaps")
    plt.tight_layout(); plt.show()


# Save down the overlap matrices in a global HDF5 file
# ---------------------------------------------------------------
def save_global_h5(outfile, overlap_matrices):
    """
    Saves one compressed dataset per scene, shape = (N,N), dtype=float32
    (Dense is OK here – ScanNet‑1500 has only 1 500 pairs total.
     Each scene has around 25 images or so. So, a rough estimate of the number of
     elements is 25*25*100 = 62 500 (68 050 in practice) elements total (100 scenes).)

    Parameters
    ----------
    outfile : Path or str
        Destination HDF5 file, e.g. Path(folder) / "scannet1500_overlap.h5"
    overlap_matrices : dict
        { scene_name : { (frame_i, frame_j) : score , ... } }
    """
    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(outfile, "w") as f:
        for scene, pairs in overlap_matrices.items():
            # ---- reconstruct dense matrix & canonical frame order ----
            frames = sorted({i for i,_ in pairs} | {j for _,j in pairs},
                            key=lambda s: int(s))
            idx    = {frm: k for k, frm in enumerate(frames)}

            M = np.zeros((len(frames), len(frames)), dtype=np.float32)
            for (fi, fj), s in pairs.items():
                M[idx[fi], idx[fj]] = s

            # ---- create a subgroup per scene and save both arrays ----
            g = f.create_group(scene)
            g.create_dataset(
                "matrix",
                data=M,
                compression="gzip",
                compression_opts=9
            )
            g.create_dataset(
                "frames",
                data=np.asarray(frames, dtype=object),
                dtype=str_dtype,
                compression="gzip",
                compression_opts=9
            )


def test_reading_h5(folder):
    # Reading the global HDF5 file
    with h5py.File(Path(folder) / "scannet1500_overlap.h5", "r") as f:
        scene = "scene0707_00"
        overlap_matrix = f[scene]["matrix"][:]
        frames = f[scene]["frames"][:]
        print(f"Overlap matrix for {scene}:\n{overlap_matrix}")
        print(f"Frames for {scene}:\n{frames}")


def load_depth_png(path, device):
    # mm→m
    return torch.tensor(cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0, device=device)


def load_pose(path, device):
    return torch.tensor(np.linalg.inv(np.loadtxt(path, delimiter=' ')).astype(np.float32), device=device)            # cam2world → world2cam


def load_K(path, device):
    K = np.loadtxt(path, delimiter=' ')
    return torch.tensor(K[:-1, :-1], dtype=torch.float32, device=device)


def load_intrinsics(scene_intr_dir, device):
    # read both, choose explicitly
    K_color4 = np.loadtxt(scene_intr_dir / 'intrinsic_color.txt', delimiter=' ')
    K_depth4 = np.loadtxt(scene_intr_dir / 'intrinsic_depth.txt', delimiter=' ')
    K_color  = torch.tensor(K_color4[:3, :3], dtype=torch.float32, device=device)
    K_depth  = torch.tensor(K_depth4[:3, :3], dtype=torch.float32, device=device)
    return K_color, K_depth


def show_consistency_pair(img_j_bgr, img_k_bgr, mask, score, hit_mask=None, alpha=0.5):
    """
    img_j_bgr : HqxWqx3  uint8   - query image (OpenCV order)
    img_r_bgr : HrxWrx3  uint8   - reference image
    mask      : HmxWmx   bool    - depth-consistent pixels (query→ref direction)
    alpha     : float            - overlay transparency
    """
    h_mask, w_mask = mask.shape

    # --- resize both images to mask size ------------------------------------
    img_j = cv2.resize(img_j_bgr, (w_mask, h_mask), interpolation=cv2.INTER_LINEAR)
    img_k = cv2.resize(img_k_bgr, (w_mask, h_mask), interpolation=cv2.INTER_LINEAR)

    # --- convert to RGB & normalise to 0–1 for plotting ----------------------
    img_j_rgb = cv2.cvtColor(img_j, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    img_k_rgb = cv2.cvtColor(img_k, cv2.COLOR_BGR2RGB).astype(np.float32) / 255

    # --- build green/red overlay --------------------------------------------
    overlay = np.zeros_like(img_k_rgb)
    overlay[ mask] = (0, 1, 0)     # consistent  → green
    if hit_mask is not None:
        overlay[(~mask) & hit_mask] = (1, 0, 0)     # inconsistent → red
    else:
        overlay[~mask] = (1, 0, 0)
    blended = (1 - alpha) * img_k_rgb + alpha * overlay

    # --- plot ----------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, im, title in zip(
        axs,
        [img_j_rgb, blended, img_k_rgb],
        ["Image j (resized)",
         f"Depth consistency mask (image k) (score={score:.3f})",
         "Image k (resized)"]
    ):
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def depth_bilinear(depth, u, v):
    H, W = depth.shape
    uu = 2.0 * (u / (W - 1)) - 1.0  # normalise to [‑1,1]
    vv = 2.0 * (v / (H - 1)) - 1.0
    grid = torch.stack((uu, vv), -1).view(1, -1, 1, 2)  # [1, N, 1, 2]
    return torch.nn.functional.grid_sample(depth.view(1, 1, H, W), grid, mode="bilinear",
                                           padding_mode="zeros", align_corners=True).view(-1)


def backproject(depth, K):
    """
    depth: (H,W) tensor of depth values in meters
    K:     (3,3) color camera intrinsics matrix

    Returns: (3, H, W) tensor of 3D points in camera coordinates
    3D points are in the form (x, y, z) where z is the depth value.
    The output is normalized such that the points are in the camera coordinate system.
    """
    h, w = depth.shape
    dt, dev = K.dtype, K.device
    assert depth.dtype == dt, "Depth tensor must have the same dtype as K"
    assert depth.device == dev, "Depth tensor must be on the same device as K"

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    y, x, ones = y.to(dt).to(dev), x.to(dt).to(dev), torch.ones_like(y, dtype=dt, device=dev)
    pix_h = torch.stack((x, y, ones), dim=0)  # (3, h, w)
    pts_norm = torch.einsum("ij,jhw->ihw", torch.linalg.inv(K), pix_h)  # (3, h, w)
    pts_cam = pts_norm * depth[None]
    return pts_cam


def overlap_score(depth_j, depth_k, K, T_w2c_j, T_w2c_k, thresh=0.05):
    """
    We calculate two sets of depth values
    dk which is the sampled depth in camera k at projected pixel
    zk which is the depth of the projected point in camera j coordinate system

    depth_j: (h, w) tensor of depth values in camera j
    depth_k: (h, w) tensor of depth values in camera k
    K:       (3, 3) depth intrinsics matrix
    T_w2c_j: (4, 4) pose for w2cam_j
    T_w2c_k: (4, 4) pose for w2cam_k
    thresh:  float, threshold for depth difference to consider overlap

    Returns: overlap_score: float, the mean score of overlap
    """
    pts_i = backproject(depth_j, K)  # (3, h, w)
    pts_i_h = torch.vstack((pts_i, torch.ones_like(pts_i[:1])))  # (4, h, w)

    # Change to world coordinates
    T_c2w_i = torch.linalg.inv(T_w2c_j)  # (4, 4)
    pts_w_h = torch.einsum("ij,jhw->ihw", T_c2w_i, pts_i_h)  # (4, h, w)

    # Change to camera k coordinate system
    pts_j = torch.einsum("ij,jhw->ihw", T_w2c_k[:3], pts_w_h)  # (3, h, w)
    # To pixels in camera k
    uv   = torch.einsum("ij,jhw->ihw", K, pts_j)  # (3, h, w)
    uv   = uv[:2] / uv[2:3]  # (2, h, w)

    # Calculate sampled depth in camera k
    h, w = depth_k.shape
    in_front = pts_j[2] > 0  # (h, w)
    inside = in_front & (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)

    # Bilinear sampling is slightly better than nearest neighbour
    dk = depth_bilinear(depth_k, uv[0, inside], uv[1, inside])  # n_inside
    # dk  = depth_k[uv_int_y, uv_int_x]  # n_inside

    # Calculate depth in camera k coordinates (of the 3D point to be projected)
    z_k  = pts_j[2]  # depth in camera coordinates (h, w)
    zk = z_k[inside]

    overlap_field  = (dk > 0) & (torch.abs(dk - zk) < thresh)  # n_inside

    uv_int = uv[:, inside].round().int()  # (2, n_inside)
    uv_int_x = torch.clamp(uv_int[0], min=0, max=w-1)  # n_inside
    uv_int_y = torch.clamp(uv_int[1], min=0, max=h-1)  # n_inside
    mask_full = torch.zeros_like(depth_j, dtype=torch.bool)
    hit_mask = torch.zeros_like(depth_j, dtype=torch.bool)
    hit_mask[uv_int_y, uv_int_x] = 1
    mask_full[uv_int_y, uv_int_x] = overlap_field

    valid_j = (depth_j > 0).sum().item()  # Count valid depth pixels in camera j
    overlap_score = overlap_field.sum() / valid_j if valid_j > 0 else 0.0
    return overlap_score, mask_full, hit_mask


def calculate_overlap(folder, first_scene, last_scene, device, plot=False):
    overlap_matrices = {}
    all_imgs = {}
    with torch.inference_mode():
        for i in range(int(first_scene), int(last_scene) + 1):
            t1 = time.time()
            print(f"Processing scene {i:04d}...")
            subfolder = f"scene{i:04d}_00"
            overlap_matrices[subfolder] = {}
            scene_root = Path(folder) / subfolder
            color_dir = scene_root / "color"
            depth_dir = scene_root / "depth"
            intrinsics_dir = scene_root / "intrinsic"
            pose_dir = scene_root / "pose"

            _, K_depth = load_intrinsics(intrinsics_dir, device)
            frames     = sorted((f.stem for f in color_dir.glob('*.jpg')),
                                key=lambda s: int(s))
            N          = len(frames)

            # Read data
            depths = {f: load_depth_png(depth_dir / f'{f}.png', device) for f in frames}  # in meters
            # poses are w2cam
            poses  = {f: load_pose(pose_dir / f'{f}.txt', device) for f in frames}
            imgs = {f: cv2.imread(str(color_dir / f'{f}.jpg'), cv2.IMREAD_COLOR) for f in frames}
            all_imgs[subfolder] = imgs

            overlap_mat = torch.zeros((N, N), dtype=torch.float32, device=device)
            for j, fj in enumerate(frames):
                for k, fk in enumerate(frames):
                    if j == k:
                        overlap_mat[j, k] = 1.0
                    else:
                        overlap_mat[j, k], mask, hit_mask = overlap_score(
                            depths[fj], depths[fk], K_depth, poses[fj], poses[fk])

                        if plot:
                            show_consistency_pair(
                                all_imgs[subfolder][fj],   # query BGR
                                all_imgs[subfolder][fk],   # reference BGR
                                mask.cpu().numpy(),         # bool mask
                                score=overlap_mat[j, k].item(),  # overlap score
                                hit_mask=hit_mask.cpu().numpy(),  # (2, n_inside)
                            )

            for j, fj in enumerate(frames):
                for k, fk in enumerate(frames):
                    overlap_matrices[subfolder][(fj, fk)] = (overlap_mat[j, k] + overlap_mat[k, j])/2  # Symmetric score

            print(f"Scene {i:04d} processed in {time.time() - t1:.2f} seconds.")
    return overlap_matrices, all_imgs


if __name__ == "__main__":
    """ Use one of the following to run the script:
    python -m gluefactory.datasets.calculate_scannet_overlap
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
    parser.add_argument("-f", "--folder", type=str, default="/home2/lu2277di/data/scannet1500/")
    parser.add_argument("-fs", "--first_scene", type=str, default="0707")
    parser.add_argument("-ls", "--last_scene", type=str, default="0806")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot the results")
    args = parser.parse_intermixed_args()
    if args.debug:
        start_debug()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    covisility_mat, all_imgs = calculate_overlap(args.folder, args.first_scene, args.last_scene, dev, args.plot)

    sanity_plot(covisility_mat, args.folder)
    save_global_h5(Path(args.folder) / "scannet1500_overlap.h5", covisility_mat)

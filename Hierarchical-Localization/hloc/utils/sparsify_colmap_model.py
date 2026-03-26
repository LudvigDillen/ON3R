import argparse
import debugpy
import shutil
from pathlib import Path
import numpy as np

from hloc.utils.read_write_model import read_model, write_model, Point3D


def start_debug():
    debugpy.listen(5678)
    print("Wait for debugger!")
    debugpy.wait_for_client()
    print("Attached!")


def sparsify_model(model_dir: Path, out_dir: Path, sparsity: float, seed: int = 0, deterministic: bool = False,
                   min_images: int = 3):
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Read ---
    cameras, images, points3D = read_model(model_dir, ext=".bin")  # returns dicts: id -> Camera/Image/Point3D

    image_ids = np.array(sorted(images.keys()))
    n_images = len(image_ids)
    if n_images == 0:
        raise ValueError("Model has no images.")

    k = max(int(round(sparsity * n_images)), min_images)  # number of images to keep
    if k == 0 or k == n_images:
        exit(f"INVALID! Sparsity value leads to {k}/{n_images} kept images.")

    # --- Choose which images to keep ---
    if deterministic:
        # Deterministic subset: sort by name and take the first k
        names = [(iid, images[int(iid)].name) for iid in image_ids]
        names.sort(key=lambda x: x[1])
        keep_ids = np.array([iid for iid, _ in names[:k]], dtype=image_ids.dtype)
    else:
        rng = np.random.RandomState(seed)
        keep_ids = rng.choice(image_ids, size=k, replace=False) if k > 0 else np.array([], dtype=image_ids.dtype)

    keep_ids_set = set(int(i) for i in keep_ids)

    # --- Keep only cameras referenced by kept images ---
    used_camera_ids = set()
    for iid in keep_ids_set:
        used_camera_ids.add(images[iid].camera_id)
    cameras_kept = {cid: cam for cid, cam in cameras.items() if cid in used_camera_ids}

    # --- Prune 3D points: keep those that have at least one obs in kept images ---
    points3D_kept = {}
    for pid, p in points3D.items():
        # Get track arrays from Point3D (hloc uses .image_ids and .point2D_idxs)
        image_ids_track = getattr(p, "image_ids", None)
        p2d_idxs = getattr(p, "point2D_idxs", getattr(p, "point2D_ids", None))
        if image_ids_track is None or p2d_idxs is None:
            raise RuntimeError("Unexpected Point3D structure; expected attributes 'image_ids' and 'point2D_idxs'.")

        # Filter the track to kept images
        mask = np.array([int(im_id) in keep_ids_set for im_id in image_ids_track], dtype=bool)
        if mask.any():
            # Create a shallow copy with filtered track
            p_new = Point3D(
                id=p.id,
                xyz=p.xyz.copy(),
                rgb=p.rgb.copy(),
                error=p.error,
                image_ids=image_ids_track[mask].copy(),
                point2D_idxs=p2d_idxs[mask].copy(),
            )
            points3D_kept[pid] = p_new
        # else: drop the point (no remaining observations)

    images_kept = {iid: images[iid] for iid in keep_ids_set}
    # --- Write pruned model ---
    write_model(cameras_kept, images_kept, points3D_kept, out_dir, ext=".bin")

    # --- Copy optional sidecar files if present ---
    for fname in ["aachen_v_1_1.nvm", "project.ini", "database_intrinsics_v1_1.txt"]:
        fsrc = model_dir / fname
        if fsrc.exists():
            shutil.copy2(fsrc, out_dir / fname)

    print(f"Kept {len(images_kept)}/{n_images} images "
          f"({len(cameras_kept)}/{len(cameras)} cameras, "
          f"{len(points3D_kept)}/{len(points3D)} points).")
    return keep_ids

def main():
    """
    python -m hloc.utils.sparsify_colmap_model \
        --model /home2/lu2277di/data/aachen/3D-models/aachen_v_1_1 --sparsity 0.01

    python -m hloc.utils.sparsify_colmap_model \
        --model /home2/lu2277di/data/outputs/cambridge/CambridgeLandmarks_Colmap_Retriangulated_1024px/all/ShopFacade/model_train \
        --sparsity 0.01
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="Path to source COLMAP model dir (with .bin files)")
    ap.add_argument("--sparsity", type=float, required=True, help="Fraction of images to keep in (0..1)")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: model parent / name_sparsityX)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (ignored if --deterministic)")
    ap.add_argument("--deterministic", action="store_true", help="Pick first k images by sorted name instead of random")
    ap.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    ap.add_argument("--min_images", type=int, default=3, help="Minimum number of images to keep (default: 3)")
    args = ap.parse_args()
    if args.debug:
        start_debug()

    model_dir = args.model
    sparsity = args.sparsity
    assert sparsity > 0 and sparsity < 1, "Sparsity must be between 0 and 1."
    sparsity_str = "_sparsity"+ str(args.sparsity).replace(".", "_")

    if args.out is None:
        # build e.g. aachen_v_1_1_sparsity0.5 next to the input folder
        parent = model_dir.parent
        out = parent / f"{model_dir.name}{sparsity_str}"
    else:
        out = args.out

    keep_ids = sparsify_model(model_dir, out, sparsity, seed=args.seed, deterministic=args.deterministic,
                              min_images=args.min_images)
    # If you want, write the kept image list:
    (out / "kept_image_ids.txt").write_text("\n".join(map(str, sorted(map(int, keep_ids)))))

if __name__ == "__main__":
    main()

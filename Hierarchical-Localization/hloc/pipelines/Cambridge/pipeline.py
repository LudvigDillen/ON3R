import argparse
from pathlib import Path
import debugpy

from ... import (
    extract_features,
    localize_sfm,
    logger,
    match_features,
    pairs_from_covisibility,
    pairs_from_retrieval,
    triangulation,
)
from .utils import create_query_list_with_intrinsics, evaluate, scale_sfm_images
from ...utils.parsers import parse_retrieval

SCENES = ["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch", "GreatCourt"]


def start_debug():
    debugpy.listen(5678)
    print("Wait for debugger!")
    debugpy.wait_for_client()
    print("Attached!")


def print_dummy_results():
    out = "\nPercentage of test images localized within:"
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        #ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {0:.2f}%"
    logger.info(out)


def run_scene(images, gt_dir, outputs, results, num_covis, num_loc, matcher, sparsity_str):
    #ref_sfm_sift = gt_dir / "model_train"
    ref_sfm_sift = gt_dir / f"model_train{sparsity_str}"
    test_list = gt_dir / "list_query.txt"

    outputs.mkdir(exist_ok=True, parents=True)
    ref_sfm = outputs / f"sfm_superpoint+{matcher}{sparsity_str}"
    ref_sfm_scaled = outputs / f"sfm_sift_scaled{sparsity_str}"
    query_list = outputs / "query_list_with_intrinsics.txt"
    sfm_pairs = outputs / f"pairs-db-covis{num_covis}{sparsity_str}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{num_loc}{sparsity_str}.txt"

    feature_conf = {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 4096,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    if matcher == "lightglue":
        matcher_conf = match_features.confs[f"superpoint+{matcher}"]
    else:
        matcher_conf = match_features.confs[f"{matcher}"]
    retrieval_conf = extract_features.confs["netvlad"]

    create_query_list_with_intrinsics(
        gt_dir / "empty_all", query_list, test_list, ext=".txt", image_dir=images
    )
    with open(test_list, "r") as f:
        query_seqs = {q.split("/")[0] for q in f.read().rstrip().split("\n")}

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        num_loc,
        db_model=ref_sfm_sift,
        query_prefix=query_seqs,
    )

    features = extract_features.main(feature_conf, images, outputs, as_half=True)
    pairs_from_covisibility.main(ref_sfm_sift, sfm_pairs, num_matched=num_covis)
    pairs = parse_retrieval(sfm_pairs)
    if len(pairs) == 0:
        open(results, "w").close()  # create empty file
        return False

    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf["output"], outputs
    )

    scale_sfm_images(ref_sfm_sift, ref_sfm_scaled, images)
    triangulation.main(
        ref_sfm, ref_sfm_scaled, images, sfm_pairs, features, sfm_matches
    )

    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    localize_sfm.main(
        ref_sfm,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=True,
    )
    return True


if __name__ == "__main__":
    """
    python -m hloc.pipelines.Cambridge.pipeline \
        --scenes GreatCourt \
        --dataset /home2/lu2277di/data/cambridge \
        --outputs /home2/lu2277di/data/outputs/cambridge \
        --num_loc 3 -s 0.01 -d
    """
    # Replicate experiments
    # python -m hloc.pipelines.Cambridge.pipeline --dataset /home2/lu2277di/data/cambridge --outputs /home2/lu2277di/data/outputs/cambridge
    # python -m hloc.pipelines.Cambridge.pipeline -s 0.0005 --num_loc 5 --dataset /home2/lu2277di/data/cambridge --outputs /home2/lu2277di/data/outputs/cambridge
    # python -m hloc.pipelines.Cambridge.pipeline -s 0.001 --num_loc 10 --dataset /home2/lu2277di/data/cambridge --outputs /home2/lu2277di/data/outputs/cambridge
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", default=SCENES, choices=SCENES, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default="datasets/cambridge",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="outputs/cambridge",
        help="Path to the output directory, default: %(default)s",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="lightglue",
        help="Matcher config for HLOC, default: %(default)s",
    )
    parser.add_argument(
        "--num_covis",
        type=int,
        default=20,
        help="Number of image pairs for SfM, default: %(default)s",
    )
    parser.add_argument(
        "--num_loc",
        type=int,
        default=10,
        help="Number of image pairs for loc, default: %(default)s",
    )
    parser.add_argument(
        '-s', '--sparsity',
        type=float,
        default=1.0,
        help="The share of images to use for 3D model triangulation"
    )
    parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
    args = parser.parse_args()
    if args.debug:
        start_debug()
    gt_dirs = args.outputs / "CambridgeLandmarks_Colmap_Retriangulated_1024px/all"
    sparsity_str = "" if args.sparsity == 1.0 else "_sparsity"+ str(args.sparsity).replace(".", "_")

    all_results = {}
    all_success = {}
    for scene in args.scenes:
        logger.info(f'Working on scene "{scene}".')
        results = args.outputs / scene / f"results{sparsity_str}_{args.num_loc}.txt"

        if args.overwrite or not results.exists():
            success = run_scene(
                args.dataset / scene,
                gt_dirs / scene,
                args.outputs / scene,
                results,
                args.num_covis,
                args.num_loc,
                args.matcher,
                sparsity_str,
            )
        else:
            with open(results, "r") as f:
                success = len(f.read()) > 0
        all_results[scene] = results
        all_success[scene] = success

    for scene in args.scenes:
        logger.info(f'Evaluate scene "{scene}".')
        if all_success[scene]:
            evaluate(
                gt_dirs / scene / "empty_all",
                all_results[scene],
                gt_dirs / scene / "list_query.txt",
                ext=".txt",
            )
        else:
            logger.info(f"Skipping evaluation for scene {scene} as no 3D model could be created.")
            print_dummy_results()

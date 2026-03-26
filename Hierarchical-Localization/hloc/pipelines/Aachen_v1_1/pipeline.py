import argparse
from pathlib import Path
from pprint import pformat
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


def start_debug():
    debugpy.listen(5678)
    print("Wait for debugger!")
    debugpy.wait_for_client()
    print("Attached!")


def run(args):
    # Setup the paths
    dataset = args.dataset
    outputs = args.outputs  # where everything will be saved
    images = dataset / "images_upright/"
    sparsity_str = "" if args.sparsity == 1.0 else "_sparsity"+ str(args.sparsity).replace(".", "_")
    # sift_sfm = outputs / f"3D-models/aachen_v_1_1{sparsity_str}"

    reference_sfm = outputs / f"sfm_superpoint+{args.matcher}{sparsity_str}"  # the SfM model we will build

    # query_list = dataset / "queries/*_time_queries_with_intrinsics.txt"
    # queries = parse_image_lists(query_list, with_intrinsics=True)
    # print(f"Found {len(queries)} queries")

    # sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}{sparsity_str}.txt"
    loc_pairs = (
        outputs / f"pairs-query-netvlad{args.num_loc}{sparsity_str}.txt"
    )  # top-k retrieved by NetVLAD
    results = (
        outputs / f"Aachen-v1.1_hloc_superpoint+{args.matcher}_netvlad{args.num_loc}{sparsity_str}.txt"
    )

    # list the standard configurations available
    logger.info("Configs for feature extractors:\n%s", pformat(extract_features.confs))
    logger.info("Configs for feature matchers:\n%s", pformat(match_features.confs))

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs["netvlad"]
    #feature_conf = extract_features.confs["superpoint_max"]  # Res: 1600x1600
    feature_conf = extract_features.confs["superpoint_aachen"]  # Res: 1024x1024
    if args.matcher == "lightglue":
        matcher_conf = match_features.confs[f"superpoint+{args.matcher}"]
    else:
        matcher_conf = match_features.confs[f"{args.matcher}"]

    features = extract_features.main(feature_conf, images, outputs)

    # pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)
    # sfm_matches = match_features.main(
    #     matcher_conf, sfm_pairs, feature_conf["output"], outputs
    # )
    # triangulation.main(
    #     reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches
    # )

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(
        global_descriptors,
        loc_pairs,
        args.num_loc,
        query_prefix="query",
        db_model=reference_sfm,
    )
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf["output"], outputs
    )

    localize_sfm.main(
        reference_sfm,
        dataset / "queries/*_time_queries_with_intrinsics.txt",
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
    )  # not required with SuperPoint+SuperGlue


if __name__ == "__main__":
    # Run e.g. python -m hloc.pipelines.Aachen_v1_1.pipeline -s 0.05 --num_loc 3 -d
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        #default="datasets/aachen_v1.1",
        default="/home2/lu2277di/data/aachen",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        #default="outputs/aachen_v1.1",
        #default="/home2/lu2277di/data/outputs/aachen",
        default="/home/lu2277di/Projects/release_code/ON3R/assets/aachen",
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
        default=50,
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
    run(args)

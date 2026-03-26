import torch
import time
import argparse
from pathlib import Path
import pycolmap
from tqdm import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from gluefactory.models.on3r.main import estimate_abs_pose_on3r
from gluefactory.geometry.wrappers import Camera
from gluefactory.utils.misc import start_debug
from gluefactory.visualization import viz2d
import gluefactory.models.on3r.geometry as geo

from hloc.utils.parsers import parse_retrieval, parse_image_lists
from hloc.utils.read_write_model import read_model, qvec2rotmat
from hloc.utils.io import get_matches, get_keypoints, read_image
from hloc import extract_features, logger, match_features
from ..settings import DATA_PATH


def check_principal_point_at_image_center(cams):
    """
    When we convert qcam to geometry.wrappers.Camera, we assume that the principal point is at the image center.
    """
    for name, cam in cams:
        cx = cam.params[1]
        cy = cam.params[2]
        width = cam.width
        height = cam.height
        if 2*cx != width or 2*cy != height:
            exit(f"We have wrong assumption about principal point being in the image center "
                 f"(cx={cx}, xy={cy}, width={width}, height={height})")


def main(args, conf):
    # Setup paths
    dataset = args.dataset
    outputs = args.outputs  # where everything will be saved
    img_pth = dataset / "images_upright/"

    sparsity_str = "" if args.sparsity == 1.0 else "_sparsity"+ str(args.sparsity).replace(".", "_")
    reference_sfm_string = outputs / f"sfm_superpoint+{args.matcher}{sparsity_str}"  # the SfM model that will be built
    retrieval = (
        outputs / f"pairs-query-netvlad{args.num_loc}{sparsity_str}.txt"
    )  # top-k retrieved by NetVLAD
    time_str = time.strftime("%Y%m%d_%H%M%S")
    results = (
        outputs / f"Aachen-v1.1_hloc_superpoint+{args.matcher}_netvlad{args.num_loc}{sparsity_str}_on3r_{time_str}.txt"
    )

    feature_conf = extract_features.confs["superpoint_aachen"]  # Res: 1024x1024
    if args.matcher == "lightglue":
        matcher_conf = match_features.confs[f"superpoint+{args.matcher}"]
    else:
        matcher_conf = match_features.confs[f"{args.matcher}"]
    loc_matches = match_features.main(
        matcher_conf, retrieval, feature_conf["output"], outputs
    )

    # Load features
    features = extract_features.main(feature_conf, img_pth, outputs)
    max_kpts = feature_conf["model"]["max_keypoints"]

    # Load queries
    query_list = dataset / "queries/*_time_queries_with_intrinsics.txt"

    queries = parse_image_lists(query_list, with_intrinsics=True)
    print(f"Found {len(queries)} queries")

    # Load database
    logger.info("Reading the 3D model...")
    reference_sfm = pycolmap.Reconstruction(reference_sfm_string)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    cameras, images, _ = read_model(reference_sfm_string, ".bin")
    retrieval_dict = parse_retrieval(retrieval)

    # Other setup
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32
    tuple_length = args.num_loc
    t11 = 0
    t22 = 0
    t33 = 0
    t44 = 0
    check_principal_point_at_image_center(queries)
    cam_from_world = {}

    for qname, qcam in tqdm(queries):
        t1 = time.time()
        if qname not in retrieval_dict:
            exit(f"No images retrieved for query image {qname}. Skipping...")
        t2 = time.time()
        t11 += t2 - t1

        db_names = retrieval_dict[qname]
        db_ids = []
        cam_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.warning(f"Image {n} was retrieved but not in database")
                continue
            db_ids.append(db_name_to_id[n])
            cam_ids.append(images[db_name_to_id[n]].camera_id)
        t3 = time.time()
        t22 += t3 - t2

        # Get intrinsics and extrinsics
        intrinsics_query = torch.tensor(qcam.calibration_matrix(), dtype=dt, device=dev)
        intrinsics = torch.zeros((tuple_length, 3, 3), device=dev, dtype=dt)
        extrinsics = torch.zeros((tuple_length, 3, 4), device=dev, dtype=dt)
        for i, (cam_id, db_id) in enumerate(zip(cam_ids, db_ids)):
            rcam = cameras[cam_id]
            intrinsics[i, 0, 0] = rcam.params[0]
            intrinsics[i, 1, 1] = rcam.params[0]
            intrinsics[i, 0, 2] = rcam.params[1]
            intrinsics[i, 1, 2] = rcam.params[2]
            intrinsics[i, 2, 2] = 1.0

            img = images[db_id]
            extrinsics[i, :, :3] = torch.tensor(qvec2rotmat(img.qvec), dtype=dt, device=dev)
            extrinsics[i, :, 3] = torch.tensor(img.tvec, dtype=dt, device=dev)
        t4 = time.time()
        t33 += t4 - t3

        ref_kpts = torch.full((max_kpts, tuple_length, 2), -1, dtype=dt, device=dev)
        kpq = get_keypoints(features, qname) + 0.5  # COLMAP coordinates
        query_kpts = torch.tensor(kpq, dtype=dt, device=dev)

        q_img_dims = torch.tensor([qcam.width, qcam.height], dtype=dt, device=dev)
        r_img_dims = torch.zeros((tuple_length, 2), dtype=dt, device=dev)

        img_query = read_image(img_pth / qname)
        img_query = torch.tensor(img_query.copy(), dtype=dt, device=dev) / 255.0
        img_query = torch.clamp(img_query, 0.0, 1.0).swapaxes(2, 1).swapaxes(1, 0)
        img_refs = []
        for i, db_id in enumerate(db_ids):
            img = reference_sfm.images[db_id]
            matches, _ = get_matches(loc_matches, qname, img.name)
            kpr = get_keypoints(features, img.name) + 0.5  # COLMAP coordinates
            keypoints_ref = kpr[matches[:, 1]]
            ref_kpts[matches[:, 0], i, :] = torch.tensor(keypoints_ref, dtype=dt, device=dev)

            r_img_dims[i, 0] = img.camera.width
            r_img_dims[i, 1] = img.camera.height

            img_ref = read_image(img_pth / img.name)
            img_ref = torch.tensor(img_ref.copy(), dtype=dt, device=dev) / 255.0
            img_ref = torch.clamp(img_ref, 0.0, 1.0).swapaxes(2, 1).swapaxes(1, 0)
            img_refs.append(img_ref)

        q_inds_with_match = (ref_kpts[..., 0] != -1).any(dim=1)
        n_query_kpts = query_kpts.shape[0]
        q_inds_q_size = q_inds_with_match[:n_query_kpts]
        assert q_inds_with_match[n_query_kpts:].sum() == 0, "There are matches for keypoints that do not exist"
        query_kpts = query_kpts[q_inds_q_size]
        ref_kpts = ref_kpts[q_inds_with_match]

        # Some visualization
        # for i in range(tuple_length):
        #     r_ind = i
        #     q_im = read_image(img_pth / qname)
        #     r_name = reference_sfm.images[db_ids[r_ind]].name
        #     r_im = read_image(img_pth / r_name)
        #     mask = ref_kpts[:, r_ind, 0] != -1
        #     query_kpts_vis = query_kpts[mask, :].cpu()
        #     ref_kpts_vis = ref_kpts[mask, r_ind, :].cpu()

        #     # plt.figure(figsize=(15, 10))
        #     # plt.subplot(1, 2, 1)
        #     # plt.imshow(q_im)
        #     # plt.scatter(query_kpts_vis[:, 0], query_kpts_vis[:, 1], s=1, c='r')
        #     # plt.title(f"Query image {qname} with {query_kpts_vis.shape[0]} keypoints")
        #     # plt.axis('off')
        #     # plt.subplot(1, 2, 2)
        #     # plt.imshow(r_im)
        #     # plt.scatter(ref_kpts_vis[:, 0], ref_kpts_vis[:, 1], s=1, c='r')
        #     # plt.title(f"Top retrieved image {r_name} with {ref_kpts_vis.shape[0]} keypoints with matches")
        #     # plt.axis('off')
        #     # plt.show()

        #     viz2d.plot_images([q_im, r_im])
        #     viz2d.plot_matches(query_kpts_vis, ref_kpts_vis, ps=3, lw=0.2)
        #     plt.show()
        t5 = time.time()
        t44 += t5 - t4

        qcam_ = Camera.from_calibration_matrix(qcam.calibration_matrix())

        # Idea, normalize the scene by centering the cameras around the origin
        # First, get the mean camera center
        if conf.on3r.hyperparameters.normalize_poses.mean:
            new_cam_center_db = geo.get_mean_cam_center(extrinsics)
            extrinsics_ = geo.center_extrinsics(extrinsics, new_cam_center_db)
        elif conf.on3r.hyperparameters.normalize_poses.median:
            new_cam_center_db = geo.get_median_cam_center(extrinsics)
            extrinsics_ = geo.center_extrinsics(extrinsics, new_cam_center_db)
        else:
            new_cam_center_db = None
            extrinsics_ = extrinsics.clone()

        poselib_pose_est_query_centered_cs, _ = estimate_abs_pose_on3r(
            conf, query_kpts, ref_kpts, q_img_dims, r_img_dims, intrinsics_query, intrinsics,
            extrinsics_, img_query, img_refs, qcam_, new_cam_center_db=new_cam_center_db)

        # Move the pose back to original coordinate system
        if conf.on3r.hyperparameters.normalize_poses.mean or conf.on3r.hyperparameters.normalize_poses.median:
            poselib_pose_est_query = geo.move_pose_back_to_original_cs(
                poselib_pose_est_query_centered_cs, new_cam_center_db)
        else:
            poselib_pose_est_query = poselib_pose_est_query_centered_cs

        cam_from_world[qname] = poselib_pose_est_query

    with open(results, "w") as f:
        for query, t in cam_from_world.items():
            #       and this is what visuallocalization.net expects. If something seems wrong, maybe check this again.
            qvec = " ".join(map(str, t.q))
            tvec = " ".join(map(str, t.t))
            name = query.split("/")[-1]
            f.write(f"{name} {qvec} {tvec}\n")


if __name__ == "__main__":
    """
    Example usage:
    Experiment 1:
    python -m gluefactory.eval.hloc_aachen_sparsity -s 0.01 --num_loc 3 \
        --conf gluefactory/configs/superpoint+lightglue_megadepth_tuples.yaml

    Experiment 2:
    python -m gluefactory.eval.hloc_aachen_sparsity --num_loc 50 \
        --conf gluefactory/configs/superpoint+lightglue_megadepth_tuples.yaml

    To run in background, run e.g.:
    nohup python -m gluefactory.eval.hloc_aachen_sparsity -s 0.01 --num_loc 3 \
        --conf gluefactory/configs/superpoint+lightglue_megadepth_tuples.yaml \
            > outputs/nohups/testing_reproducibility_aachen_sfm_K3.txt 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_PATH / "aachen",
        help="Path to the dataset, default: %(default)s",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default="assets/aachen",
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
    parser.add_argument("--conf", type=str)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    if args.debug:
        start_debug()
    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    main(args, conf.model.matcher)

import numpy as np
import pycolmap
import os
import poselib

from . import transitive_matching_utils as tmu


def collect_transitive_matching(kpts_ref, matches_q2ref):
    # Transitive matching between query and reference keypoints
    # Returns pairwise matches between all reference images
    num_refs = kpts_ref.shape[0]
    matches = {}
    for i in range(num_refs):
        for j in range(i + 1, num_refs):
            m1 = matches_q2ref[i]
            m2 = matches_q2ref[j]
            shared_matches = np.all(np.c_[m1,m2] >= 0,axis=1)
            matches[(i, j)] = np.array([m1[shared_matches], m2[shared_matches]]).T
    return matches


def setup_colmap_database(db_filename, cameras, kpts, matches):
    # Set up a COLMAP database with the given intrinsics, keypoints, and matches
    db_dir = os.path.dirname(db_filename)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    if os.path.exists(db_filename):
        os.remove(db_filename)
    os.system(f'colmap database_creator --database_path {db_filename}')
    db = pycolmap.Database.open(db_filename)

    num_images = kpts.shape[0]

    # Add images and cameras
    im_ids = []
    for i in range(num_images):
        cam = pycolmap.Camera(cameras[i])
        cam_id = db.write_camera(cam, False)
        im = pycolmap.Image(name = 'image_' + str(i+1), camera_id = cam_id)
        im_id = db.write_image(im, False)
        db.write_keypoints(im_id, kpts[i])
        im_ids.append(im_id)

    # Add matches
    for (i, j), m in matches.items():
        # Add matches between images i and j
        db.write_matches(im_ids[i], im_ids[j], m)
        tw = pycolmap.TwoViewGeometry()
        tw.inlier_matches=m
        db.write_two_view_geometry(im_ids[i], im_ids[j], tw)

    db.close()


def write_empty_colmap_rec(rec_path, cameras, poses):
    # Write an empty COLMAP reconstruction with the given cameras and poses

    files = ['cameras.txt', 'images.txt', 'points3D.txt',
            'cameras.bin', 'images.bin', 'points3D.bin']

    for f in files:
        if os.path.exists(rec_path + '/' + f):
            os.remove(rec_path + '/' + f)

    with open(rec_path + '/cameras.txt', 'w') as f:
        for k, cam in enumerate(cameras):
            assert len(cam['params']) == 4, "Cam params should be of format: [fx, fy, cx, cy]"
            f.write(f"{k+1} {cam['model']} {cam['width']} {cam['height']} {cam['params'][0]:.16f} {cam['params'][1]:.16f} {cam['params'][2]:.16f} {cam['params'][3]:.16f}\n")

    with open(rec_path + '/images.txt', 'w') as f:
        for k, pose in enumerate(poses):
            q, t = pose
            #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            f.write(f"{k+1} {q[0]:.16f} {q[1]:.16f} {q[2]:.16f} {q[3]:.16f} {t[0]:.16f} {t[1]:.16f} {t[2]:.16f} {k+1} image_{k+1}\n")
            f.write('\n')

    with open(rec_path + '/points3D.txt', 'w') as f:
        f.write('')


def collect_matches(rec, kpts, matches):
    p2d = []
    p3d = []
    for k, m in enumerate(matches):
        im = rec.images[k+1]
        if im.num_points3D == 0:  # is not triangulated (triangulated = 0/0 which gives error)
            continue

        have_match = m >= 0
        ind = np.where(have_match)[0]
        for ind_q in ind:
            ind_r = m[ind_q]

            pt3d_id = im.points2D[ind_r].point3D_id
            if pt3d_id < 0 or pt3d_id == 2**64-1:
                continue

            p2d.append(kpts[ind_q])
            p3d.append(rec.points3D[pt3d_id].xyz)

    return p2d, p3d


def get_reconstruction(tuple_length):
    opts = pycolmap.IncrementalPipelineOptions()
    opts.min_num_matches = 1  # defaults to 15 in colmap
    #if tuple_length == 2:
    opts.triangulation.ignore_two_view_tracks = False  # keep 2-view tracks

    pycolmap.logging.minloglevel     = pycolmap.logging.Level.ERROR
    pycolmap.logging.stderrthreshold = pycolmap.logging.Level.FATAL

    with pycolmap.ostream():              # ← mutes COLMAP’s C++ logging
        rec = pycolmap.triangulate_points(
            reconstruction= pycolmap.Reconstruction('tmp'),
            database_path='tmp/test.db',
            image_path='tmp',
            output_path='tmp',
            options=opts,
    )
    return rec


def transitive_matching(kpts_q, kpts_ref, matches_q2ref, cameras_ref, poses_ref, camera_q, ba):
    """
    Performs 2D-3D matching using COLMAP by triangulating reference points with track length >= 2.

    Args:
        kpts_q: np.ndarray of shape (K, 2) keypoints of the query image (np.float32)
        kpts_ref: np.ndarray of shape (T, K, 2) keypoints of the reference  (np.float32)
        matches_q2ref: np.ndarray of shape (T, K) matches between query and ref images (np.int32)
        cameras_ref: list of camera parameters for reference images (len T of list)
        poses_ref: list of camera poses for reference images (len T of list)
            poses_ref[i] = [q, t] where q is the quaternion and t is the translation vector
        camera_q: camera parameters for the query image
            (model, width, height, params) e.g.
            camera_q = {'model': 'PINHOLE', 'width': 640, 'height': 480,
                        'params': np.array([fx, fy, cx, cy])}
        ba: bool, whether to run bundle adjustment after pose estimation

    Returns:
        pose: estimated pose of the query image (poselib pose object)
    """
    # Collect transitive matches (getting pairwise matches between reference images)
    matches_ref2ref = collect_transitive_matching(kpts_ref, matches_q2ref)

    # Write matches to database
    setup_colmap_database('tmp/test.db', cameras_ref, kpts_ref, matches_ref2ref)

    n_matches = 0
    for key in matches_ref2ref.keys():
        n_matches += len(matches_ref2ref[key])

    if n_matches < 3:  # 3 is the minimal sample for absolute pose estimation
        n_3d_pts = 0
        pose = poselib.CameraPose()
    else:
        # Write empty COLMAP reconstruction
        write_empty_colmap_rec('tmp', cameras_ref, poses_ref)

        # Call COLMAP point triangulator
        tuple_length = kpts_ref.shape[0]
        rec = get_reconstruction(tuple_length)
        n_3d_pts = rec.num_points3D()

        if n_3d_pts < 3:  # Not enough points were triangulated for pose estimation
            pose = poselib.CameraPose()
        else:
            # Collect 2D-3D correspondences between query and triangulated points
            p2d, p3d = collect_matches(rec, kpts_q, matches_q2ref)

            # Estimate pose  # TODO: A little sloppy to include each 3D point twice
            pose, info = poselib.estimate_absolute_pose(p2d, p3d, camera_q, {'max_reproj_error': 16.0})

            if np.isnan(pose.Rt).any():
                pose.Rt = np.zeros((3, 4))
                pose.R = np.eye(3)
            else:
                if ba:
                    result = tmu.run_ba(
                        rec, kpts_q, kpts_ref, matches_q2ref, camera_q, pose, cameras_ref, poses_ref)
                    pose.Rt = result['pose_query']

    return pose, n_3d_pts


def transitive_matching_runner(data, kpts_q, kpts_ref, matches, batch_ind, mode="type1+", ba=False):
    assert mode in ["type1+", "type2+"], "Mode should be either 'type1+' or 'type2+'"
    matches_q2ref, cameras_ref, poses_ref, camera_q = tmu.assemble_transitive_matching_data(
        data, matches, batch_ind)
    # n_3d_pts is of type 2+ (i.e. >= 2 matches)
    pose, n_3d_pts = transitive_matching(
        kpts_q, kpts_ref, matches_q2ref, cameras_ref, poses_ref, camera_q, ba)
    
    if mode == "type1+":
        kpts_ref_with_q = np.concatenate([kpts_q[np.newaxis, ...], kpts_ref], axis=0)
        matches_q2ref_with_q = np.concatenate([np.arange(kpts_q.shape[0])[np.newaxis, ...], matches_q2ref], axis=0)
        cameras_ref_with_q = [camera_q] + cameras_ref
        poses_ref_with_q = [[pose.q, pose.t]] + poses_ref
        # n_3d_pts is of type 1+ (i.e. >= 1 match)
        pose, n_3d_pts = transitive_matching(
            kpts_q, kpts_ref_with_q, matches_q2ref_with_q, cameras_ref_with_q,
            poses_ref_with_q, camera_q, ba
        )
    return pose, n_3d_pts

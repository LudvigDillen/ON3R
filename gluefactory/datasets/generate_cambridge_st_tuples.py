"""
Lot's of content in this file is take from Hloc. Kudos to the ETH Zurich team for their work.
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import cv2
import time
import collections
import struct
import re

from gluefactory.datasets.generate_megadepth1500_st_tuples import plot_tuples
from ..settings import DATA_PATH
from gluefactory.utils.misc import start_debug
from .megadepth_utils import sample_star_topology_tuples
from hloc.utils.read_write_model import read_images_text, read_cameras_text

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# def load_dataset_txt(path):
#     data = {}
#     with open(path, "r") as f:
#         for line in f:
#             line = line.strip()
#             # skip empty lines or header/comment lines
#             if not line or not line.startswith("seq"):
#                 continue

#             parts = line.split()
#             name = parts[0]  # e.g. seq4/frame00133.png
#             try:
#                 nums = list(map(float, parts[1:]))
#             except ValueError:
#                 # skip malformed lines
#                 continue

#             # Parse sequence and frame
#             m = re.match(r"(.*)/(frame\d+)\.png", name)
#             if not m:
#                 continue
#             seq, frame = m.groups()

#             data.setdefault(seq, {})[frame] = nums
#     return data


# def merge_datasets(*dicts):
#     merged = {}
#     for d in dicts:
#         for seq, frames in d.items():
#             merged.setdefault(seq, {}).update(frames)
#     return merged


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def load_color_img(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


def to_str(x):
    """bytes → str, anything else → str untouched"""
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def canonical_key(scene, query_str, refs_seq):
    """Return a hashable key with refs in sorted order"""
    return (scene, query_str, tuple(sorted(refs_seq)))


def generate_data(args):
    out_folder = DATA_PATH / Path(args.out_folder)
    img_folder = DATA_PATH / Path(args.img_folder)
    existing_combinations = set()
    t1 = time.time()  # Start the timer
    t2 = t1  # Initialize t2 to t1
    with open(out_folder / f"test_tuples/tuples_combinations_{args.text_file_id}.txt", "w") as tuple_combinations, \
         open(out_folder / f"test_tuples/tuples_{args.text_file_id}.txt", "w") as tuple_file, \
         h5py.File(out_folder / "cambridge_overlap.h5", "r") as f:
        k_queries = args.num_tuples
        queries_sampled = 0
        while queries_sampled < k_queries and (t2 - t1) < 360:
            print(f"Sampling tuples: {queries_sampled}/{k_queries} after {t2 - t1:.2f} seconds")
            for i, (scene_name, scene_grp) in enumerate(f.items()):
                # img_pth_metric = out_folder / f"CambridgeLandmarks_Colmap_Retriangulated_1024px/all/{scene_name}/model_train/images.bin"
                # images_metric = read_images_binary(img_pth_metric)
                img_pth_metric = out_folder / f"CambridgeLandmarks_Colmap_Retriangulated_1024px/all/{scene_name}/empty_all/images.txt"
                images_metric = read_images_text(img_pth_metric)
                cam_pth_metric = out_folder / f"CambridgeLandmarks_Colmap_Retriangulated_1024px/all/{scene_name}/empty_all/cameras.txt"
                cameras_metric = read_cameras_text(cam_pth_metric)
                images_metric_by_name = {img.name: img for img in images_metric.values()}
                cam_metric_by_name = {img.name: cam for img, cam in zip(images_metric.values(), cameras_metric.values())}
                gt_keys = images_metric_by_name.keys()
                # --- Load and merge ---
                # dataset_path = img_folder / scene_name
                # train_data = load_dataset_txt(dataset_path/"dataset_train.txt")
                # test_data  = load_dataset_txt(dataset_path/"dataset_test.txt")
                # data = merge_datasets(train_data, test_data)
                

                for seq in scene_grp.keys():
                    if queries_sampled >= k_queries:
                        break
                    grp = scene_grp[seq]
                    overlap_matrix = grp["matrix"][:]
                    frames = grp["frames"][:]

                    queries, k_n_ref_tuples, sampling_success = sample_star_topology_tuples(
                        overlap_matrix, 1, args.tuple_length)
                    
                    if args.plot:
                        n_queries = len(queries)
                        img_subfolder = Path(img_folder / scene_name / seq)
                        for i in range(n_queries):
                            if i > 5:
                                break
                            q_ind = queries[i]
                            ref_inds = k_n_ref_tuples[i]
                            q_img = load_color_img(img_subfolder / frames[q_ind].decode())
                            ref_imgs = [load_color_img(img_subfolder / frames[ref_ind].decode())
                                        for ref_ind in ref_inds]
                            plot_tuples(q_ind, ref_inds, overlap_matrix, args.tuple_length, q_img, ref_imgs)
    
                    if sampling_success:
                        assert len(queries) == 1, "Only one query is expected in star topology tuples."
                        query_str = to_str(frames[queries[0]])
                        refs_list = [to_str(frames[idx]) for idx in k_n_ref_tuples[0]]
                        seq_query_str = f"{seq}/{query_str}"
                        seq_frame_str = [f"{seq}/{ref_str}" for ref_str in refs_list]
                        if ((seq_query_str not in gt_keys) or any([(ref_str not in gt_keys) for ref_str in seq_frame_str])):
                            continue  # skip if any image is not in the metric model
                        key = canonical_key(scene_name, query_str, refs_list)

                        if key in existing_combinations:
                            continue                                # skip duplicate combo
                        existing_combinations.add(key)

                        # pretty line — bytes removed, refs sorted for stability
                        pretty_line = (
                            f"scene: {scene_name}; "
                            f"query: {query_str}; "
                            f"refs: [{', '.join(sorted(refs_list))}]\n"
                        )
                        tuple_combinations.write(pretty_line)

                        queries_sampled += len(queries)
                        # Save the queries and tuples to text file
                        frames_q = frames[queries]
                        frames_refs = frames[k_n_ref_tuples]
                        frames_q_and_refs = np.concatenate((frames_q[:, None], frames_refs), axis=1)

                        rows, cols = frames_q_and_refs.shape
                        scene_seq_dir = out_folder / scene_name / seq
                        for j in range(rows):
                            # Image names
                            image_names = ""
                            intrinsic_str = ""
                            pose_str = ""

                            for k in range(cols):
                                frame_str = frames_q_and_refs[j, k].decode()
                                frame_key = int(re.search(r'\d+', frame_str).group())
                                image_names += f"{scene_name}/{seq}/{frame_str} "

                                # Intrinsics (K)
                                K_pth = scene_seq_dir / "sfm/cameras.bin"
                                K_all = read_cameras_binary(K_pth)
                                cam = K_all[frame_key]
                                #focal_length, cx, cy, k1 = cam.params
                                cam_met = cam_metric_by_name[f"{seq}/{frame_str}"]
                                focal_length_met, cx_met, cy_met, k1_met = cam_met.params

                                width_factor = cam.width / cam_met.width
                                height_factor = cam.height / cam_met.height
                                fx_new = focal_length_met * width_factor
                                fy_new = focal_length_met * height_factor
                                cx_new = cx_met * width_factor
                                cy_new = cy_met * height_factor
                                assert fx_new == fy_new, "fx_new != fy_new"
                                K_metric = np.array([[fx_new, 0.0, cx_new],
                                                    [0.0, fy_new,  cy_new],
                                                    [0.0, 0.0, 1.0]])
                                intrinsic_str += " ".join(map(str, K_metric.flatten())) + " "

                                # Absolute poses (w2cam) (non-metric)
                                # img_pth = scene_seq_dir / "sfm/images.bin"
                                # images = read_images_binary(img_pth)
                                # image = images[frame_key]
                                # R = qvec2rotmat(image.qvec)
                                # t = image.tvec
                                # R_list.append(R)
                                # t_list.append(t)
                                # center_list.append(- R.T @ t)

                                # Absolute poses (w2cam) (metric) (Torsten)
                                name_key = f"{seq}/{frame_str}"
                                image = images_metric_by_name[name_key]
                                R = qvec2rotmat(image.qvec)
                                t = image.tvec
                                # R_metric_list.append(R)
                                # t_metric_list.append(t)
                                # center_metric_list.append(- R.T @ t)

                                # GT with datasset
                                # frame = frame_str.split(".")[0]
                                # pose_list = data[seq][frame]
                                # t = np.array(pose_list[0:3]).reshape(3,1)
                                # q = np.array(pose_list[3:]).reshape(4,1)
                                # R = qvec2rotmat(q.flatten())
                                # R_metric_list.append(R)
                                # t_metric_list.append(t)
                                # center_metric_list.append(- R.T @ t)

                                pose_str += " ".join(map(str, R.flatten())) + " "
                                pose_str += " ".join(map(str, t.flatten())) + " "
                        
                            # n_samples = len(center_list)
                            # for i in range(n_samples):
                            #     for j in range(n_samples):
                            #         if i < j:
                            #             d1 = np.linalg.norm(center_list[i] - center_list[j]).item()
                            #             d2 = np.linalg.norm(center_metric_list[i] - center_metric_list[j]).item()
                            #             print(f"{i}-{j}: (Non: {d1:.2f}, met: {d2:.2f}, scale: {d2/d1:.2f}")

                            # Write the row to the file
                            str_row = (image_names + intrinsic_str + pose_str).strip()
                            tuple_file.write(str_row + "\n")
            t2 = time.time()
    print(f"Sampled {queries_sampled} tuples in {t2 - t1:.2f} seconds.")
    success_str = "successful" if queries_sampled >= k_queries else "unsuccessful"
    print(f"Thus, it was {success_str} to sample {k_queries} tuples.")

if __name__ == "__main__":
    """ Use one of the following to run the script:
    python -m gluefactory.datasets.generate_cambridge_st_tuples -tf_id star_topology_2tuples -d
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_tuples", type=int, default=300)
    parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
    parser.add_argument("-tf_id", "--text_file_id", type=str, default="")
    parser.add_argument("-t", "--tuple_length", type=int, default=2)
    parser.add_argument("-if", "--img_folder", type=str, default="cambridge")
    parser.add_argument("-of", "--out_folder", type=str, default="outputs/cambridge")
    parser.add_argument("-p", "--plot", action='store_true', help="Plot the tuples")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    if args.debug:
        start_debug()
    generate_data(args)

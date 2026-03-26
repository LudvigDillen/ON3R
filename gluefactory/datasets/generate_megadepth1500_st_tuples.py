# NOTE: This file containts quite much unnecessary code for the task at hand. The reason for that
#       is that the code is partially copied from megadepth.py and when my goal with this task was
#       completed (i.e. creating the necessary files for MegaDepth1500 tuples (pairs_tuples.txt
#       and pairs_calibrated_tuples.txt)) I didn't want to spend time on cleaning up the code.
#       Performance is not and issue here, it takes like a second to generate the necessary files.
#       (Ludvig Dillén, 2024-07-27)
import argparse
import logging
import shutil
import tarfile
from collections.abc import Iterable
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from omegaconf import OmegaConf

from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_heatmaps, plot_image_grid
from .base_dataset import BaseDataset
from .utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics
from .megadepth import count_sample_info, init_counts, pth_to_img
from gluefactory.utils.misc import start_debug
from .megadepth_utils import sample_star_topology_tuples
#from .megadepth import _TupleDataset


logger = logging.getLogger(__name__)
scene_lists_path = Path(__file__).parent / "megadepth_scene_lists"


class MegaDepth(BaseDataset):
    default_conf = {
        # paths
        "data_dir": "megadepth/",
        "depth_subpath": "depth_undistorted/",
        "image_subpath": "Undistorted_SfM/",
        "info_dir": "scene_info/",  # @TODO: intrinsics problem?
        # Training
        "train_split": "train_scenes_clean.txt",
        "train_num_per_scene": 500,
        # Validation
        "val_split": "valid_scenes_clean.txt",
        "val_num_per_scene": 1500,
        "val_pairs": None,
        "val_samples": None,
        # Test
        "test_split": "test_scenes_clean.txt",
        "test_num_per_scene": None,
        "test_pairs": None,
        # data sampling
        "views": 2,
        "min_overlap": 0.3,  # only with D2-Net format
        "max_overlap": 1.0,  # only with D2-Net format
        "num_overlap_bins": 1,
        "sort_by_overlap": False,
        "triplet_enforce_overlap": False,  # only with views==3
        # image options
        "read_depth": True,
        "read_image": True,
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0,
        # features from cache
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
        "tuples": {
            "tuple_length": 5,
        },
        "text_file_id": "",
        "plot": False,  # Plot the tuples
    }

    def _init(self, conf):
        if not (DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading the MegaDepth dataset.")
            self.download()

    def download(self):
        data_dir = DATA_PATH / self.conf.data_dir
        tmp_dir = data_dir.parent / "megadepth_tmp"
        if tmp_dir.exists():  # The previous download failed.
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        url_base = "https://cvg-data.inf.ethz.ch/megadepth/"
        for tar_name, out_name in (
            ("Undistorted_SfM.tar.gz", self.conf.image_subpath),
            ("depth_undistorted.tar.gz", self.conf.depth_subpath),
            ("scene_info.tar.gz", self.conf.info_dir),
        ):
            tar_path = tmp_dir / tar_name
            torch.hub.download_url_to_file(url_base + tar_name, tar_path)
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=tmp_dir)
            tar_path.unlink()
            shutil.move(tmp_dir / tar_name.split(".")[0], tmp_dir / out_name)
        shutil.move(tmp_dir, data_dir)

    def get_dataset(self, split, sample_tuples=True):
        assert self.conf.views in [1, 2, 3]
        return _TupleDataset(self.conf, split)


class _TupleDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        split_conf = conf[split + "_split"]
        if isinstance(split_conf, (str, Path)):
            scenes_path = scene_lists_path / split_conf
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            scenes = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        scenes = sorted(set(scenes))

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.images = {}
        self.depths = {}
        self.poses = {}
        self.intrinsics = {}
        self.valid = {}
        self.tuple_length = conf.tuples.tuple_length
        self.ref_keys = ["query_to_ref_" + str(ref_idx) for ref_idx in range(self.tuple_length)]

        # load metadata
        self.info_dir = self.root / self.conf.info_dir
        self.scenes = []
        for scene in scenes:
            path = self.info_dir / (scene + ".npz")
            try:
                info = np.load(str(path), allow_pickle=True)
            except Exception:
                logger.warning(
                    "Cannot load scene info for scene %s at %s.", scene, path
                )
                continue
            self.images[scene] = info["image_paths"]
            self.depths[scene] = info["depth_paths"]
            self.poses[scene] = info["poses"]
            self.intrinsics[scene] = info["intrinsics"]
            self.scenes.append(scene)

        self.write_to_megadepth1500tuples_files()

    def _weighted_n_tuple_sampling(self, num_pos, mutual_overlap_mat):
        """
        Sample database tuples to match query against.

        Args
            num_pos: number of positive tuples to sample
            mutual_overlap_mat: (N, N) matrix with the mutual overlap between images
            seed: random seed

        Returns
            an array of query K query images
            an array of (K x tuple_length) reference images
        """
        k_queries = num_pos // len(self.scenes)
        queries, k_n_ref_tuples, sampling_success = sample_star_topology_tuples(
            mutual_overlap_mat, k_queries, self.tuple_length)
        equals = (k_n_ref_tuples[..., None] == k_n_ref_tuples[:, None])  # (N, T, T)
        diag_inds = np.arange(self.tuple_length)
        equals[:, diag_inds, diag_inds] = False
        assert not np.any(equals), "Duplicate elements in k_n_ref_tuples."

        return queries, k_n_ref_tuples

    def _gather_info(self, scene, counts):
        path = self.info_dir / (scene + ".npz")
        assert path.exists(), path
        info = np.load(str(path), allow_pickle=True)
        valid = (self.images[scene] != None) & (  # noqa: E711
            self.depths[scene] != None  # noqa: E711
        )
        self.valid[scene] = valid
        # ind = np.where(valid)[0]
        # mat is a matrix that tells us about the overlap between the images in a scene
        mat = info["overlap_matrix"][valid][:, valid]
        if True:
            counts = count_sample_info(mat, counts, scene, verbose=False)
        return mat, counts

    def _sample_tuples(self, mat, scene, num_pos):
        mutual_mat = np.sqrt(mat*mat.T)  # (mat+mat.T)/2 is another alternative

        k_queries, k_n_ref_tuples = self._weighted_n_tuple_sampling(num_pos, mutual_mat)

        # Ludde plot some MegaDepth pairs
        if self.conf.plot:
            n_queries = len(k_queries)
            for i in range(n_queries):
                if i > 5:
                    break
                q_ind, ref_inds = k_queries[i], k_n_ref_tuples[i]
                query_img = self._get_img(scene, q_ind)
                ref_imgs = [self._get_img(scene, ref_ind) for ref_ind in ref_inds]
                plot_tuples(q_ind, ref_inds, mutual_mat, self.tuple_length, query_img, ref_imgs)

        return k_queries, k_n_ref_tuples

    def write_to_megadepth1500tuples_files(self):
        """
        The output files contains on tuple on each line.
        Each tuple consists of a query image and T reference images.
        For each line we also write the intrinsics and extrinsics (poses) of the images.
        The specific format is on each line is:
        scene/query_image_name ref_image_name_1 ref_image_name_2 ... ref_image_name_T (T+1 strings in total)
        K_query K_ref_1 K_ref_2 ... K_ref_T (9 numbers for each K => 9(T+1) numbers in total)
        T_query T_ref_1 T_ref_2 ... T_ref_T (12 numbers for each T => 12(T+1) numbers in total)
        So, line.split(" ") returns a list with T+1 + 9(T+1) + 12(T+1) = 22(T+1) elements.
        """
        counts = init_counts()
        self.tuple_items = []
        split = self.split

        assert self.conf.views == 2
        print(f"Split: {split}, scenes: {self.scenes}")
        query_images_scenes = {}
        ref_images_scenes = {}
        query_poses = {}
        query_intrinsics = {}
        ref_poses = {}
        ref_intrinsics = {}

        for scene in self.scenes:
            mat, counts = self._gather_info(scene, counts)
            # image names
            k_queries, k_n_ref_tuples = self._sample_tuples(mat, scene, self.conf.val_num_per_scene)

            valid_image_names = [name.split("/")[-1] for name in self.images[scene][self.valid[scene]]]
            query_images_scenes[scene] = np.array(valid_image_names)[k_queries]
            ref_images_scenes[scene] = np.array(valid_image_names)[k_n_ref_tuples]

            query_poses[scene] = self.poses[scene][self.valid[scene]][k_queries]
            ref_poses[scene] = self.poses[scene][self.valid[scene]][k_n_ref_tuples]

            query_intrinsics[scene] = self.intrinsics[scene][self.valid[scene]][k_queries]
            ref_intrinsics[scene] = self.intrinsics[scene][self.valid[scene]][k_n_ref_tuples]

        # Return the tuple of data
        id = self.conf.text_file_id
        string_end = ".txt" if id == "" else "_" + id + ".txt"
        tuples_file = DATA_PATH / Path("megadepth1500/tuples" + string_end)
        tuples_calibrated_file = DATA_PATH / Path("megadepth1500/calibrated_tuples" + string_end)
        if tuples_file.exists():
            logger.warning(
                "The file %s already exists. It will be overwritten.", tuples_file
            )
            tuples_file.unlink()

        with open(tuples_file, "w") as file_t, open(tuples_calibrated_file, "w") as file_t_cal:
            for scene in self.scenes:
                for j, (query, refs) in enumerate(zip(query_images_scenes[scene], ref_images_scenes[scene])):
                    # Write scene and query image name
                    file_t.write(f"{scene}/{query}")
                    file_t_cal.write(f"{scene}/{query}")
                    # Write scene and ref image names
                    for ref in refs:
                        file_t.write(f" {scene}/{ref}")
                        file_t_cal.write(f" {scene}/{ref}")

                    # Write intrinsics query
                    K = query_intrinsics[scene][j]
                    file_t_cal.write(f" {K[0, 0]} {K[0, 1]} {K[0, 2]}"
                                     f" {K[1, 0]} {K[1, 1]} {K[1, 2]}"
                                     f" {K[2, 0]} {K[2, 1]} {K[2, 2]}")
                    # Write intrinsics ref
                    for i in range(self.tuple_length):
                        K = ref_intrinsics[scene][j][i]
                        file_t_cal.write(f" {K[0, 0]} {K[0, 1]} {K[0, 2]}"
                                         f" {K[1, 0]} {K[1, 1]} {K[1, 2]}"
                                         f" {K[2, 0]} {K[2, 1]} {K[2, 2]}")

                    # Write extrinsics (absolute poses)
                    T_query = query_poses[scene][j]
                    file_t_cal.write(f" {T_query[0, 0]} {T_query[0, 1]} {T_query[0, 2]}"
                                     f" {T_query[1, 0]} {T_query[1, 1]} {T_query[1, 2]}"
                                     f" {T_query[2, 0]} {T_query[2, 1]} {T_query[2, 2]}"
                                     f" {T_query[0, 3]} {T_query[1, 3]} {T_query[2, 3]}")

                    for i in range(self.tuple_length):
                        T_ref = ref_poses[scene][j][i]
                        file_t_cal.write(f" {T_ref[0, 0]} {T_ref[0, 1]} {T_ref[0, 2]}"
                                         f" {T_ref[1, 0]} {T_ref[1, 1]} {T_ref[1, 2]}"
                                         f" {T_ref[2, 0]} {T_ref[2, 1]} {T_ref[2, 2]}"
                                         f" {T_ref[0, 3]} {T_ref[1, 3]} {T_ref[2, 3]}")

                    file_t.write("\n")
                    file_t_cal.write("\n")

    def _read_view(self, scene, idx):
        path = self.root / self.images[scene][idx]

        # read pose data
        K = self.intrinsics[scene][idx].astype(np.float32, copy=False)
        T = self.poses[scene][idx].astype(np.float32, copy=False)

        # read image
        if self.conf.read_image:
            img = load_image(self.root / self.images[scene][idx], self.conf.grayscale)
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros(
                [3 - 2 * int(self.conf.grayscale), size[0], size[1]]
            ).float()

        # read depth
        if self.conf.read_depth:
            depth_path = (
                self.root / self.conf.depth_subpath / scene / (path.stem + ".h5")
            )
            with h5py.File(str(depth_path), "r") as f:
                depth = f["/depth"].__array__().astype(np.float32, copy=False)
                depth = torch.Tensor(depth)[None]
            assert depth.shape[-2:] == img.shape[-2:]
        else:
            depth = None

        # add random rotations
        do_rotate = self.conf.p_rotate > 0.0 and self.split == "train"
        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = np.rot90(img, k=-k, axes=(-2, -1))
                if self.conf.read_depth:
                    depth = np.rot90(depth, k=-k, axes=(-2, -1)).copy()
                K = rotate_intrinsics(K, img.shape, k + 2)
                T = rotate_pose_inplane(T, k + 2)

        name = path.name

        data = self.preprocessor(img)
        if depth is not None:
            data["depth"] = self.preprocessor(depth, interpolation="nearest")["image"][
                0
            ]
        K = scale_intrinsics(K, data["scales"])

        data = {
            "name": name,
            "scene": scene,
            "T_w2cam": Pose.from_4x4mat(T),
            "depth": depth,
            "camera": Camera.from_calibration_matrix(K).float(),
            **data,
        }

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            if do_rotate and k != 0:
                # ang = np.deg2rad(k * 90.)
                kpts = features["keypoints"].copy()
                x, y = kpts[:, 0].copy(), kpts[:, 1].copy()
                w, h = data["image_size"]
                if k == 1:
                    kpts[:, 0] = w - y
                    kpts[:, 1] = x
                elif k == -1:
                    kpts[:, 0] = y
                    kpts[:, 1] = h - x

                else:
                    raise ValueError
                features["keypoints"] = kpts

            data = {"cache": features, **data}
        return data

    def _getitem_tuple(self, idx):
        data = {}
        for ref_idx, ref_key in enumerate(self.ref_keys):
            data[ref_key] = self.getitem(idx, ref_idx)
        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                data = self._getitem_tuple(idx)
        else:
            data = self._getitem_tuple(idx)
        return data

    def getitem(self, idx, ref_idx):
        """
        ref_idx should be 0 to self.tuple_length-1
        """
        if self.conf.views == 2:
            if isinstance(idx, list):
                sys.exit("Don't use list as input to getitem_tuple" + str(idx))
            else:
                scene, idx0, idx1, overlap = self.tuple_items[idx][ref_idx]
            data0 = self._read_view(scene, idx0)
            data1 = self._read_view(scene, idx1)
            data = {
                "view0": data0,
                "view1": data1,
            }
            data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
            data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
            data["overlap_0to1"] = overlap
            data["name"] = f"{scene}/{data0['name']}_{data1['name']}"
        else:
            sys.exit("Don't use getitem_tuple when views != 2")
        data["scene"] = scene
        data["idx"] = idx
        return data

    def __len__(self):
        return len(self.tuple_items)

    def _get_img(self, scene, ind):
        img_pth = self.images[scene][self.valid[scene]][ind]
        return pth_to_img(img_pth)


def plot_tuples(q_ind, ref_inds, mutual_overlap_mat, tuple_length, query_img, ref_imgs):
    plt.close('all')

    plt.figure()
    plt.subplot(231)
    plt.imshow(query_img)
    plt.title("Query image")

    ref_ref_cov = mutual_overlap_mat[np.ix_(ref_inds, ref_inds)]

    for j, ref_ind in enumerate(ref_inds):
        ref_img = ref_imgs[j]
        plt.subplot(2, 3, 2+j)
        plt.imshow(ref_img)
        s = f"Mutual Overlap Score (Q) {mutual_overlap_mat[q_ind, ref_ind]:.2f} (db = ["
        for k in range(tuple_length):
            s += f" {ref_ref_cov[j, k]:.2f}"
            if k < tuple_length - 1:
                s += ", "
            else:
                s += "])"
        plt.title(s)
    plt.suptitle("Query and N-tuple")
    plt.show()


def main(args):
    conf = {
        "num_overlap_bins": 3,
        "sort_by_overlap": False,
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": args.num_items,  # we have two sccenes
        "text_file_id": args.text_file_id,
        "tuples": {
            "tuple_length": args.tuple_length,
        },
        "plot": args.plot,
    }

    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = MegaDepth(conf)
    loader = dataset.get_data_loader(args.split)

if __name__ == "__main__":
    """ Use one of the following to run the script:
    python -m gluefactory.datasets.generate_megadepth1500_st_tuples -tf_id star_topology_2tuples -d
    """
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_items", type=int, default=1500)
    parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
    parser.add_argument("-tf_id", "--text_file_id", type=str, default="")
    parser.add_argument("-t", "--tuple_length", type=int, default=2)
    parser.add_argument("-p", "--plot", action='store_true', help="Plot the tuples")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    if args.debug:
        start_debug()
    main(args)

# NOTE: This file containts quite much unnecessary code for the task at hand. The reason for that
#       is that the code is partially copied from megadepth.py and when my goal with this task was
#       completed (i.e. creating the necessary files for MegaDepth1500 tuples (pairs_tuples.txt
#       and pairs_calibrated_tuples.txt)) I didn't want to spend time on cleaning up the code.
#       Performance is not and issue here, it takes like a second to generate the necessary files.
#       (Ludvig Dillén, 2024-07-27)


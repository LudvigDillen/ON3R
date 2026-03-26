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
from .megadepth_utils import sample_n_difficult_tuples


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
            "difficulty": "easy",
            "n_tuples": 1500,
        },
        "text_file_id": "",
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

        self.write_to_megadepth1500tuples_files(conf.seed)

    def _read_megadepth1500_file(self):
        # Read the content of the file into a Nx2 array
        file = str(DATA_PATH / Path("megadepth1500/pairs.txt"))
        with open(file, 'r') as f:
            content = f.readlines()
        content = np.array([line.strip().split() for line in content])
        scenes_order = [s[0].split("/")[0] for s in content]

        scenes_dict = {}
        for scene in self.scenes:
            scenes_dict[scene] = np.array([[item[0].split("/")[-1], item[1].split("/")[-1]] \
                                           for item in content if item[0].startswith(scene)])
        return scenes_dict, scenes_order

    def _weighted_n_tuple_sampling(self, mutual_overlap_mat, seed, megadepth1500_image_ids):
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
        tuple_difficulty = self.conf.tuples.difficulty
        k_queries = megadepth1500_image_ids.shape[0]
        queries = megadepth1500_image_ids[:, 0]
        k_n_ref_tuples = np.empty((k_queries, self.tuple_length), dtype=np.int_)
        k_n_ref_tuples[:, 0] = megadepth1500_image_ids[:, 1]

        # Sample a subset of pairs, binned by overlap.
        covisible_neighbor_mask = (mutual_overlap_mat >= self.conf.min_overlap) & \
                                  (mutual_overlap_mat <= self.conf.max_overlap)
        low_overlap_tuples = []

        for i, query_ind in enumerate(queries):
            score_to_query = mutual_overlap_mat[query_ind]
            reference_scores = score_to_query[covisible_neighbor_mask[query_ind]]
            reference_inds = np.argwhere(covisible_neighbor_mask[query_ind]).squeeze(1)
            if k_n_ref_tuples[i, 0] in reference_inds:  # If we get same ref again
                ind_to_remove = np.where(reference_inds == k_n_ref_tuples[i, 0])
                reference_inds = np.delete(reference_inds, ind_to_remove)
                reference_scores = np.delete(reference_scores, ind_to_remove)
            reference_scores_normalized = reference_scores / reference_scores.sum()
            # NOTE: We add a few extra images below the threshold 0.1 if there aren't enough
            #       samples within the range 0.1-0.7.
            if len(reference_inds) < self.tuple_length - 1:  # if there are few db imgs for query
                candidates = mutual_overlap_mat[query_ind]
                # We do not want to add to easy examples nor the query itself
                candidates[candidates > self.conf.max_overlap] = 0
                top_k_indices = np.argsort(candidates)[::-1][:self.tuple_length]
                # Remove already added ref image from the top_k_indices
                top_k_indices_without_already_added_image = \
                    top_k_indices[top_k_indices != k_n_ref_tuples[i, 0]][:self.tuple_length-1]
                k_n_ref_tuples[i, 1:] = top_k_indices_without_already_added_image
                low_overlap_tuples.append(i)
            elif tuple_difficulty == "easy":
                # there are n_neighbors_query[query_ind] neighbors
                k_n_ref_tuples[i, 1:] = np.random.RandomState(seed).choice(
                    reference_inds, size=(self.tuple_length-1), replace=False,
                    p=reference_scores_normalized)
            else:
                k_n_ref_tuples[i, 1:] = sample_n_difficult_tuples(
                    self.tuple_length-1, reference_inds, reference_scores_normalized,
                    mutual_overlap_mat, seed, score_to_query, covisible_neighbor_mask, query_ind, i,
                    tuple_difficulty)

        # Checks for duplicate elements in each row of k_n_ref_tuples
        equals = (k_n_ref_tuples[..., None] == k_n_ref_tuples[:, None])  # (N, T, T)
        diag_inds = np.arange(self.tuple_length)
        equals[:, diag_inds, diag_inds] = False
        assert not np.any(equals), "Duplicate elements in k_n_ref_tuples."

        # Checks that none of the elements in k_n_ref_tuples are found in the corresponding row of
        # queries
        for i, query in enumerate(queries):
            assert not np.any(np.isin(k_n_ref_tuples[i], query)), "Query found in reference tuples."

        print("Number of tuples with at least one pair with overlap below thresh:",
              len(low_overlap_tuples))
        return queries, k_n_ref_tuples, low_overlap_tuples

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

    def _sample_tuples(self, mat, seed, scene, megadepth1500_scenes, valid_image_names):
        mutual_mat = np.sqrt(mat*mat.T)  # (mat+mat.T)/2 is another alternative

        megadepth1500_image_ids = np.zeros(megadepth1500_scenes[scene].shape, dtype=np.int_)
        for i, item in enumerate(megadepth1500_scenes[scene]):
            # Let the query be the first image in the pair and the second image be the reference
            megadepth1500_image_ids[i] = [valid_image_names.index(item[0]),
                                          valid_image_names.index(item[1])]
        k_queries, k_n_ref_tuples, low_overlap_tuples = self._weighted_n_tuple_sampling(
            mutual_mat, seed, megadepth1500_image_ids)

        # Ludde plot some MegaDepth pairs
        if False:
            for i in low_overlap_tuples:
                self._plot_tuples(scene, k_queries, k_n_ref_tuples, mutual_mat, i)

        return k_queries, k_n_ref_tuples

    def write_to_megadepth1500tuples_files(self, seed):
        logger.info("Sampling new %s data with seed %d.", self.split, seed)
        counts = init_counts()
        self.tuple_items = []
        split = self.split

        assert self.conf.views == 2
        print(f"Split: {split}, scenes: {self.scenes}")
        megadepth1500_scenes, scenes_order = self._read_megadepth1500_file()
        query_images_scenes = {}
        ref_images_scenes = {}
        query_poses = {}
        query_intrinsics = {}
        ref_poses = {}
        ref_intrinsics = {}

        for scene in self.scenes:
            mat, counts = self._gather_info(scene, counts)
            # image names
            valid_image_names = [name.split("/")[-1] for name in self.images[scene][self.valid[scene]]]
            k_queries, k_n_ref_tuples = self._sample_tuples(mat, seed, scene, megadepth1500_scenes,
                                                            valid_image_names)

            query_images_scenes[scene] = np.array(valid_image_names)[k_queries]
            ref_images_scenes[scene] = np.array(valid_image_names)[k_n_ref_tuples]

            query_poses[scene] = self.poses[scene][self.valid[scene]][k_queries]
            ref_poses[scene] = self.poses[scene][self.valid[scene]][k_n_ref_tuples]

            query_intrinsics[scene] = self.intrinsics[scene][self.valid[scene]][k_queries]
            ref_intrinsics[scene] = self.intrinsics[scene][self.valid[scene]][k_n_ref_tuples]

        # Return the tuple of data
        counter = {scene: 0 for scene in self.scenes}
        id = self.conf.text_file_id
        string_end = ".txt" if id == "" else "_" + id + ".txt"
        tuples_file = DATA_PATH / Path("megadepth1500/tuples" + string_end)
        tuples_calibrated_file = DATA_PATH / Path("megadepth1500/calibrated_tuples" + string_end)
        scenes_order_subsampled = scenes_order[:self.conf.tuples.n_tuples]

        with open(tuples_file, "w") as file_t, open(tuples_calibrated_file, "w") as file_t_cal:
            for scene in scenes_order_subsampled:
                query = query_images_scenes[scene][counter[scene]]
                refs = ref_images_scenes[scene][counter[scene]]

                # Write scene and query image name
                file_t.write(f"{scene}/{query}")
                file_t_cal.write(f"{scene}/{query}")
                # Write scene and ref image names
                for ref in refs:
                    file_t.write(f" {scene}/{ref}")
                    file_t_cal.write(f" {scene}/{ref}")

                # Write intrinsics query
                K = query_intrinsics[scene][counter[scene]]
                file_t_cal.write(f" {K[0, 0]} {K[0, 1]} {K[0, 2]}"
                                 f" {K[1, 0]} {K[1, 1]} {K[1, 2]}"
                                 f" {K[2, 0]} {K[2, 1]} {K[2, 2]}")
                # Write intrinsics ref
                for i in range(self.tuple_length):
                    K = ref_intrinsics[scene][counter[scene]][i]
                    file_t_cal.write(f" {K[0, 0]} {K[0, 1]} {K[0, 2]}"
                                     f" {K[1, 0]} {K[1, 1]} {K[1, 2]}"
                                     f" {K[2, 0]} {K[2, 1]} {K[2, 2]}")

                # Write extrinsics (relative poses)
                for i in range(self.tuple_length):
                    T_query = query_poses[scene][counter[scene]]
                    T_ref = ref_poses[scene][counter[scene]][i]
                    T_rel = T_ref @ np.linalg.inv(T_query)
                    file_t_cal.write(f" {T_rel[0, 0]} {T_rel[0, 1]} {T_rel[0, 2]}"
                                     f" {T_rel[1, 0]} {T_rel[1, 1]} {T_rel[1, 2]}"
                                     f" {T_rel[2, 0]} {T_rel[2, 1]} {T_rel[2, 2]}"
                                     f" {T_rel[0, 3]} {T_rel[1, 3]} {T_rel[2, 3]}")

                file_t.write("\n")
                file_t_cal.write("\n")
                counter[scene] += 1

    def _get_img(self, scene, ind):
        img_pth = self.images[scene][self.valid[scene]][ind]
        return pth_to_img(img_pth)

    def _plot_tuples(self, scene, k_queries, k_n_ref_tuples, mutual_overlap_mat, i):
        q = k_queries[i]
        query_img = self._get_img(scene, q)
        plt.close('all')

        plt.figure()
        plt.subplot(231)
        plt.imshow(query_img)
        plt.title("Query image")

        for j, ref_ind in enumerate(k_n_ref_tuples[i]):
            ref_img = self._get_img(scene, ref_ind)
            plt.subplot(2, 3, 2+j)
            plt.imshow(ref_img)
            plt.title(f"Mutual Overlap Score {mutual_overlap_mat[q, ref_ind]:.2f}")
        plt.suptitle("Query and N-tuple")
        plt.show()

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


def main(args):
    conf = {
        "min_overlap": args.min_overlap,
        "max_overlap": args.max_overlap,
        "num_overlap_bins": 3,
        "sort_by_overlap": False,
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
        "text_file_id": args.text_file_id,
        "tuples": {
            "tuple_length": args.tuple_length,
            "difficulty": args.difficult,
            "n_tuples": args.num_items,
        },
    }

    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = MegaDepth(conf)
    loader = dataset.get_data_loader(args.split)

if __name__ == "__main__":
    """ Use one of the following to run the script:
    python -m gluefactory.datasets.generate_megadepth1500tuples

    python -m gluefactory.datasets.generate_megadepth1500tuples \
    -tf_id very_hard_05_debug -di easy -mo 0.05 -d
    """
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num_items", type=int, default=1500)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
    parser.add_argument("-tf_id", "--text_file_id", type=str, default="")
    parser.add_argument("-t", "--tuple_length", type=int, default=5)
    parser.add_argument("-di", "--difficult", type=str, default="easy")
    parser.add_argument("-mo", "--min_overlap", type=float, default=0.1)
    parser.add_argument("-xo", "--max_overlap", type=float, default=0.7)
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

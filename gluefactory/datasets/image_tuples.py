"""
Simply load images from a folder or nested folders (does not have any split).
"""

from pathlib import Path
import h5py

import numpy as np
import torch

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from .base_dataset import BaseDataset
from .image_pairs import names_to_pair, parse_homography, parse_camera, parse_relative_pose


class TuplePairs(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "pairs": "???",  # ToDo: add image folder interface
        "root": "???",
        "preprocessing": ImagePreprocessor.default_conf,
        "extra_data": None,  # relative_pose, homography
    }

    def _init(self, conf):
        pair_f = (
            Path(conf.pairs) if Path(conf.pairs).exists() else DATA_PATH / conf.pairs
        )
        with open(str(pair_f), "r") as f:
            self.items = [line.rstrip() for line in f]

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        # each image (tuple_length + 1) has a name which gives tuple_length + 2 parts in total
        self.tuple_length = conf.tuple_length
        self.ref_keys = ["query_to_ref_" + str(ref_idx) for ref_idx in range(self.tuple_length)]
        self.intrinsic_start_idx = self.tuple_length + 1
        self.rel_pose_start_idx = self.intrinsic_start_idx + 9 * (self.tuple_length + 1)

    def get_dataset(self, split, sample_tuples):
        return self

    def _read_view(self, name):
        path = DATA_PATH / self.conf.root / name
        img = load_image(path)
        return self.preprocessor(img)

    def __getitem__(self, idx):
        line = self.items[idx]

        pair_data = line.split(" ")
        data = {}
        for i, key in enumerate(self.ref_keys):
            name0 = pair_data[0]
            name1 = pair_data[i+1]
            data0 = self._read_view(name0)
            data1 = self._read_view(name1)
            data[key] = {
                "view0": data0,
                "view1": data1,
            }
            scene = name0.split("/")[0]
            file_name0 = name0.split("/")[-1]
            a = name1.split("/")[0]
            file_name1 = name1.split("/")[-1]
            assert scene == a, f"Scene {scene} does not match {a} in {name0} and {name1}"
            data[key]['view0']["name"] = file_name0
            data[key]['view1']["name"] = file_name1
            data[key]["view0"]["scene"] = scene
            data[key]["view1"]["scene"] = scene
            data[key]["scene"] = scene

            data[key]["unique_identifier"] = "_" +  str(idx) + "_" + str(i)
            if self.conf.extra_data == "relative_pose":
                in_query_start_ind = self.intrinsic_start_idx
                in_query_end_ind = in_query_start_ind + 9
                data[key]["view0"]["camera"] = parse_camera(pair_data[in_query_start_ind:in_query_end_ind]).scale(
                    data0["scales"]
                )
                # The 9 comes from that the intrinsic matrix is 3x3
                in_start_ind = self.intrinsic_start_idx + 9*(i+1)
                in_end_ind = in_start_ind + 9
                data[key]["view1"]["camera"] = parse_camera(pair_data[in_start_ind:in_end_ind]).scale(
                    data1["scales"]
                )
                # The 12 comes from that the relative pose is 3x4
                ex_start_ind = self.rel_pose_start_idx + 12*i
                ex_end_ind = ex_start_ind + 12
                data[key]["T_0to1"] = parse_relative_pose(pair_data[ex_start_ind:ex_end_ind])
                data[key]["T_1to0"] = data[key]["T_0to1"].inv()
            elif self.conf.extra_data == "homography":
                exit("We have not implemented this yet")
                # data["H_0to1"] = (
                #     data1["transform"]
                #     @ parse_homography(pair_data[2:11])
                #     @ np.linalg.inv(data0["transform"])
                # )
            elif self.conf.extra_data == "absolute_pose":
                in_query_start_ind = self.intrinsic_start_idx
                in_query_end_ind = in_query_start_ind + 9
                data[key]["view0"]["camera"] = parse_camera(pair_data[in_query_start_ind:in_query_end_ind]).scale(
                    data0["scales"]
                )
                # The 9 comes from that the intrinsic matrix is 3x3
                in_start_ind = self.intrinsic_start_idx + 9*(i+1)
                in_end_ind = in_start_ind + 9
                data[key]["view1"]["camera"] = parse_camera(pair_data[in_start_ind:in_end_ind]).scale(
                    data1["scales"]
                )

                # The 12 comes from that the absolute pose is 3x4
                # NOTE: does not matter that it is called "rel_pose_start_idx" here
                ex_query_start_ind = self.rel_pose_start_idx
                ex_query_end_ind = ex_query_start_ind + 12
                ex_ref_i_start_ind = self.rel_pose_start_idx + 12*(i+1)
                ex_ref_i_end_ind = ex_ref_i_start_ind + 12
                # NOTE: Even if we call the function parse_relative_pose, it also works for absolute poses
                data[key]['view0']["T_w2cam"] = parse_relative_pose(pair_data[ex_query_start_ind:ex_query_end_ind])
                data[key]['view1']["T_w2cam"] = parse_relative_pose(pair_data[ex_ref_i_start_ind:ex_ref_i_end_ind])
            else:
                assert (
                    self.conf.extra_data is None
                ), f"Unknown extra data format {self.conf.extra_data}"

            data[key]["name"] = names_to_pair(name0, name1)

        return data

    def __len__(self):
        return len(self.items)

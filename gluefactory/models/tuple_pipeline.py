"""
A tuple matching pipeline. 1 query image and N reference images.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""
import time
from omegaconf import OmegaConf
from pathlib import Path

from . import get_model
from .base_model import BaseModel
from .on3r import main as on3r_main


to_ctr = OmegaConf.to_container  # convert DictConfig to dict

class TuplePipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
    }
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        self.tuple_length = conf["matcher"]["tuple_length"]
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )

    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.extractor.name and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i

    def _forward(self, data):
        t1 = time.time()
        pred = {}
        match_conf = self.conf["matcher"]
        for query_ref_key in data.keys():
            pred[query_ref_key] = self._pair_forward(data[query_ref_key])

        time_so_far = time.time() - t1
        out_folder = Path(match_conf.on3r.experiment.timings.out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        for time_method in ["ours", "transitive_matching", "motion_averaging"]:
            with open(out_folder / f'{time_method}_timings.txt', 'a') as f:
                f.write(f"\nTime: {time_so_far:.3f}")
        on3r_main.estimate_abs_pose_on3r_b(pred, match_conf, data)
        return pred

    def _pair_forward(self, data):
        # Get keypoints and descriptors
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }

        # Get matches
        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        # Filter matches
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        # Solve for pose
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}
        # Get ground truth
        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        return pred

    def loss(self):
        pass

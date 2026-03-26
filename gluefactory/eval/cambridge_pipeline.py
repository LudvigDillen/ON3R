import zipfile
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import load_model
from .utils import eval_matches_epipolar, eval_poses, eval_relative_pose_robust


class CambridgeTuplesPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "image_tuples",
            "pairs": "cambridge/calibrated_tuples.txt",
            "root": "cambridge/",  # This is correct as this is just a folder with images for the scenes
            "extra_data": "absolute_pose",  # relative_pose, homography
            "preprocessing": {
                "side": "long",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "estimator": "poselib",
            "ransac_th": 1.0,  # -1 runs a bunch of thresholds and selects the best
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []

    def _init(self, conf):
        conf.data.tuple_length = conf.model.matcher.tuple_length

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
                tuple_mode=True,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        conf = self.conf.eval
        results = defaultdict(list)
        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )
        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        pred = {}
        results_i = {}
        for i, data in enumerate(tqdm(loader)):
            for j, key in enumerate(data.keys()):
                if j > 0:  # TODO: Unclear if we should do this ...
                    continue  # TODO: This is to emulate that we only care about the first pair
                pred[key] = cache_loader(data[key])
                # add custom evaluations here
                results_i[key] = eval_matches_epipolar(data[key], pred[key])
                for th in test_thresholds:
                    pose_results_i = eval_relative_pose_robust(
                        data[key],
                        pred[key],
                        {"estimator": conf.estimator, "ransac_th": th},
                    )
                    [pose_results[th][k].append(v) for k, v in pose_results_i.items()]

                # we also store the names for later reference
                results_i[key]["names"] = data[key]["name"][0]
                if "scene" in data.keys():
                    results_i[key]["scenes"] = data[key]["scene"][0]

                for k, v in results_i[key].items():
                    results[k].append(v)

        # summarize results as a dict[str, float]
        # you can also add your custom evaluations here
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=[5, 10, 20], key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {
            **summaries,
            **best_pose_results,
        }

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            )
        }

        return summaries, figures, results

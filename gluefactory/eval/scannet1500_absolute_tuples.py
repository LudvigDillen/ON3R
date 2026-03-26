import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..settings import EVAL_PATH, set_dataset
from .io import get_eval_parser, parse_eval_args
from ..models.on3r.stats import initialize_3D_stats
from .scannet1500_pipeline import ScanNet1500TuplesPipeline
from gluefactory.utils.misc import start_debug

logger = logging.getLogger(__name__)

set_dataset("scannet1500")  # Set the default dataset to ScanNet1500


""" To evaluate the tuple_pipeline, run the following command:
python -m gluefactory.eval.<benchmark_name> --conf "a name in gluefactory/configs/ or path"
    --checkpoint "and/or a checkpoint name"

##### STAR-TOPOLOGY EVALUATION #####
python -m gluefactory.eval.scannet1500_absolute_tuples \
    --conf gluefactory/configs/superpoint+lightglue_megadepth_tuples.yaml \
    --checkpoint assets/lightglue_checkpoint.tar  \
    eval.estimator=poselib eval.ransac_th=-1 -b star_topology_300_2tuples --overwrite -d

    or     --conf /home2/lu2277di/data/on3r/outputs/training/sp+lg_megadepth/20250718_141531_id867/config.yaml \

    
nohup python -m gluefactory.eval.scannet1500_absolute_tuples \
    --conf gluefactory/configs/superpoint+lightglue_megadepth_tuples.yaml \
    --checkpoint assets/lightglue_checkpoint.tar \
    eval.estimator=poselib eval.ransac_th=-1 -b star_topology_300_2tuples \
    --overwrite > outputs/nohups/filename.txt  2>&1 &
"""
if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    parser.add_argument("-b", "--benchmark_id", type=str, default="")
    args = parser.parse_intermixed_args()

    # Start the debugger if the debug argument was given
    if args.debug:
        start_debug()
    id = "" if args.benchmark_id == "" else ("_" + args.benchmark_id)
    ScanNet1500TuplesPipeline.default_conf["data"]["pairs"] = \
        "assets/scannet/test_tuples/tuples" + id + ".txt"

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    default_conf = OmegaConf.create(ScanNet1500TuplesPipeline.default_conf)
    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )
    conf.model.matcher.max_num_keypoints = conf.model.extractor.max_num_keypoints

    subdir = "_1500tuples/" if args.benchmark_id == "" else ("_" + args.benchmark_id + "/")
    experiment_dir = output_dir / name / subdir
    experiment_dir.mkdir(exist_ok=True, parents=True)

    if conf.model.matcher.on3r.tuning_stats:
        initialize_3D_stats(conf.model.matcher.on3r.stats_file)
        initialize_3D_stats(conf.model.matcher.on3r.stats_file_transitive)
        initialize_3D_stats(conf.model.matcher.on3r.stats_file_motion_averaging)
        initialize_3D_stats(conf.model.matcher.on3r.stats_file_vggt)
        initialize_3D_stats(conf.model.matcher.on3r.stats_file_reloc3r)
        initialize_3D_stats(conf.model.matcher.on3r.stats_file_ace)

    pipeline = ScanNet1500TuplesPipeline(conf)

    pipeline.save_conf(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )
    logger.info(f"Running eval pipeline {pipeline.__class__.__name__}.")
    logger.info(f'Loop 1: Exporting predictions to "{experiment_dir}".')

    pred_file = pipeline.get_predictions(
        experiment_dir, model=None, overwrite=args.overwrite)
    logger.info(f"\nEvaluations can be found in {conf.model.matcher.on3r.stats_file} (ours)\n"
                f"                            {conf.model.matcher.on3r.stats_file_transitive} (transitive matching)\n"
                f"                            {conf.model.matcher.on3r.stats_file_motion_averaging} (motion averaging of LightGlue)\n"
                f"                            {conf.model.matcher.on3r.stats_file_vggt} (motion averaging of VGGT)\n"
                f"                            {conf.model.matcher.on3r.stats_file_reloc3r} (motion averaging of Reloc3R)")
    logger.info(f"Loop 1 finished. Predictions saved to {pred_file}.")

# Code inspired by the train script by Paul-Edouard Sarlin (skydes)
import argparse
import copy
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf
from datetime import datetime

from .. import __module_name__, logger
from ..datasets import get_dataset
from ..models import get_model
from ..settings import TRAINING_PATH
from ..utils.stdout_capturing import capture_outputs
from ..utils.tensor import batch_to_device
from ..utils.misc import start_debug
from ..models.on3r.stats import initialize_3D_stats


def run_eval(conf):
    conf.model.matcher["tuple_length"] = conf.data.tuples.tuple_length
    conf.model.matcher.max_num_keypoints = conf.model.extractor.max_num_keypoints
    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    data_conf = copy.deepcopy(conf.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device {device}")

    dataset = get_dataset(data_conf.name)(data_conf)
    val_loader = dataset.get_data_loader("val", sample_tuples=True)
    logger.info(f"Validation loader has {len(val_loader)} batches")
    model = get_model(conf.model.name)(conf.model).to(device)
    torch.backends.cudnn.benchmark = True

    model.eval()
    for data in val_loader:
        data = batch_to_device(data, device, non_blocking=True)
        _ = model(data)


if __name__ == "__main__":
    """
    When debugging/evaluating, run:
    python -m gluefactory.eval.megadepth_absolute_tuples sp+lg_megadepth \
    --conf gluefactory/configs/superpoint+lightglue_megadepth_tuples.yaml --od debugging
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--conf", type=str)
    parser.add_argument("dotlist", nargs="*")
    parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
    parser.add_argument("--output_dir", "--od", default=None, type=str)
    args = parser.parse_intermixed_args()

    if args.debug:
        start_debug()

    logger.info(f"Starting experiment {args.experiment}")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_dir = Path(TRAINING_PATH, args.experiment, current_time)
    else:
        output_dir = Path(TRAINING_PATH, args.experiment, current_time + "_" + args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    OmegaConf.save(conf, str(output_dir / "config.yaml"))
    # copy gluefactory and submodule into output dir
    for module in [__module_name__]:
        mod_dir = Path(__import__(str(module)).__file__).parent
        shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)

    if conf.model.matcher.on3r.tuning_stats:
        initialize_3D_stats(conf.model.matcher.on3r.stats_file)
        if conf.model.matcher.on3r.run_baselines:
            initialize_3D_stats(conf.model.matcher.on3r.stats_file_transitive)
            initialize_3D_stats(conf.model.matcher.on3r.stats_file_motion_averaging)
            initialize_3D_stats(conf.model.matcher.on3r.stats_file_vggt)
            initialize_3D_stats(conf.model.matcher.on3r.stats_file_reloc3r)
            initialize_3D_stats(conf.model.matcher.on3r.stats_file_ace)

    with capture_outputs(output_dir / "log.txt"):
        run_eval(conf)

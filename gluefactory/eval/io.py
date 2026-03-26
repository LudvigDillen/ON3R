import argparse
from pathlib import Path
from pprint import pprint
from typing import Optional

from omegaconf import OmegaConf

from ..models import get_model
from ..settings import TRAINING_PATH
from ..utils.experiments import load_experiment


def _find_defaults_dir(defaults: str) -> Optional[Path]:
    defaults_path = Path(defaults)
    if defaults_path.is_absolute() and defaults_path.is_dir():
        return defaults_path

    package_root = Path(__file__).resolve().parents[1]
    package_defaults = package_root / defaults_path
    if package_defaults.is_dir():
        return package_defaults

    cwd_defaults = Path.cwd() / defaults_path
    if cwd_defaults.is_dir():
        return cwd_defaults

    return None


def parse_config_path(name_or_path: Optional[str], defaults: str) -> Optional[Path]:
    default_configs = {}
    defaults_dir = _find_defaults_dir(defaults)
    if defaults_dir is not None:
        for config_path in defaults_dir.iterdir():
            if config_path.suffix == ".yaml":
                default_configs[config_path.stem] = config_path
    if name_or_path is None:
        return None
    if name_or_path in default_configs:
        return default_configs[name_or_path]
    path = Path(name_or_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find the config file: {name_or_path}. "
            f"Not in the default configs {list(default_configs.keys())} "
            "and not an existing path."
        )
    return Path(path)


def extract_benchmark_conf(conf, benchmark):
    mconf = OmegaConf.create(
        {
            "model": conf.get("model", {}),
        }
    )
    if "benchmarks" in conf.keys():
        return OmegaConf.merge(mconf, conf.benchmarks.get(benchmark, {}))
    else:
        return mconf


def parse_eval_args(benchmark, args, configs_path, default=None):
    conf = {"data": {}, "model": {}, "eval": {}}
    if args.conf:
        conf_path = parse_config_path(args.conf, configs_path)
        custom_conf = OmegaConf.load(conf_path)
        try:
            custom_conf.model.matcher["tuple_length"] = custom_conf.data.tuples.tuple_length
        except AttributeError:
            pass
        OmegaConf.resolve(custom_conf)
        conf = extract_benchmark_conf(OmegaConf.merge(conf, custom_conf), benchmark)
        args.tag = (
            args.tag if args.tag is not None else conf_path.name.replace(".yaml", "")
        )

    cli_conf = OmegaConf.from_cli(args.dotlist)
    conf = OmegaConf.merge(conf, cli_conf)
    conf.checkpoint = args.checkpoint if args.checkpoint else conf.get("checkpoint")

    if conf.checkpoint and not conf.checkpoint.endswith(".tar"):
        checkpoint_conf = OmegaConf.load(
            TRAINING_PATH / conf.checkpoint / "config.yaml"
        )
        conf = OmegaConf.merge(extract_benchmark_conf(checkpoint_conf, benchmark), conf)

    if default:
        conf = OmegaConf.merge(default, conf)

    if args.tag:
        name = args.tag
    elif args.conf and conf.checkpoint:
        name = f"{args.conf}_{conf.checkpoint}"
    elif args.conf:
        name = args.conf
    elif conf.checkpoint:
        name = conf.checkpoint
    if len(args.dotlist) > 0 and not args.tag:
        name = name + "_" + ":".join(args.dotlist)
    print("Running benchmark:", benchmark)
    print("Experiment tag:", name)
    print("Config:")
    pprint(OmegaConf.to_container(conf))
    return name, conf


def load_model(model_conf, checkpoint):
    if checkpoint:
        model = load_experiment(checkpoint, conf=model_conf).eval()
    else:
        model = get_model("two_view_pipeline")(model_conf).eval()
    if not model.is_initialized():
        raise ValueError(
            "The provided model has non-initialized parameters. "
            + "Try to load a checkpoint instead."
        )
    return model


def get_eval_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_eval", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    parser.add_argument('-d', '--debug', action='store_true', help="Run in debug mode")
    return parser

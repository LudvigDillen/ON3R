import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

import torch
from omegaconf import OmegaConf

from gluefactory.geometry.wrappers import Camera, Pose
from gluefactory.models import get_model
from gluefactory.utils.image import ImagePreprocessor, load_image
from gluefactory.utils.misc import start_debug
from gluefactory.utils.tensor import batch_to_device
from gluefactory.models.on3r.geometry import quaternion_to_matrix


def _flatten_matrix(values: Sequence, rows: int, cols: int, field_name: str) -> List[float]:
    if len(values) == rows * cols and not any(isinstance(v, (list, tuple)) for v in values):
        return [float(v) for v in values]
    if len(values) == rows and all(isinstance(v, (list, tuple)) and len(v) == cols for v in values):
        return [float(x) for row in values for x in row]
    raise ValueError(
        f"Field '{field_name}' must be a flat {rows * cols}-vector or a {rows}x{cols} matrix. "
        f"Got: {values}"
    )


def _flatten_vector(values: Sequence, dim: int, field_name: str) -> List[float]:
    if len(values) != dim or any(isinstance(v, (list, tuple)) for v in values):
        raise ValueError(f"Field '{field_name}' must be a flat {dim}-vector. Got: {values}")
    return [float(v) for v in values]


def _load_tuple_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Tuple file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "query" not in data or "references" not in data:
        raise ValueError("Tuple JSON must contain keys: 'query' and 'references'.")
    if not isinstance(data["references"], list) or len(data["references"]) == 0:
        raise ValueError("Tuple JSON must contain at least one reference in 'references'.")
    return data


def _to_pose_from_qvec_tvec(entry: Mapping, field_prefix: str) -> Pose:
    if "qvec" not in entry or "tvec" not in entry:
        raise ValueError(f"{field_prefix} must contain 'qvec' ([qw,qx,qy,qz]) and 'tvec' ([tx,ty,tz]).")

    R = quaternion_to_matrix(torch.tensor(entry["qvec"])).view(1, 3, 3)
    t = torch.tensor(_flatten_vector(entry["tvec"], 3, f"{field_prefix}.tvec"), dtype=torch.float32).view(1, 3)
    return Pose.from_Rt(R, t)


def _to_camera(K_3x3, scales: torch.Tensor, field_name: str) -> Camera:
    flat = _flatten_matrix(K_3x3, 3, 3, field_name)
    K = torch.tensor(flat, dtype=torch.float32).view(1, 3, 3)
    return Camera.from_calibration_matrix(K).scale(scales[None])


def _build_view(entry: Mapping, preprocessor: ImagePreprocessor, scene: str, role: str) -> Dict:
    if "image" not in entry or "K" not in entry or "qvec" not in entry or "tvec" not in entry:
        raise ValueError(f"{role} entry must contain: image, K, qvec, tvec")

    image_path = Path(entry["image"]).expanduser().resolve()
    image_t = load_image(image_path)
    proc = preprocessor(image_t)

    scales = proc["scales"].to(torch.float32)
    image_size = torch.as_tensor(proc["image_size"], dtype=torch.long)[None]

    view = {
        "image": proc["image"][None],
        "image_size": image_size,
        "scales": scales[None],
        "camera": _to_camera(entry["K"], scales, f"{role}.K"),
        "T_w2cam": _to_pose_from_qvec_tvec(entry, field_prefix=role),
        "name": [image_path.name],
        "scene": [scene],
    }
    return view


def _preprocessing_conf(conf) -> MutableMapping:
    try:
        return OmegaConf.to_container(conf.data.preprocessing, resolve=True)
    except Exception:
        return {"resize": 1024, "side": "long", "square_pad": True}


def run(args, conf) -> None:
    tuple_data = _load_tuple_json(args.tuple)
    scene = tuple_data.get("scene", "demo")
    refs = tuple_data["references"]
    tuple_length = len(refs)

    conf.model.matcher["tuple_length"] = tuple_length
    conf.model.matcher.on3r.plot = args.plot
    conf.model.matcher.on3r.tuning_stats = False
    try:
        conf.model.matcher.max_num_keypoints = conf.model.extractor.max_num_keypoints
    except Exception:
        pass

    preprocessor = ImagePreprocessor(_preprocessing_conf(conf))
    query_view = _build_view(tuple_data["query"], preprocessor, scene, role="query")

    data = {}
    for i, ref_entry in enumerate(refs):
        ref_view = _build_view(ref_entry, preprocessor, scene, role=f"ref{i}")
        key = f"query_to_ref_{i}"
        data[key] = {
            "view0": query_view,
            "view1": ref_view,
            "name": [f"{query_view['name'][0]}_{ref_view['name'][0]}"],
            "scene": [scene],
            "unique_identifier": [f"_{i}"],
        }

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(conf.model.name)(conf.model).to(device).eval()

    data_dev = batch_to_device(data, device, non_blocking=True)

    print(
        f"[demo.py] Running {conf.model.name} on one tuple with {tuple_length} refs "
        f"using extractor={conf.model.extractor.name}, matcher={conf.model.matcher.name}, device={device}."
    )

    pred = model(data_dev)

    print("[demo.py] Matching summary:")
    for key in sorted(pred.keys()):
        matches0 = pred[key]["matches0"][0]
        n_match = int((matches0 > -1).sum().item())
        n_total = int(matches0.numel())
        pair_name = data[key]["name"][0]
        print(f"  {key} ({pair_name}): {n_match}/{n_total} matched query keypoints")

    print("[demo.py] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-tuple ON3R demo: known intrinsics + qvec/tvec poses + extractor/matcher + absolute pose."
    )
    parser.add_argument("--tuple", type=Path, required=True, help="Path to tuple JSON file.")
    parser.add_argument(
        "--conf",
        type=str,
        default="gluefactory/configs/superpoint+lightglue_megadepth_tuples.yaml",
        help="Glue Factory config (extractor/matcher/ON3R settings).",
    )
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("-d", "--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the results (only for ON3R).")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()

    if args.debug:
        start_debug()

    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)

    run(args, conf)

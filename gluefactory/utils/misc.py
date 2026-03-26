import torch
from collections.abc import Mapping, Sequence
from typing import Any, List, Tuple, Union
import debugpy

Path = Tuple[Union[str, int], ...]  # keys for dicts, indices for lists/tuples


def to_view(data, i):
    return {k + i: v for k, v in data.items()}


def get_view(data, i):
    data_g = {k: v for k, v in data.items() if not k[-1].isnumeric()}
    data_i = {k[:-1]: v for k, v in data.items() if k[-1] == i}
    return {**data_g, **data_i}


def get_twoview(data, idx):
    li = idx[0]
    ri = idx[-1]
    assert idx == f"{li}to{ri}"
    data_lr = {k[:-4] + "0to1": v for k, v in data.items() if k[-4:] == f"{li}to{ri}"}
    data_rl = {k[:-4] + "1to0": v for k, v in data.items() if k[-4:] == f"{ri}ito{li}"}
    data_l = {
        k[:-1] + "0": v for k, v in data.items() if k[-1:] == li and k[-3:-1] != "to"
    }
    data_r = {
        k[:-1] + "1": v for k, v in data.items() if k[-1:] == ri and k[-3:-1] != "to"
    }
    return {**data_lr, **data_rl, **data_l, **data_r}


def stack_twoviews(data, indices=["0to1", "0to2", "1to2"]):
    idx0 = indices[0]
    m_data = data[idx0] if idx0 in data else get_twoview(data, idx0)
    # stack on dim=0
    for idx in indices[1:]:
        data_i = data[idx] if idx in data else get_twoview(data, idx)
        for k, v in data_i.items():
            m_data[k] = torch.cat([m_data[k], v], dim=0)
    return m_data


def unstack_twoviews(data, B, indices=["0to1", "0to2", "1to2"]):
    out = {}
    for i, idx in enumerate(indices):
        out[idx] = {k: v[i * B : (i + 1) * B] for k, v in data.items()}
    return out


def find_all_key_paths(
    obj: Any,
    target_key: str,
    *,
    _path: Path = (),
    _seen: set[int] | None = None,
) -> List[Path]:
    """
    Return all paths leading to occurrences of `target_key` in nested mappings/lists/tuples.
    Each path is a tuple like ('data', 'tuples', 'tuple_length') or ('items', 3, 'name').

    - Dict-like nodes are traversed by keys.
    - Lists/tuples are traversed by indices.
    - Strings/bytes are not descended into.
    - Cycles are guarded by object id tracking.
    """
    if _seen is None:
        _seen = set()

    out: List[Path] = []
    oid = id(obj)
    if oid in _seen:
        return out
    _seen.add(oid)

    # Descend into dict-like objects
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            new_path = _path + (k,)
            if k == target_key:
                out.append(new_path)
            out.extend(find_all_key_paths(v, target_key, _path=new_path, _seen=_seen))
        return out

    # Descend into sequences but not into strings/bytes
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for i, item in enumerate(obj):
            out.extend(find_all_key_paths(item, target_key, _path=_path + (i,), _seen=_seen))
        return out

    # Scalars / non-iterables: nothing to do
    return out


def get_by_path(obj: Any, path: Path) -> Any:
    """Follow a path returned by `find_all_key_paths` to fetch the value."""
    cur = obj
    for step in path:
        cur = cur[step]
    return cur


def format_path(path: Path) -> str:
    """Pretty string like data.tuples.tuple_length or items[3].name"""
    out: List[str] = []
    for step in path:
        if isinstance(step, int):
            out[-1] = f"{out[-1]}[{step}]"
        else:
            out.append(str(step))
    return ".".join(out)


def start_debug():
    debugpy.listen(5678)
    print("Wait for debugger!")
    debugpy.wait_for_client()
    print("Attached!")

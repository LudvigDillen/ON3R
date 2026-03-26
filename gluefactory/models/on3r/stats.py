import os
import re
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional
import time


def initialize_3D_stats(filepath):
    """
    Appends a new chunk to the file by writing a header line and
    a placeholder mean row.
    Each chunk begins with a header line.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        # Now also mention rotation/translation error **and** number of 3‑D points in the header
        f.write("\nValidation: recall@[3, 5, 10, 25, 50]px, time, error_R(°), error_t(m), n_3d_pts, (0.1m, 1°) / (0.25m, 2°) / (1.0m, 5°) / (5.0m, 10°)\n")
        # Placeholder mean row ("-" for yet‑unknown values)
        f.write("Mean row: (-, -, -, -, -), -s, -°, -m, -pts, (-, -, -, -)\n")


def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return None


def calculate_means_last_chunk(filepath):
    """Read the last validation chunk and return the column‑wise means, incl. n_3d_pts."""

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the *last* header starting with "Validation:"
    header_indices = [i for i, line in enumerate(lines) if line.startswith("Validation:")]
    if not header_indices:
        return None
    last_header_index = header_indices[-1]

    # Gather sample rows (everything up to the next header or EOF, skipping the mean row)
    sample_rows = []
    for line in lines[last_header_index + 2:]:  # +2 → skip header and placeholder mean row
        if line.startswith("Validation:"):
            break
        if line.strip():
            sample_rows.append(line.strip())

    if not sample_rows:
        return None

    # ──────────────────────────────────────────────────────────────────────────────
    # Regex that captures:
    #   1) the comma‑separated recall string inside parentheses
    #   2) elapsed time (s)    3) rot error (°)    4) trans error (m)
    #   5) number of 3‑D points
    # Example row:
    #   (0.123, 0.456, 0.789, 0.111, 0.222), 12.34s, 1.23°, 0.04m, 3852
    # ──────────────────────────────────────────────────────────────────────────────
    pattern = re.compile(r"\(\s*([^)]*?)\s*\),\s*([\d.]+)s,\s*([\d.]+)°,\s*([\d.]+)m,\s*([\d.]+)")

    # Column collectors
    col_recall = [[] for _ in range(5)]  # five recall columns
    col_time, col_rot, col_trans, col_pts = [], [], [], []
    threshold_counts = [0, 0, 0, 0]
    total_valid = 0

    thresholds = [  # (translation, rotation)
        (0.1, 1.0),   # 0.1 m & 1°
        (0.25, 2.0),  # 0.25 m & 2°
        (1.0, 5.0),   # 1.0 m & 5°
        (5.0, 10.0),  # 5.0 m & 10° (not used in the stats, but kept for consistency)
    ]

    for i, line in enumerate(sample_rows):
        # only touch lines that actually contain “nan°” or “nanm”
        if 'nan°' in line or 'nanm' in line:
            # replace the first nan (rotation) with 180° and the second (translation) with 100m
            line = line.replace('nan°', '180.0°', 1) \
                       .replace('nanm', '100.0m', 1)
            sample_rows[i] = line

        m = pattern.search(line)
        if not m:
            continue

        # ── Recalls ───────────────────────────────────────────────────────
        recall_parts = [p.strip() for p in m.group(1).split(',')]
        try:
            recall_vals = [float(val) for val in recall_parts]
            if len(recall_vals) == 5:
                for i, v in enumerate(recall_vals):
                    col_recall[i].append(v)
        except ValueError:
            pass  # skip if any recall is non‑numeric ("N/A")

        # ── Time / pose errors / #pts ──────────────────────────────────
        time_val  = safe_float(m.group(2))
        rot_val   = safe_float(m.group(3))
        trans_val = safe_float(m.group(4))
        pts_val   = safe_float(m.group(5))

        col_time.append(time_val)
        col_rot.append(rot_val)
        col_trans.append(trans_val)
        if pts_val is not None:
            col_pts.append(pts_val)

        # Update threshold success counters
        for i, (t_thresh, r_thresh) in enumerate(thresholds):
            if trans_val < t_thresh and rot_val < r_thresh:
                threshold_counts[i] += 1
        total_valid += 1

    # Helper mean (return None when list is empty)
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else None
    def safe_median(lst):
        return np.median(lst) if lst else None

    means = {
        **{f"recall{i+1}": safe_mean(col) for i, col in enumerate(col_recall)},
        "time":         safe_mean(col_time),
        "rot_error":    safe_median(col_rot),
        "trans_error":  safe_median(col_trans),
        "n_3d_pts":     safe_mean(col_pts),
        "thresholds":   [(c / total_valid) if total_valid else None for c in threshold_counts],
    }
    return means


def update_mean_row_last_chunk(filepath):
    """Re‑write the placeholder mean row of the last chunk with freshly computed means."""

    means = calculate_means_last_chunk(filepath)
    if means is None:
        return

    # ── Small formatter helpers ──────────────────────────────────
    def fmt(val, pattern="{:.3f}"):
        return pattern.format(val) if val is not None else "N/A"

    thresholds_str = "({} / {} / {} / {})".format(
        fmt(means["thresholds"][0]),
        fmt(means["thresholds"][1]),
        fmt(means["thresholds"][2]),
        fmt(means["thresholds"][3]),
    )

    mean_line = (
        "Mean row: ({}, {}, {}, {}, {}), {}s, {}°, {}m, {}pts, {}\n"
    ).format(
        fmt(means["recall1"]), fmt(means["recall2"]), fmt(means["recall3"]), fmt(means["recall4"]), fmt(means["recall5"]),
        fmt(means["time"], "{:.2f}"),
        fmt(means["rot_error"], "{:.2f}"),
        fmt(means["trans_error"], "{:.2f}"),
        fmt(means["n_3d_pts"], "{:.0f}"),
        thresholds_str,
    )

    # ── Replace placeholder mean row ───────────────────────────────
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_indices = [i for i, l in enumerate(lines) if l.startswith("Validation:")]
    if not header_indices:
        return
    last_header = header_indices[-1]

    if len(lines) > last_header + 1:
        lines[last_header + 1] = mean_line + ("" if lines[last_header + 1].endswith("\n") else "\n")
    else:
        # Shouldn't really happen, but keep file consistent.
        lines.append(mean_line + "\n")

    with open(filepath, 'w') as f:
        f.writelines(lines)


def append_sample_to_last_chunk(filepath, recalls_val, elapsed_time, error_R, error_t, n_3d_pts):
    """Append a single sample line to the last validation chunk and refresh the mean row."""

    if recalls_val is None:
        sample = "(N/A, N/A, N/A, N/A, N/A), {:.2f}s, {:.2f}°, {:.2f}m, {:.0f}".format(
            elapsed_time, error_R, error_t, n_3d_pts
        )
    else:
        sample = "({:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}), {:.2f}s, {:.2f}°, {:.2f}m, {:.0f}".format(
            *recalls_val, elapsed_time, error_R, error_t, n_3d_pts
        )

    with open(filepath, 'a') as f:
        f.write(sample + "\n")

    # Refresh mean row afterwards
    update_mean_row_last_chunk(filepath)



# --- reusable helpers --------------------------------------------------------

_FLOAT = re.compile(r"[-+]?\d*\.\d+|\d+")          # matches 12, 1.2, .3, -4.5 …

def _to_float(token: str) -> Optional[float]:
    """Convert a numeric token to float, return None on '-', 'N/A', etc."""
    return float(token) if _FLOAT.fullmatch(token) else None


def _parse_tuple(segment: str) -> List[Optional[float]]:
    """Parse a parenthesised tuple like '(0.000 / 0.750 / 1.000)' or
    '(N/A, N/A, …)' into a list of floats / None."""
    segment = segment.strip("()")
    # handle both ',' and '/' as separators
    tokens = re.split(r"[,/]", segment)
    return [_to_float(tok.strip()) for tok in tokens]


def read_last_mean_row(file_path: str | Path) -> Dict[str, Any]:
    """
    Read *only the last* line that starts with 'Mean row:' in `file_path`
    and return its values in a dict.

    Returns
    -------
    dict
        {
            "recall":           [f1, f2, f3, f4, f5]  or list[None],
            "time_s":        float | None,
            "error_R_deg":   float | None,
            "error_t_m":     float | None,
            "n_3d_pts":      int   | None,
            "recall":        [r1, r2, r3, r4]          or list[None],
        }
        If the file contains no 'Mean row:' at all every value is None.
    """
    path = Path(file_path)
    last_line: Optional[str] = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Mean row:"):
                last_line = line.strip()

    # Nothing found → return an all-None dict
    if last_line is None:
        return {
            "recall": [None] * 5,
            "time_s": None,
            "error_R_deg": None,
            "error_t_m": None,
            "n_3d_pts": None,
            "recall": [None] * 4,
        }

    # Strip the label and split on commas that separate the columns
    _, payload = last_line.split(":", maxsplit=1)
    cols = [c.strip() for c in payload.split(",")]

    # Column 0  : 5-tuple of recalls
    recall = _parse_tuple(cols[0] + ", " + cols[1] + ", " + cols[2] + ", " + cols[3] + ", " + cols[4])

    # Column 1  : time, strip 's'
    time_s = _to_float(cols[5].rstrip("s"))

    # Column 2  : R-error, strip '°'
    error_R_deg = _to_float(cols[6].rstrip("°"))

    # Column 3  : t-error, strip 'm'
    error_t_m = _to_float(cols[7].rstrip("m"))

    # Column 4  : number of points, strip 'pts'
    n_3d_pts = _to_float(cols[8].rstrip("pts"))
    n_3d_pts = int(n_3d_pts) if n_3d_pts is not None else None

    # Column 5  : 4-tuple of recalls
    recall = _parse_tuple(cols[9])

    return {
        "recall": recall,
        "time_s": time_s,
        "error_R_deg": error_R_deg,
        "error_t_m": error_t_m,
        "n_3d_pts": n_3d_pts,
        "recall": recall,
    }


def update_last_line_time(file, extra_time):
    t1 = time.time()
    with open(file, 'r+') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()
        last_time = float(last_line.split(":")[1].strip())
        total_time = last_time + extra_time
        lines[-1] = f"Time: {total_time:.3f}\n"
        f.seek(0)
        f.writelines(lines)
        f.truncate()
    t2 = time.time()
    if t2 - t1 > 0.01:
        print(f"Warning: updating last line time took {t2 - t1:.3f}s")


def read_last_line_time(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()
        last_time = float(last_line.split(":")[1].strip())
    return last_time

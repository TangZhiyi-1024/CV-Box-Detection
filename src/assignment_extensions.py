import csv
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .box_measure import box_height, box_length_width
from .components import largest_component
from .geometry import point_plane_signed_distance
from .io_utils import ensure_dir, load_example
from .masks import clean_mask
from .ransac import preemptive_ransac_plane, ransac_plane


@dataclass
class EstimateResult:
    dataset: str
    method: str
    thresh: float
    m: int
    b: int
    runtime_sec: float
    height: float
    length: float
    width: float
    floor_inliers: int
    top_inliers: int


def _fit_plane(
    pc: np.ndarray,
    mask: np.ndarray,
    method: str,
    thresh: float,
    max_iters: int,
    m: int,
    b: int,
) -> Tuple[np.ndarray, float, np.ndarray]:
    if method == "preemptive":
        return preemptive_ransac_plane(pc, mask, thresh=thresh, M=m, B=b)
    return ransac_plane(pc, mask, thresh=thresh, max_iters=max_iters, mode=method)


def estimate_box(
    mat_path: str,
    method: str,
    thresh: float,
    max_iters: int = 1000,
    m: int = 256,
    b: int = 200,
    floor_open: int = 3,
    floor_close: int = 5,
    top_open: int = 1,
    top_close: int = 3,
    min_height_from_floor: float = 0.05,
) -> EstimateResult:
    dataset = Path(mat_path).stem
    _, _, pc, valid = load_example(mat_path)

    t0 = time.perf_counter()

    n_floor, d_floor, floor_mask = _fit_plane(
        pc, valid, method=method, thresh=thresh, max_iters=max_iters, m=m, b=b
    )
    floor_mask = clean_mask(floor_mask, floor_open, floor_close)

    not_floor = valid & (~floor_mask)
    dist_to_floor = np.abs(point_plane_signed_distance(pc, n_floor, d_floor))
    not_floor = not_floor & (dist_to_floor > min_height_from_floor)

    n_top, d_top, top_mask_candidates = _fit_plane(
        pc, not_floor, method=method, thresh=thresh, max_iters=max_iters, m=m, b=b
    )
    top_mask_candidates = clean_mask(top_mask_candidates, top_open, top_close)
    box_top_mask, _ = largest_component(top_mask_candidates)

    h = box_height(n_floor, d_floor, n_top, d_top)
    l, w = box_length_width(pc, box_top_mask)

    runtime = time.perf_counter() - t0
    return EstimateResult(
        dataset=dataset,
        method=method,
        thresh=thresh,
        m=m,
        b=b,
        runtime_sec=runtime,
        height=float(h),
        length=float(l),
        width=float(w),
        floor_inliers=int(floor_mask.sum()),
        top_inliers=int(box_top_mask.sum()),
    )


def _write_csv(path: str, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _collect_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "example*kinect.mat")))
    if not files:
        raise FileNotFoundError(f"No .mat files found in: {data_dir}")
    return files


def run_mlesac_epsilon_study(
    data_dir: str,
    out_dir: str,
    eps_values: Iterable[float] = (0.003, 0.005, 0.007, 0.01),
) -> Tuple[str, str]:
    ensure_dir(out_dir)
    files = _collect_files(data_dir)

    rows: List[Dict[str, object]] = []
    for method in ("ransac", "mlesac"):
        for eps in eps_values:
            for mat in files:
                r = estimate_box(mat, method=method, thresh=float(eps))
                rows.append(
                    {
                        "dataset": r.dataset,
                        "method": r.method,
                        "epsilon": r.thresh,
                        "runtime_sec": round(r.runtime_sec, 4),
                        "height_m": round(r.height, 4),
                        "length_m": round(r.length, 4),
                        "width_m": round(r.width, 4),
                        "floor_inliers": r.floor_inliers,
                        "top_inliers": r.top_inliers,
                    }
                )

    csv_path = str(Path(out_dir) / "mlesac_epsilon_sweep.csv")
    _write_csv(csv_path, rows)

    # Summarize epsilon sensitivity by standard deviation across eps for each dataset and dimension.
    by_key: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        k = (str(row["method"]), str(row["dataset"]))
        by_key.setdefault(k, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (method, dataset), vals in sorted(by_key.items()):
        h = np.array([float(v["height_m"]) for v in vals], dtype=float)
        l = np.array([float(v["length_m"]) for v in vals], dtype=float)
        w = np.array([float(v["width_m"]) for v in vals], dtype=float)
        rt = np.array([float(v["runtime_sec"]) for v in vals], dtype=float)
        summary_rows.append(
            {
                "method": method,
                "dataset": dataset,
                "height_std_m": round(float(h.std()), 5),
                "length_std_m": round(float(l.std()), 5),
                "width_std_m": round(float(w.std()), 5),
                "mean_runtime_sec": round(float(rt.mean()), 4),
            }
        )

    summary_csv = str(Path(out_dir) / "mlesac_epsilon_summary.csv")
    _write_csv(summary_csv, summary_rows)
    return csv_path, summary_csv


def run_preemptive_study(
    data_dir: str,
    out_dir: str,
    m_values: Iterable[int] = (64, 256, 1024),
    b_values: Iterable[int] = (100, 200, 400),
    thresh: float = 0.005,
) -> Tuple[str, str]:
    ensure_dir(out_dir)
    files = _collect_files(data_dir)

    baseline: Dict[str, EstimateResult] = {}
    for mat in files:
        r = estimate_box(mat, method="ransac", thresh=thresh)
        baseline[r.dataset] = r

    rows: List[Dict[str, object]] = []
    for m in m_values:
        for b in b_values:
            for mat in files:
                r = estimate_box(mat, method="preemptive", thresh=thresh, m=m, b=b)
                bline = baseline[r.dataset]
                rows.append(
                    {
                        "dataset": r.dataset,
                        "method": r.method,
                        "epsilon": r.thresh,
                        "M": r.m,
                        "B": r.b,
                        "runtime_sec": round(r.runtime_sec, 4),
                        "height_m": round(r.height, 4),
                        "length_m": round(r.length, 4),
                        "width_m": round(r.width, 4),
                        "abs_err_height_vs_ransac_m": round(abs(r.height - bline.height), 4),
                        "abs_err_length_vs_ransac_m": round(abs(r.length - bline.length), 4),
                        "abs_err_width_vs_ransac_m": round(abs(r.width - bline.width), 4),
                    }
                )

    csv_path = str(Path(out_dir) / "preemptive_ransac_sweep.csv")
    _write_csv(csv_path, rows)

    by_budget: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for row in rows:
        k = (int(row["M"]), int(row["B"]))
        by_budget.setdefault(k, []).append(row)

    summary_rows: List[Dict[str, object]] = []
    for (m, b), vals in sorted(by_budget.items()):
        rt = np.array([float(v["runtime_sec"]) for v in vals], dtype=float)
        eh = np.array([float(v["abs_err_height_vs_ransac_m"]) for v in vals], dtype=float)
        el = np.array([float(v["abs_err_length_vs_ransac_m"]) for v in vals], dtype=float)
        ew = np.array([float(v["abs_err_width_vs_ransac_m"]) for v in vals], dtype=float)
        summary_rows.append(
            {
                "M": m,
                "B": b,
                "mean_runtime_sec": round(float(rt.mean()), 4),
                "mean_abs_err_height_m": round(float(eh.mean()), 5),
                "mean_abs_err_length_m": round(float(el.mean()), 5),
                "mean_abs_err_width_m": round(float(ew.mean()), 5),
            }
        )

    summary_csv = str(Path(out_dir) / "preemptive_ransac_summary.csv")
    _write_csv(summary_csv, summary_rows)
    return csv_path, summary_csv


def save_preemptive_m_visualization(
    mat_path: str,
    out_path: str,
    m_values: Iterable[int] = (64, 256, 1024),
    b: int = 200,
    thresh: float = 0.005,
) -> str:
    _, d, pc, valid = load_example(mat_path)
    m_values = list(m_values)
    fig, axes = plt.subplots(1, len(m_values), figsize=(4.0 * len(m_values), 4.2))
    if len(m_values) == 1:
        axes = [axes]

    for ax, m in zip(axes, m_values):
        t0 = time.perf_counter()
        _, _, floor_mask = preemptive_ransac_plane(pc, valid, thresh=thresh, M=m, B=b)
        floor_mask = clean_mask(floor_mask, open_size=3, close_size=5)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Overlay floor mask on distance image.
        base = np.asarray(d, dtype=float)
        p2, p98 = np.percentile(base[np.isfinite(base)], [2, 98])
        base = np.clip((base - p2) / (p98 - p2 + 1e-6), 0, 1)
        rgb = np.stack([base, base, base], axis=-1)
        rgb[floor_mask] = [0.2, 0.95, 0.2]
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"M={m}, B={b}\n{dt_ms:.1f} ms")
        ax.axis("off")

    fig.suptitle(f"Preemptive RANSAC floor plane: {Path(mat_path).stem}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


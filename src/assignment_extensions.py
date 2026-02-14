import csv
import glob
import hashlib
import os
import time

import numpy as np

from .box_measure import box_height, box_length_width
from .components import largest_component
from .geometry import point_plane_signed_distance
from .io_utils import ensure_dir, load_example
from .masks import clean_mask
from .ransac import mlesac_plane, preemptive_ransac_plane, ransac_plane

FULL_EPS_VALUES = (0.003, 0.005, 0.007, 0.01)
FULL_M_VALUES = (64, 256, 1024)
FULL_B_VALUES = (100, 200, 400)
FULL_PREEMPTIVE_EPS = 0.005


def _seed_from_key(*parts):
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def _fit_plane(
    method,
    PC,
    valid_mask,
    eps,
    max_iters,
    gamma_factor=1.5,
    M=256,
    B=200,
    score_mode="msac",
    rng_seed=42,
):
    rng = np.random.default_rng(rng_seed)
    if method == "ransac":
        return ransac_plane(PC, valid_mask, thresh=eps, max_iters=max_iters, rng=rng)
    if method == "mlesac":
        gamma = gamma_factor * eps
        return mlesac_plane(PC, valid_mask, eps=eps, gamma=gamma, max_iters=max_iters, rng=rng)
    if method == "preemptive":
        return preemptive_ransac_plane(
            PC,
            valid_mask,
            thresh=eps,
            M=M,
            B=B,
            score_mode=score_mode,
            gamma=gamma_factor * eps,
            rng=rng,
        )
    raise ValueError("method must be one of {'ransac', 'mlesac', 'preemptive'}")


def run_plane_pipeline(
    mat_path,
    method="ransac",
    eps=0.005,
    max_iters=1000,
    gamma_factor=1.5,
    M=256,
    B=200,
    score_mode="msac",
):
    A, D, PC, valid = load_example(mat_path)

    seed_base = _seed_from_key(mat_path, method, eps, M, B, score_mode)

    t0 = time.perf_counter()

    n_floor, d_floor, floor_mask = _fit_plane(
        method=method,
        PC=PC,
        valid_mask=valid,
        eps=eps,
        max_iters=max_iters,
        gamma_factor=gamma_factor,
        M=M,
        B=B,
        score_mode=score_mode,
        rng_seed=seed_base,
    )
    floor_mask = clean_mask(floor_mask, 3, 5)

    not_floor = valid & (~floor_mask)
    dist_to_floor = np.abs(point_plane_signed_distance(PC, n_floor, d_floor))
    not_floor = not_floor & (dist_to_floor > 0.05)

    n_top, d_top, top_mask_candidates = _fit_plane(
        method=method,
        PC=PC,
        valid_mask=not_floor,
        eps=eps,
        max_iters=max_iters,
        gamma_factor=gamma_factor,
        M=M,
        B=B,
        score_mode=score_mode,
        rng_seed=seed_base + 1,
    )

    top_mask_candidates = clean_mask(top_mask_candidates, 1, 3)
    box_top_mask, _ = largest_component(top_mask_candidates)

    h = box_height(n_floor, d_floor, n_top, d_top)
    l, w = box_length_width(PC, box_top_mask)

    runtime_sec = time.perf_counter() - t0

    return {
        "A": A,
        "D": D,
        "PC": PC,
        "valid": valid,
        "n_floor": n_floor,
        "d_floor": d_floor,
        "n_top": n_top,
        "d_top": d_top,
        "floor_mask": floor_mask,
        "box_top_mask": box_top_mask,
        "height_m": float(h),
        "length_m": float(l),
        "width_m": float(w),
        "floor_inliers": int(floor_mask.sum()),
        "top_inliers": int(box_top_mask.sum()),
        "runtime_sec": float(runtime_sec),
    }


def _write_csv(rows, out_path, fieldnames):
    ensure_dir(os.path.dirname(out_path) or ".")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _list_mat_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "example*kinect.mat")))
    if not files:
        raise FileNotFoundError(f"No files found with pattern {data_dir}/example*kinect.mat")
    return files


def sweep_mlesac_eps(
    data_dir="data",
    eps_values=(0.003, 0.005, 0.007, 0.01),
    max_iters=1000,
    gamma_factor=1.5,
    out_sweep_csv="outputs/mlesac_epsilon_sweep.csv",
    out_summary_csv="outputs/mlesac_epsilon_summary.csv",
):
    rows = []
    files = _list_mat_files(data_dir)

    for eps in eps_values:
        for mat_path in files:
            dataset = os.path.basename(mat_path).replace(".mat", "")
            for method in ("ransac", "mlesac"):
                r = run_plane_pipeline(
                    mat_path=mat_path,
                    method=method,
                    eps=eps,
                    max_iters=max_iters,
                    gamma_factor=gamma_factor,
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "epsilon": eps,
                        "runtime_sec": round(r["runtime_sec"], 4),
                        "height_m": round(r["height_m"], 4),
                        "length_m": round(r["length_m"], 4),
                        "width_m": round(r["width_m"], 4),
                        "floor_inliers": r["floor_inliers"],
                        "top_inliers": r["top_inliers"],
                    }
                )

    _write_csv(
        rows,
        out_sweep_csv,
        fieldnames=[
            "dataset",
            "method",
            "epsilon",
            "runtime_sec",
            "height_m",
            "length_m",
            "width_m",
            "floor_inliers",
            "top_inliers",
        ],
    )

    summary = []
    methods = sorted({r["method"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})

    for method in methods:
        for dataset in datasets:
            sel = [r for r in rows if r["method"] == method and r["dataset"] == dataset]
            hs = np.array([r["height_m"] for r in sel], dtype=np.float64)
            ls = np.array([r["length_m"] for r in sel], dtype=np.float64)
            ws = np.array([r["width_m"] for r in sel], dtype=np.float64)
            ts = np.array([r["runtime_sec"] for r in sel], dtype=np.float64)

            summary.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "height_std_m": round(float(hs.std(ddof=0)), 5),
                    "length_std_m": round(float(ls.std(ddof=0)), 5),
                    "width_std_m": round(float(ws.std(ddof=0)), 5),
                    "mean_runtime_sec": round(float(ts.mean()), 4),
                }
            )

    _write_csv(
        summary,
        out_summary_csv,
        fieldnames=[
            "method",
            "dataset",
            "height_std_m",
            "length_std_m",
            "width_std_m",
            "mean_runtime_sec",
        ],
    )

    return out_sweep_csv, out_summary_csv


def sweep_preemptive_mb(
    data_dir="data",
    eps=0.005,
    M_values=(64, 256, 1024),
    B_values=(100, 200, 400),
    max_iters=1000,
    score_mode="msac",
    out_sweep_csv="outputs/preemptive_ransac_sweep.csv",
    out_summary_csv="outputs/preemptive_ransac_summary.csv",
):
    files = _list_mat_files(data_dir)

    baseline = {}
    for mat_path in files:
        dataset = os.path.basename(mat_path).replace(".mat", "")
        r = run_plane_pipeline(
            mat_path=mat_path,
            method="ransac",
            eps=eps,
            max_iters=max_iters,
        )
        baseline[dataset] = r

    rows = []
    for M in M_values:
        for B in B_values:
            for mat_path in files:
                dataset = os.path.basename(mat_path).replace(".mat", "")
                r = run_plane_pipeline(
                    mat_path=mat_path,
                    method="preemptive",
                    eps=eps,
                    max_iters=max_iters,
                    M=M,
                    B=B,
                    score_mode=score_mode,
                )
                base = baseline[dataset]

                rows.append(
                    {
                        "dataset": dataset,
                        "method": "preemptive",
                        "epsilon": eps,
                        "M": int(M),
                        "B": int(B),
                        "runtime_sec": round(r["runtime_sec"], 4),
                        "height_m": round(r["height_m"], 4),
                        "length_m": round(r["length_m"], 4),
                        "width_m": round(r["width_m"], 4),
                        "abs_err_height_vs_ransac_m": round(
                            abs(r["height_m"] - base["height_m"]), 4
                        ),
                        "abs_err_length_vs_ransac_m": round(
                            abs(r["length_m"] - base["length_m"]), 4
                        ),
                        "abs_err_width_vs_ransac_m": round(
                            abs(r["width_m"] - base["width_m"]), 4
                        ),
                    }
                )

    _write_csv(
        rows,
        out_sweep_csv,
        fieldnames=[
            "dataset",
            "method",
            "epsilon",
            "M",
            "B",
            "runtime_sec",
            "height_m",
            "length_m",
            "width_m",
            "abs_err_height_vs_ransac_m",
            "abs_err_length_vs_ransac_m",
            "abs_err_width_vs_ransac_m",
        ],
    )

    summary = []
    for M in M_values:
        for B in B_values:
            sel = [r for r in rows if int(r["M"]) == int(M) and int(r["B"]) == int(B)]
            t = np.array([r["runtime_sec"] for r in sel], dtype=np.float64)
            eh = np.array([r["abs_err_height_vs_ransac_m"] for r in sel], dtype=np.float64)
            el = np.array([r["abs_err_length_vs_ransac_m"] for r in sel], dtype=np.float64)
            ew = np.array([r["abs_err_width_vs_ransac_m"] for r in sel], dtype=np.float64)

            summary.append(
                {
                    "M": int(M),
                    "B": int(B),
                    "mean_runtime_sec": round(float(t.mean()), 4),
                    "mean_abs_err_height_m": round(float(eh.mean()), 5),
                    "mean_abs_err_length_m": round(float(el.mean()), 5),
                    "mean_abs_err_width_m": round(float(ew.mean()), 5),
                }
            )

    _write_csv(
        summary,
        out_summary_csv,
        fieldnames=[
            "M",
            "B",
            "mean_runtime_sec",
            "mean_abs_err_height_m",
            "mean_abs_err_length_m",
            "mean_abs_err_width_m",
        ],
    )

    return out_sweep_csv, out_summary_csv


def compare_extension_results(
    mlesac_summary_csv="outputs/mlesac_epsilon_summary.csv",
    preemptive_summary_csv="outputs/preemptive_ransac_summary.csv",
    out_mlesac_compare_csv="outputs/mlesac_vs_ransac_comparison.csv",
    out_preemptive_rank_csv="outputs/preemptive_tradeoff_ranking.csv",
):
    # Part A: MLESAC vs RANSAC stability/runtime comparison
    ms_rows = _read_csv_rows(mlesac_summary_csv)
    by_dataset = {}
    for r in ms_rows:
        ds = r["dataset"]
        method = r["method"]
        by_dataset.setdefault(ds, {})[method] = r

    cmp_rows = []
    for dataset, mm in sorted(by_dataset.items()):
        if "ransac" not in mm or "mlesac" not in mm:
            continue
        r = mm["ransac"]
        m = mm["mlesac"]

        r_h = float(r["height_std_m"])
        r_l = float(r["length_std_m"])
        r_w = float(r["width_std_m"])
        m_h = float(m["height_std_m"])
        m_l = float(m["length_std_m"])
        m_w = float(m["width_std_m"])
        r_t = float(r["mean_runtime_sec"])
        m_t = float(m["mean_runtime_sec"])

        def pct_gain(base, new):
            if abs(base) < 1e-12:
                return 0.0
            return (base - new) / base * 100.0

        cmp_rows.append(
            {
                "dataset": dataset,
                "ransac_height_std_m": round(r_h, 5),
                "mlesac_height_std_m": round(m_h, 5),
                "height_std_reduction_pct": round(pct_gain(r_h, m_h), 2),
                "ransac_length_std_m": round(r_l, 5),
                "mlesac_length_std_m": round(m_l, 5),
                "length_std_reduction_pct": round(pct_gain(r_l, m_l), 2),
                "ransac_width_std_m": round(r_w, 5),
                "mlesac_width_std_m": round(m_w, 5),
                "width_std_reduction_pct": round(pct_gain(r_w, m_w), 2),
                "ransac_runtime_sec": round(r_t, 4),
                "mlesac_runtime_sec": round(m_t, 4),
                "runtime_delta_sec_mlesac_minus_ransac": round(m_t - r_t, 4),
            }
        )

    _write_csv(
        cmp_rows,
        out_mlesac_compare_csv,
        fieldnames=[
            "dataset",
            "ransac_height_std_m",
            "mlesac_height_std_m",
            "height_std_reduction_pct",
            "ransac_length_std_m",
            "mlesac_length_std_m",
            "length_std_reduction_pct",
            "ransac_width_std_m",
            "mlesac_width_std_m",
            "width_std_reduction_pct",
            "ransac_runtime_sec",
            "mlesac_runtime_sec",
            "runtime_delta_sec_mlesac_minus_ransac",
        ],
    )

    # Part B: Preemptive runtime-accuracy trade-off ranking
    pr_rows = _read_csv_rows(preemptive_summary_csv)
    parsed = []
    for r in pr_rows:
        e_h = float(r["mean_abs_err_height_m"])
        e_l = float(r["mean_abs_err_length_m"])
        e_w = float(r["mean_abs_err_width_m"])
        total_err = (e_h + e_l + e_w) / 3.0
        parsed.append(
            {
                "M": int(r["M"]),
                "B": int(r["B"]),
                "mean_runtime_sec": float(r["mean_runtime_sec"]),
                "mean_abs_err_height_m": e_h,
                "mean_abs_err_length_m": e_l,
                "mean_abs_err_width_m": e_w,
                "mean_abs_err_total_m": total_err,
            }
        )

    # Rank by runtime and by total error.
    runtime_sorted = sorted(parsed, key=lambda x: x["mean_runtime_sec"])
    err_sorted = sorted(parsed, key=lambda x: x["mean_abs_err_total_m"])
    rank_runtime = {(r["M"], r["B"]): i + 1 for i, r in enumerate(runtime_sorted)}
    rank_error = {(r["M"], r["B"]): i + 1 for i, r in enumerate(err_sorted)}

    # Pareto frontier: not dominated in both runtime and total error.
    for r in parsed:
        dominated = False
        for q in parsed:
            if q is r:
                continue
            no_worse = (
                q["mean_runtime_sec"] <= r["mean_runtime_sec"]
                and q["mean_abs_err_total_m"] <= r["mean_abs_err_total_m"]
            )
            strictly_better_one = (
                q["mean_runtime_sec"] < r["mean_runtime_sec"]
                or q["mean_abs_err_total_m"] < r["mean_abs_err_total_m"]
            )
            if no_worse and strictly_better_one:
                dominated = True
                break
        r["pareto_optimal"] = (not dominated)
        r["rank_runtime"] = rank_runtime[(r["M"], r["B"])]
        r["rank_error"] = rank_error[(r["M"], r["B"])]
        r["rank_sum"] = r["rank_runtime"] + r["rank_error"]

    parsed = sorted(
        parsed,
        key=lambda x: (
            0 if x["pareto_optimal"] else 1,
            x["rank_sum"],
            x["mean_runtime_sec"],
            x["mean_abs_err_total_m"],
        ),
    )

    _write_csv(
        [
            {
                "M": r["M"],
                "B": r["B"],
                "mean_runtime_sec": round(r["mean_runtime_sec"], 4),
                "mean_abs_err_height_m": round(r["mean_abs_err_height_m"], 5),
                "mean_abs_err_length_m": round(r["mean_abs_err_length_m"], 5),
                "mean_abs_err_width_m": round(r["mean_abs_err_width_m"], 5),
                "mean_abs_err_total_m": round(r["mean_abs_err_total_m"], 5),
                "rank_runtime": r["rank_runtime"],
                "rank_error": r["rank_error"],
                "rank_sum": r["rank_sum"],
                "pareto_optimal": int(r["pareto_optimal"]),
            }
            for r in parsed
        ],
        out_preemptive_rank_csv,
        fieldnames=[
            "M",
            "B",
            "mean_runtime_sec",
            "mean_abs_err_height_m",
            "mean_abs_err_length_m",
            "mean_abs_err_width_m",
            "mean_abs_err_total_m",
            "rank_runtime",
            "rank_error",
            "rank_sum",
            "pareto_optimal",
        ],
    )

    return out_mlesac_compare_csv, out_preemptive_rank_csv


def run_full_parameter_suite(data_dir="data", max_iters=1000):
    m_sweep, m_summary = sweep_mlesac_eps(
        data_dir=data_dir,
        eps_values=FULL_EPS_VALUES,
        max_iters=max_iters,
    )
    p_sweep, p_summary = sweep_preemptive_mb(
        data_dir=data_dir,
        eps=FULL_PREEMPTIVE_EPS,
        M_values=FULL_M_VALUES,
        B_values=FULL_B_VALUES,
        max_iters=max_iters,
    )
    cmp_m, cmp_p = compare_extension_results(
        mlesac_summary_csv=m_summary,
        preemptive_summary_csv=p_summary,
    )
    return {
        "mlesac_sweep_csv": m_sweep,
        "mlesac_summary_csv": m_summary,
        "preemptive_sweep_csv": p_sweep,
        "preemptive_summary_csv": p_summary,
        "mlesac_compare_csv": cmp_m,
        "preemptive_rank_csv": cmp_p,
    }


def visualize_preemptive_m_choices(
    mat_path,
    eps=0.005,
    M_choices=(64, 256, 1024),
    B=200,
    max_iters=1000,
    out_path="outputs/preemptive_M_comparison_floor.png",
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualization. Install it to use --mode viz."
        ) from e

    _, D, _, _ = load_example(mat_path)

    fig, axes = plt.subplots(1, len(M_choices), figsize=(5 * len(M_choices), 4))
    if len(M_choices) == 1:
        axes = [axes]

    for ax, M in zip(axes, M_choices):
        r = run_plane_pipeline(
            mat_path=mat_path,
            method="preemptive",
            eps=eps,
            max_iters=max_iters,
            M=M,
            B=B,
            score_mode="msac",
        )
        floor = r["floor_mask"]

        bg = np.asarray(D, dtype=np.float32)
        bg = np.nan_to_num(bg)
        bg = (bg - np.percentile(bg, 2)) / (np.percentile(bg, 98) - np.percentile(bg, 2) + 1e-6)
        bg = np.clip(bg, 0, 1)

        overlay = np.dstack([bg, bg, bg])
        overlay[floor] = [0.2, 0.9, 0.2]

        ax.imshow(overlay)
        ax.set_title(f"M={M}, B={B}\n{r['runtime_sec']:.3f}s")
        ax.axis("off")

    ensure_dir(os.path.dirname(out_path) or ".")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path

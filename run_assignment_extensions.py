import argparse
import glob
import os

from src.assignment_extensions import (
    FULL_B_VALUES,
    FULL_EPS_VALUES,
    FULL_M_VALUES,
    FULL_PREEMPTIVE_EPS,
    compare_extension_results,
    run_full_parameter_suite,
    sweep_mlesac_eps,
    sweep_preemptive_mb,
    visualize_preemptive_m_choices,
)


def _parse_floats(text):
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def _parse_ints(text):
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def _default_mat(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "example*kinect.mat")))
    if not files:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    return files[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CV assignment extension experiments")
    parser.add_argument(
        "--mode",
        choices=["all", "full", "mlesac", "preemptive", "viz"],
        default="all",
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--eps_values", type=str, default="0.003,0.005,0.007,0.01")
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--m_values", type=str, default="64,256,1024")
    parser.add_argument("--b_values", type=str, default="100,200,400")
    parser.add_argument("--viz_m", type=str, default="64,256,1024")
    parser.add_argument("--viz_b", type=int, default=200)
    parser.add_argument("--mat", type=str, default="")
    parser.add_argument("--max_iters", type=int, default=1000)
    args = parser.parse_args()

    mlesac_summary_csv = ""
    preemptive_summary_csv = ""

    if args.mode == "full":
        results = run_full_parameter_suite(data_dir=args.data_dir, max_iters=args.max_iters)
        print(f"[MLESAC] sweep:   {results['mlesac_sweep_csv']}")
        print(f"[MLESAC] summary: {results['mlesac_summary_csv']}")
        print(f"[PREEMPTIVE] sweep:   {results['preemptive_sweep_csv']}")
        print(f"[PREEMPTIVE] summary: {results['preemptive_summary_csv']}")
        print(f"[COMPARE] mlesac_vs_ransac: {results['mlesac_compare_csv']}")
        print(f"[COMPARE] preemptive_rank: {results['preemptive_rank_csv']}")
        mlesac_summary_csv = results["mlesac_summary_csv"]
        preemptive_summary_csv = results["preemptive_summary_csv"]

    if args.mode in ("all", "mlesac"):
        eps_values = _parse_floats(args.eps_values)
        sweep_csv, summary_csv = sweep_mlesac_eps(
            data_dir=args.data_dir,
            eps_values=eps_values,
            max_iters=args.max_iters,
        )
        print(f"[MLESAC] sweep:   {sweep_csv}")
        print(f"[MLESAC] summary: {summary_csv}")
        mlesac_summary_csv = summary_csv

    if args.mode in ("all", "preemptive"):
        m_values = _parse_ints(args.m_values)
        b_values = _parse_ints(args.b_values)
        sweep_csv, summary_csv = sweep_preemptive_mb(
            data_dir=args.data_dir,
            eps=args.epsilon,
            M_values=m_values,
            B_values=b_values,
            max_iters=args.max_iters,
        )
        print(f"[PREEMPTIVE] sweep:   {sweep_csv}")
        print(f"[PREEMPTIVE] summary: {summary_csv}")
        preemptive_summary_csv = summary_csv

    if args.mode == "all" and mlesac_summary_csv and preemptive_summary_csv:
        cmp_m, cmp_p = compare_extension_results(
            mlesac_summary_csv=mlesac_summary_csv,
            preemptive_summary_csv=preemptive_summary_csv,
        )
        print(f"[COMPARE] mlesac_vs_ransac: {cmp_m}")
        print(f"[COMPARE] preemptive_rank: {cmp_p}")

    if args.mode in ("all", "full", "viz"):
        mat_path = args.mat if args.mat else _default_mat(args.data_dir)
        viz_m = _parse_ints(args.viz_m) if args.mode != "full" else FULL_M_VALUES
        viz_b = args.viz_b if args.mode != "full" else FULL_B_VALUES[1]
        viz_eps = args.epsilon if args.mode != "full" else FULL_PREEMPTIVE_EPS
        try:
            out_png = visualize_preemptive_m_choices(
                mat_path=mat_path,
                eps=viz_eps,
                M_choices=viz_m,
                B=viz_b,
                max_iters=args.max_iters,
            )
            print(f"[VIZ] {out_png}")
        except ImportError as e:
            print(f"[VIZ] skipped: {e}")

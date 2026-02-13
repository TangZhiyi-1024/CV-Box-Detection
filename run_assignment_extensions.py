import argparse
from pathlib import Path

from src.assignment_extensions import (
    run_mlesac_epsilon_study,
    run_preemptive_study,
    save_preemptive_m_visualization,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run assignment extension experiments (MLESAC + Preemptive RANSAC)."
    )
    parser.add_argument("--data_dir", type=str, default="data", help="Folder containing *.mat files.")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output folder for csv/figures.")
    parser.add_argument(
        "--vis_mat",
        type=str,
        default="data/example1kinect.mat",
        help="One .mat file used for preemptive M visualization.",
    )
    args = parser.parse_args()

    print("[1/3] Running MLESAC epsilon study...")
    mlesac_csv, mlesac_summary = run_mlesac_epsilon_study(args.data_dir, args.out_dir)
    print("  ->", mlesac_csv)
    print("  ->", mlesac_summary)

    print("[2/3] Running Preemptive RANSAC M/B study...")
    preemptive_csv, preemptive_summary = run_preemptive_study(args.data_dir, args.out_dir)
    print("  ->", preemptive_csv)
    print("  ->", preemptive_summary)

    print("[3/3] Saving M-comparison visualization...")
    vis_out = Path(args.out_dir) / "preemptive_M_comparison_floor.png"
    vis_path = save_preemptive_m_visualization(args.vis_mat, str(vis_out))
    print("  ->", vis_path)

    print("Done.")


if __name__ == "__main__":
    main()


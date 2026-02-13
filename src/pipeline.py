from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import *
from .io_utils import load_example, ensure_dir
from .viz import show_image, show_mask, scatter_pc, visualize_box_scene
from .ransac import ransac_plane, preemptive_ransac_plane
from .masks import clean_mask
from .components import largest_component
from .box_measure import box_height, box_length_width
from .geometry import point_plane_signed_distance


RANSAC_VARIANT = "ransac"
# 可选："ransac" | "mlesac" | "preemptive"

PREEMPTIVE_M = 256
PREEMPTIVE_B = 200
def run(mat_path: str):
    print("[DBG] VARIANT:", RANSAC_VARIANT, "M:", PREEMPTIVE_M, "B:", PREEMPTIVE_B, "TH:", RANSAC_THRESH)
    ensure_dir(OUT_DIR)
    A, D, PC, valid = load_example(mat_path)

    # 1) 可视化
    show_image(A, "Amplitude")
    show_image(D, "Distance")
    scatter_pc(PC, valid, step=PC_SCATTER_SUBSAMPLE, title="Point Cloud")

    # 2) 地面 RANSAC
    # n_floor, d_floor, floor_mask = ransac_plane(
    #     PC, valid, thresh=RANSAC_THRESH, max_iters=RANSAC_MAX_ITERS
    # )
    if RANSAC_VARIANT == "preemptive":
        n_floor, d_floor, floor_mask = preemptive_ransac_plane(
            PC, valid,
            thresh=RANSAC_THRESH,
            M=PREEMPTIVE_M,
            B=PREEMPTIVE_B
        )
    else:
        n_floor, d_floor, floor_mask = ransac_plane(
            PC, valid,
            thresh=RANSAC_THRESH,
            max_iters=RANSAC_MAX_ITERS,
            mode=RANSAC_VARIANT  # "ransac" or "mlesac"
        )
    floor_mask = clean_mask(floor_mask, MORPH_OPEN_SIZE, MORPH_CLOSE_SIZE)
    show_mask(floor_mask, "Floor mask (cleaned)")

    # 3) 非地面点（包含盒子等）
    not_floor = valid & (~floor_mask)

    # 4) 盒子顶面 RANSAC（在非地面点里找主平面；若背景多，可能需要迭代或取第二大平面）
    dist_to_floor = np.abs(point_plane_signed_distance(PC, n_floor, d_floor))
    not_floor = not_floor & (dist_to_floor > 0.05)  # 0.05米 = 5厘米
    # n_top, d_top, top_mask_candidates = ransac_plane(
    #     PC, not_floor, thresh=RANSAC_THRESH, max_iters=RANSAC_MAX_ITERS
    # )
    if RANSAC_VARIANT == "preemptive":
        n_top, d_top, top_mask_candidates = preemptive_ransac_plane(
            PC, not_floor,
            thresh=RANSAC_THRESH,
            M=PREEMPTIVE_M,
            B=PREEMPTIVE_B
        )
    else:
        n_top, d_top, top_mask_candidates = ransac_plane(
            PC, not_floor,
            thresh=RANSAC_THRESH,
            max_iters=RANSAC_MAX_ITERS,
            mode=RANSAC_VARIANT  # "ransac" or "mlesac"
        )
    top_mask_candidates = clean_mask(top_mask_candidates, 1, 3)


    # 5) 取最大连通区域作为盒子顶面
    box_top_mask, _ = largest_component(top_mask_candidates)
    show_mask(box_top_mask, "Box top (largest component)")

    # 6) 尺寸
    H = box_height(n_floor, d_floor, n_top, d_top)
    L, W = box_length_width(PC, box_top_mask)

    print("[DBG] floor_inliers:", floor_mask.sum())
    print("[DBG] top_inliers:", top_mask_candidates.sum())
    print("[DBG] box:", H, L, W)
    # 7)最终可视化
    visualize_box_scene(D, floor_mask, box_top_mask, valid_mask=valid)

    print(f"[RESULT] Height: {H:.3f}  Length: {L:.3f}  Width: {W:.3f}")
    plt.savefig(
        f"outputs/debug_{Path(mat_path).stem}.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

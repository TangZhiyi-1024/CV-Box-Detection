import numpy as np
from .geometry import plane_plane_distance

# 计算箱子的垂直高度
def box_height(n_floor, d_floor, n_top, d_top):
    return plane_plane_distance(n_floor, d_floor, n_top, d_top)

# 测量长宽
def box_length_width(PC, box_mask):
    pts = PC[box_mask]  # (N,3)点集 1顶0地背景
    from .corners import length_width_from_top_points
    L, W = length_width_from_top_points(pts, axis="auto")   # 计算箱子顶面在二维平面上的精准长和宽
    return L, W

import numpy as np
from scipy.ndimage import label

# 在给定的二值掩码 (Binary Mask) 中，只保留面积最大的那一个连通区域。
def largest_component(mask):
    lbl, n = label(mask)  # 标记所有连通区域
    if n == 0:
        return mask, 0
    counts = np.bincount(lbl.ravel())       # 连通区域的像素总数（一维）
    counts[0] = 0  # 背景忽略
    k = counts.argmax()
    return (lbl == k), k        # 返回经过提纯的、只包含最大物体的掩码

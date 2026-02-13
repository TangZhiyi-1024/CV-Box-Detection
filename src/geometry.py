import numpy as np

# 平面用 (n, d) 表示，其中 n 为单位法向量，满足 n·x + d = 0 和 距离原点的偏移量 d
# 空间中任意三个不共线的点，计算出唯一确定的平面方程参数 the unique plane equation parameters
def plane_from_3pts(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1        # 三个点转化为Coplanar Vectors
    n = np.cross(v1, v2)        # Normal Vector法向量计算
    norm = np.linalg.norm(n)        #  计算长度接近于0几乎Collinear Normalization缩放为1
    if norm < 1e-9:
        return None, None
    n = n / norm
    d = -np.dot(n, p1)      #计算Offset偏移量
    return n, d

# 计算Point-to-Plane Signed Distance
def point_plane_signed_distance(pts, n, d):
    # pts: (..., 3)
    return (pts @ n + d)

def plane_plane_distance(n1, d1, n2, d2):
    # 两平面法向量方向一致时，平行距离 = |d2 - d1|
    # 若方向相反，取反一个法向量保证一致
    if np.dot(n1, n2) < 0: # direction
        n2, d2 = -n2, -d2
    # 夹角不近似 0 时要先校准，这里假设已找到同向主平面
    return abs(d2 - d1)

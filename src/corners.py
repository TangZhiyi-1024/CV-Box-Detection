import numpy as np

def pca_oriented_bbox(pts_2d):
    """
    用 PCA 求 2D 顶面点（投影）最小外接矩形近似：
    1) PCA 得主方向 u,v
    2) 将点投影到 (u,v)，取 min/max 得到长宽
    返回: center(2,), axes(2,2), extents(2,)  # 轴向单位向量与半长宽
    Dimensions Orientation center
    """
    # 去均值coordinate system
    mu = pts_2d.mean(axis=0)
    X = pts_2d - mu
    # 协方差 & 特征
    C = np.cov(X.T)         # geometric shape and spread of the data distribution
    w, V = np.linalg.eigh(C)       # 列为特征向量（升序）Ascending Eigenvalues Eigenvectors
    # 主轴从大到小
    order = np.argsort(w)[::-1]  #最大到最小的索引列表
    V = V[:, order]                # [u, v]
    U = V[:, 0:2]                   #1箱子的长边 2箱子宽边

    # 到 (u,v) 坐标Coordinate Transformation Axis-Aligned
    Y = X @ U
    mins = Y.min(axis=0) #Boundary Calculation
    maxs = Y.max(axis=0)
    extents = (maxs - mins) / 2.0 #半长宽
    center_uv = (maxs + mins) / 2.0 # 局部中心点
    center = mu + center_uv @ U.T #转回原来的歪角度(Inverse Transformation)+搬回世界坐标系原本的位置Global Coordinate System
    return center, U, extents

def length_width_from_top_points(pts_top_xyz, axis="auto"):
    """
    以 (x,y) 平面投影求长宽（近似）。若相机坐标更适合用 (x,z) 或 (y,z)，可切换。
    计算箱子顶面在二维平面上的精准长和宽
    """
    if axis == "auto":
        # 默认投影到 x-yProjection
        pts_2d = pts_top_xyz[:, :2] #3D 的点云Top-down View
    elif axis == "xz":
        pts_2d = pts_top_xyz[:, [0,2]]
    elif axis == "yz":
        pts_2d = pts_top_xyz[:, [1,2]]
    else:
        raise ValueError("axis must be auto/xz/yz/xy")

    center, axes, extents = pca_oriented_bbox(pts_2d)
    length, width = 2*extents[0], 2*extents[1]
    return length, width

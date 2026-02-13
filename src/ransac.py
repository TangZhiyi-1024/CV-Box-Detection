import numpy as np
from .geometry import plane_from_3pts, point_plane_signed_distance


# def ransac_plane(PC, valid_mask, thresh=0.01, max_iters=1000, rng=None):
def ransac_plane(PC, valid_mask, thresh=0.01, max_iters=1000,
                 rng=None, mode="ransac", gamma=None):
    """
    在点云上拟合主平面
    返回: n, d, inlier_mask
    """
    if rng is None:
        rng = np.random.default_rng(42)

    H, W, _ = PC.shape
    pts = PC.reshape(-1, 3)
    vm = valid_mask.reshape(-1)
    idxs = np.flatnonzero(vm)
    if len(idxs) < 3:
        raise ValueError("有效点不足以拟合平面")

    best_inliers = None
    if mode == "ransac":
        best_score = -1  # 越大越好
    else:  # mlesac
        best_score = np.inf  # 越小越好
    best_model = (None, None)

    for _ in range(max_iters):
        # 随机抽 3 点
        sample = rng.choice(idxs, size=3, replace=False)
        p1, p2, p3 = pts[sample]
        n, d = plane_from_3pts(p1, p2, p3)
        if n is None:
            continue

        # 距离阈值内的内点
        dist = np.abs(point_plane_signed_distance(pts, n, d))

        if mode == "ransac":
            inliers = (dist < thresh) & vm
            score = inliers.sum()  # 越大越好
            better = score > best_score

        elif mode == "mlesac":
            if gamma is None:
                gamma = 2.0 * thresh

            dv = dist[vm]  # 只对有效点算
            score = np.where(dv < thresh, dv, gamma).sum()  # 越小越好
            inliers = (dist < thresh) & vm  # 仍然用阈值定义内点
            better = score < best_score
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if better:
            best_score = score
            best_inliers = inliers
            best_model = (n, d)
        # dist = np.abs(point_plane_signed_distance(pts, n, d))
        # inliers = (dist < thresh) & vm
        # count = inliers.sum()
        # if count > best_count:
        #     best_count = count
        #     best_inliers = inliers
        #     best_model = (n, d)

        # 如果几乎所有有效点都是内点可提前结束
        if mode == "ransac" and score > 0.9 * vm.sum():
            break

    n, d = best_model
    inlier_mask = best_inliers.reshape(H, W)
    return n, d, inlier_mask

def preemptive_ransac_plane(PC, valid_mask, thresh=0.01, M=256, B=200, rng=None, gamma=None):
    if rng is None:
        rng = np.random.default_rng(42)
    if gamma is None:
        gamma = 2.0 * thresh

    H, W, _ = PC.shape
    pts = PC.reshape(-1, 3)
    vm = valid_mask.reshape(-1)
    idxs = np.flatnonzero(vm)
    if len(idxs) < 3:
        raise ValueError("有效点不足以拟合平面")

    # 1) 生成 M 个 hypothesis
    models = []
    for _ in range(M):
        sample = rng.choice(idxs, size=3, replace=False)
        p1, p2, p3 = pts[sample]
        n, d = plane_from_3pts(p1, p2, p3)
        if n is not None:
            models.append((n, d))
    if len(models) == 0:
        raise ValueError("未生成有效平面假设")

    # 打乱用于评估的点顺序
    eval_ids = rng.permutation(idxs)

    # 2) 分批评估 + 淘汰
    costs = np.zeros(len(models), dtype=float)

    for t in range(0, len(eval_ids), B):
        batch = eval_ids[t:t+B]
        batch_pts = pts[batch]

        # 更新每个模型在这一批点上的 cost（用 MLESAC 的 p(d)）
        for j, (n, d) in enumerate(models):
            dv = np.abs(point_plane_signed_distance(batch_pts, n, d))
            costs[j] += np.where(dv < thresh, dv, gamma).sum()

        # 每处理完一批就 preempt
        k = int(np.floor(len(models) * (2 ** (-np.floor((t + B) / B)))))  # 你也可以直接每轮减半
        k = max(1, min(k, len(models)))

        # 保留 cost 最小的 k 个
        keep = np.argsort(costs)[:k]
        models = [models[i] for i in keep]
        costs = costs[keep]

        if len(models) == 1:
            break

    # 输出最终模型，并算 inlier mask
    n, d = models[0]
    dist_all = np.abs(point_plane_signed_distance(pts, n, d))
    inliers = (dist_all < thresh) & vm
    return n, d, inliers.reshape(H, W)

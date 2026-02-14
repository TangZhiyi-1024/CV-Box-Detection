import numpy as np

from .geometry import plane_from_3pts, point_plane_signed_distance


def _prepare_points(PC, valid_mask):
    H, W, _ = PC.shape
    pts_all = PC.reshape(-1, 3)
    vm = valid_mask.reshape(-1).astype(bool)
    idxs = np.flatnonzero(vm)
    if idxs.size < 3:
        raise ValueError("Not enough valid points to estimate a plane")
    pts_valid = pts_all[idxs]
    return H, W, pts_all, vm, idxs, pts_valid


def _model_from_three_points(pts_all, sample_indices):
    p1, p2, p3 = pts_all[sample_indices]
    return plane_from_3pts(p1, p2, p3)


def _evaluate_distances(pts, n, d):
    return np.abs(point_plane_signed_distance(pts, n, d))


def ransac_plane(PC, valid_mask, thresh=0.01, max_iters=1000, rng=None):
    """
    Standard RANSAC plane fitting.
    Returns: n, d, inlier_mask
    """
    if rng is None:
        rng = np.random.default_rng(42)

    H, W, pts_all, vm, idxs, _ = _prepare_points(PC, valid_mask)

    best_inliers = None
    best_count = -1
    best_model = (None, None)

    for _ in range(max_iters):
        sample = rng.choice(idxs, size=3, replace=False)
        n, d = _model_from_three_points(pts_all, sample)
        if n is None:
            continue

        dist = _evaluate_distances(pts_all, n, d)
        inliers = (dist < thresh) & vm
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_model = (n, d)

        if count > 0.9 * int(vm.sum()):
            break

    if best_model[0] is None:
        raise RuntimeError("RANSAC failed to produce a valid plane model")

    n, d = best_model
    inlier_mask = best_inliers.reshape(H, W)
    return n, d, inlier_mask


def mlesac_plane(PC, valid_mask, eps=0.01, gamma=None, max_iters=1000, rng=None):
    """
    MLESAC plane fitting with cost:
      C = sum(dist_i if dist_i < eps else gamma)
    Returns: n, d, inlier_mask
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if gamma is None:
        gamma = 1.5 * eps
    if gamma <= eps:
        raise ValueError("gamma must be strictly larger than eps")

    H, W, pts_all, vm, idxs, _ = _prepare_points(PC, valid_mask)
    vm_count = int(vm.sum())

    best_cost = np.inf
    best_model = (None, None)
    best_inliers = None

    for _ in range(max_iters):
        sample = rng.choice(idxs, size=3, replace=False)
        n, d = _model_from_three_points(pts_all, sample)
        if n is None:
            continue

        dist = _evaluate_distances(pts_all, n, d)
        dist_valid = dist[vm]
        cost = np.where(dist_valid < eps, dist_valid, gamma).sum()

        if cost < best_cost:
            best_cost = cost
            best_model = (n, d)
            best_inliers = (dist < eps) & vm

            # Early stop: near-perfect fit where almost all valid points are inliers.
            if int(best_inliers.sum()) > 0.98 * vm_count:
                break

    if best_model[0] is None:
        raise RuntimeError("MLESAC failed to produce a valid plane model")

    n, d = best_model
    return n, d, best_inliers.reshape(H, W)


def _build_hypotheses(pts_all, idxs, M, rng, max_attempt_factor=25):
    normals = []
    offsets = []
    attempts = 0
    max_attempts = max(200, M * max_attempt_factor)

    while len(normals) < M and attempts < max_attempts:
        attempts += 1
        sample = rng.choice(idxs, size=3, replace=False)
        n, d = _model_from_three_points(pts_all, sample)
        if n is None:
            continue
        normals.append(n)
        offsets.append(d)

    if not normals:
        raise RuntimeError("Failed to initialize any valid hypothesis")

    return np.asarray(normals), np.asarray(offsets)


def _batch_cost(dist_abs, eps, score_mode="msac", gamma=None):
    if score_mode == "ransac":
        return (dist_abs >= eps).astype(np.float64)
    if score_mode == "msac":
        return np.minimum(dist_abs, eps)
    if score_mode == "mlesac":
        if gamma is None:
            gamma = 1.5 * eps
        return np.where(dist_abs < eps, dist_abs, gamma)
    raise ValueError("score_mode must be one of {'ransac', 'msac', 'mlesac'}")


def preemptive_ransac_plane(
    PC,
    valid_mask,
    thresh=0.01,
    M=256,
    B=200,
    score_mode="msac",
    gamma=None,
    rng=None,
):
    """
    Preemptive RANSAC for plane fitting.

    1) Sample M hypotheses once.
    2) Evaluate hypotheses on shuffled points in batches of size B.
    3) After each batch, keep only top f(i) hypotheses:
       f(i) = floor(M * 2^(-floor(i / B)))
    4) Stop when one model remains or all points are consumed.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if M < 1 or B < 1:
        raise ValueError("M and B must be positive integers")

    H, W, pts_all, vm, idxs, pts_valid = _prepare_points(PC, valid_mask)
    n_pts = pts_valid.shape[0]

    normals, offsets = _build_hypotheses(pts_all, idxs, M=M, rng=rng)
    n_models = normals.shape[0]
    costs = np.zeros(n_models, dtype=np.float64)
    active = np.arange(n_models, dtype=np.int64)

    order = rng.permutation(n_pts)
    pts_eval = pts_valid[order]

    processed = 0
    while processed < n_pts and active.size > 1:
        end = min(processed + B, n_pts)
        batch = pts_eval[processed:end]  # (b, 3)

        n_active = normals[active]  # (k, 3)
        d_active = offsets[active]  # (k,)
        # (b, k): absolute point-to-plane distance
        dist_abs = np.abs(batch @ n_active.T + d_active)
        costs[active] += _batch_cost(
            dist_abs=dist_abs, eps=thresh, score_mode=score_mode, gamma=gamma
        ).sum(axis=0)

        processed = end
        rank = np.argsort(costs[active])
        active = active[rank]

        stage = processed // B
        keep = max(1, int(np.floor(M * (2.0 ** (-stage)))))
        keep = min(keep, active.size)
        active = active[:keep]

    # Final selection over surviving hypotheses using all valid points.
    n_active = normals[active]
    d_active = offsets[active]
    dist_full = np.abs(pts_valid @ n_active.T + d_active)
    full_costs = _batch_cost(
        dist_abs=dist_full, eps=thresh, score_mode=score_mode, gamma=gamma
    ).sum(axis=0)
    best_idx = active[int(np.argmin(full_costs))]

    n = normals[best_idx]
    d = offsets[best_idx]
    dist_all = _evaluate_distances(pts_all, n, d)
    inlier_mask = ((dist_all < thresh) & vm).reshape(H, W)
    return n, d, inlier_mask

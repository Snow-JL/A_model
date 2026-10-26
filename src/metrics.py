"""评价指标：分级一致性与分布距离。

分级指标（Acc / ±1 / QWK / MAE）在 utils 中；本模块补充分布层面与集合层面。
"""
from __future__ import annotations

import numpy as np

from .utils import accuracy, mae, quadratic_weighted_kappa, tolerance_accuracy


# ------------------------------------------------------------------ 集合
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def topk_jaccard_matrix(sets: list[set]) -> np.ndarray:
    m = len(sets)
    J = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            J[i, j] = J[j, i] = jaccard(sets[i], sets[j])
    return J


# ------------------------------------------------------------------ 分布
def pairwise_sq_dist(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)


def mmd2(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """高斯核 MMD² 的有偏估计。"""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    Z = np.vstack([X, Y])
    if gamma is None:
        d2 = pairwise_sq_dist(Z, Z)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / med
    kxx = np.exp(-gamma * pairwise_sq_dist(X, X)).mean()
    kyy = np.exp(-gamma * pairwise_sq_dist(Y, Y)).mean()
    kxy = np.exp(-gamma * pairwise_sq_dist(X, Y)).mean()
    return float(kxx + kyy - 2 * kxy)


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    dxy = np.sqrt(pairwise_sq_dist(X, Y)).mean()
    dxx = np.sqrt(pairwise_sq_dist(X, X)).mean()
    dyy = np.sqrt(pairwise_sq_dist(Y, Y)).mean()
    return float(2 * dxy - dxx - dyy)


# ------------------------------------------------------------------ 覆盖
def knn_coverage(gen: np.ndarray, real: np.ndarray, k: int = 5) -> float:
    """生成样本落在真实样本 k 近邻球内的比例（多样性代理）。"""
    gen = np.asarray(gen, float)
    real = np.asarray(real, float)
    d = np.sqrt(pairwise_sq_dist(gen, real))
    radii = np.sort(d, axis=1)[:, :k].max(axis=1)
    d_real = np.sqrt(pairwise_sq_dist(real, real))
    np.fill_diagonal(d_real, np.inf)
    r_real = np.sort(d_real, axis=1)[:, :k].max(axis=1)
    return float(np.mean(radii <= np.quantile(r_real, 0.95)))


def memorization_rate(gen: np.ndarray, real: np.ndarray, tol: float = 1e-6) -> float:
    """生成样本与某个真实样本几乎重合的比例（越低越好）。"""
    d = np.sqrt(pairwise_sq_dist(gen, real))
    return float(np.mean(d.min(axis=1) <= tol))


# ------------------------------------------------------------------ 分级报告
def grade_report(y_true: np.ndarray, y_pred: np.ndarray, n_level: int = 7) -> dict:
    """单维度或整体（展平）的分级指标。"""
    return {
        "acc": accuracy(y_true, y_pred),
        "acc_pm1": tolerance_accuracy(y_true, y_pred, tol=1),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, n_level=n_level),
        "mae": mae(y_true, y_pred),
    }


def grade_report_per_dim(G_true: np.ndarray, G_pred: np.ndarray,
                         columns: list[str], n_level: int = 7):
    """逐维度报告，返回 DataFrame。"""
    import pandas as pd

    rows = []
    for j, c in enumerate(columns):
        r = grade_report(G_true[:, j], G_pred[:, j], n_level)
        rows.append({"dimension": c, **{k: round(v, 4) for k, v in r.items()}})
    return pd.DataFrame(rows)


def summarize_per_dim(df, name: str = "model") -> dict:
    """把逐维度报告汇总为均值与中位数。"""
    return {
        "model": name,
        "acc": round(float(df["acc"].mean()), 4),
        "acc_pm1": round(float(df["acc_pm1"].mean()), 4),
        "qwk": round(float(df["qwk"].mean()), 4),
        "mae": round(float(df["mae"].mean()), 4),
        "acc_median": round(float(df["acc"].median()), 4),
        "qwk_median": round(float(df["qwk"].median()), 4),
    }

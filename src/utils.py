"""公共数值工具：秩变换、逆正态、Spearman、二次加权 Kappa、稳健统计。

零 scipy 依赖，保证在任意 numpy 环境可运行。
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ 秩
def rankdata(a: np.ndarray) -> np.ndarray:
    """平均秩（处理并结）。"""
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(a.size, dtype=int)
    inv[sorter] = np.arange(a.size)
    sorted_a = a[sorter]
    obs = np.concatenate(([True], sorted_a[1:] != sorted_a[:-1]))
    dense = obs.cumsum()[inv]
    count = np.concatenate((np.nonzero(obs)[0], [a.size]))
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关。"""
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / den) if den > 0 else float("nan")


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson 相关（忽略 NaN 对）。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x, y = x[m], y[m]
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / den) if den > 0 else float("nan")


# ------------------------------------------------------------------ 逆正态
def ndtri(p: np.ndarray) -> np.ndarray:
    """标准正态分位函数，Acklam 有理逼近，绝对误差 < 1.15e-9。"""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    q = p - 0.5
    out = np.empty_like(p)
    lo = p < 0.02425
    hi = p > 1 - 0.02425
    mid = ~(lo | hi)

    if np.any(mid):
        r = q[mid] * q[mid]
        out[mid] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q[mid] /
                    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    if np.any(lo):
        r = np.sqrt(-2.0 * np.log(p[lo]))
        out[lo] = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
                  ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
    if np.any(hi):
        r = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
        out[hi] = -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
                   ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
    return out


# ------------------------------------------------------------------ 变换
def safe_log10(x: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log10(np.clip(x, floor, None))


def sigmoid(y: np.ndarray) -> np.ndarray:
    """数值稳定的 logistic。"""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    pos = y >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-y[pos]))
    ey = np.exp(y[~pos])
    out[~pos] = ey / (1.0 + ey)
    return out


def quantile_normalize(X: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """逐列秩映射到标准正态。"""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    out = np.empty_like(X)
    for j in range(p):
        col = X[:, j]
        v = np.isfinite(col)
        if v.sum() < 2:
            out[:, j] = 0.0
            continue
        r = rankdata(col[v])
        out[v, j] = ndtri((r - 0.5) / v.sum())
    return np.clip(out, -clip, clip)


# ------------------------------------------------------------------ 稳健统计
def mad(x: np.ndarray) -> float:
    """中位绝对偏差，已乘 1.4826 换算到标准差尺度。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ------------------------------------------------------------------ 评价
def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray,
                             n_level: int = 7) -> float:
    """二次加权 Kappa（有序分级的标准一致性指标）。"""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    m = (y_true >= 0) & (y_true < n_level) & (y_pred >= 0) & (y_pred < n_level)
    y_true, y_pred = y_true[m], y_pred[m]
    if y_true.size == 0:
        return float("nan")

    O = np.zeros((n_level, n_level))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    w = (np.arange(n_level)[:, None] - np.arange(n_level)[None, :]) ** 2 / (n_level - 1) ** 2
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / O.sum()
    den = (w * E).sum()
    return float(1.0 - (w * O).sum() / den) if den > 0 else float("nan")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    return float(np.mean(y_true == y_pred))


def tolerance_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tol: int = 1) -> float:
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def entropy_of_distribution(P: np.ndarray) -> np.ndarray:
    """按行计算香农熵，P 为 (n, K) 概率矩阵。"""
    P = np.clip(np.asarray(P, float), 1e-12, None)
    return -np.sum(P * np.log(P), axis=1)

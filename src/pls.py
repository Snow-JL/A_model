"""偏最小二乘（PLS / NIPALS），自实现，不依赖 scikit-learn。

用途：把 231 维化合物压到 q 个潜变量，同时利用 X 与 Y 的协方差。
理由：p ≈ n 且化合物强共线（最大 |r| = 0.966），PLS 提取的是协方差方向而非
单变量方向，共线不影响成分提取。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pls_nipals(
    X: np.ndarray,
    Y: np.ndarray,
    n_comp: int,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> dict:
    """NIPALS 算法。

    返回 W(成分权重)、T(得分)、P(X 负载)、Q(Y 负载)、B(回归系数)、
    x_mean/y_mean（预测时需要）。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, p = X.shape
    q = Y.shape[1]

    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    X0, Y0 = X - x_mean, Y - y_mean

    W = np.zeros((p, n_comp))
    T = np.zeros((n, n_comp))
    P = np.zeros((p, n_comp))
    Q = np.zeros((q, n_comp))

    for a in range(n_comp):
        u = Y0[:, [0]].copy()
        w = np.zeros((p, 1))
        t = np.zeros((n, 1))
        for _ in range(max_iter):
            w = X0.T @ u
            nw = np.linalg.norm(w)
            if nw < 1e-12:
                break
            w /= nw
            t_new = X0 @ w
            if np.linalg.norm(t_new - t) < tol * (np.linalg.norm(t_new) + 1e-12):
                t = t_new
                break
            t = t_new
            q_vec = Y0.T @ t / (t.T @ t + 1e-12)
            u_new = Y0 @ q_vec
            if np.linalg.norm(u_new - u) < tol * (np.linalg.norm(u_new) + 1e-12):
                u = u_new
                break
            u = u_new

        denom = t.T @ t + 1e-12
        p_load = X0.T @ t / denom
        q_load = Y0.T @ t / denom

        W[:, a] = w.ravel()
        T[:, a] = t.ravel()
        P[:, a] = p_load.ravel()
        Q[:, a] = q_load.ravel()

        X0 = X0 - t @ p_load.T
        Y0 = Y0 - t @ q_load.T

        if np.linalg.norm(X0) < 1e-10:
            W, T, P, Q = W[:, :a + 1], T[:, :a + 1], P[:, :a + 1], Q[:, :a + 1]
            break

    # 回归系数 B = W (P^T W)^{-1} Q^T
    PtW = P.T @ W
    B = W @ np.linalg.pinv(PtW) @ Q.T
    return {"W": W, "T": T, "P": P, "Q": Q, "B": B,
            "x_mean": x_mean, "y_mean": y_mean, "n_comp": W.shape[1]}


def pls_predict(model: dict, X: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=float) - model["x_mean"]) @ model["B"] + model["y_mean"]


def pls_scores(model: dict, X: np.ndarray) -> np.ndarray:
    """返回潜变量得分矩阵（有序回归的输入）。"""
    return (np.asarray(X, dtype=float) - model["x_mean"]) @ model["W"]


def select_ncomp(
    X: np.ndarray,
    Y: np.ndarray,
    comp_grid: tuple[int, ...] = tuple(range(2, 21)),
    n_fold: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """K 折交叉验证选成分数，指标为各输出维度的平均 RMSE。"""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_fold)

    rows = []
    for a in comp_grid:
        errs = []
        for f in folds:
            tr = np.setdiff1d(idx, f)
            model = pls_nipals(X[tr], Y[tr], a)
            pred = pls_predict(model, X[f])
            errs.append(np.mean((pred - Y[f]) ** 2))
        rows.append({"n_comp": a, "rmse": float(np.sqrt(np.mean(errs)))})
    return pd.DataFrame(rows)

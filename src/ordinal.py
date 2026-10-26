"""累积链接有序回归（proportional-odds / ordinal logit），自实现。

模型
----
    P(y ≥ k | z) = σ(η − τ_{k−1}),   η = βᵀz,   k = 1,…,K
    τ_0 < τ_1 < … < τ_{K−1}

单点概率由相邻累积概率之差给出：

    P(y = 0)   = 1 − S_1
    P(y = k)   = S_k − S_{k+1},   1 ≤ k ≤ K−1
    P(y = K−1) = S_{K−1}

注意：这与"相邻类别连乘"（continuation-ratio）模型不同，后者把 P(y ≥ k) 拆成
条件概率的连乘。此处必须用差分形式，否则似然与梯度都会写错。

参数化保证有序：τ_0 自由，τ_k = τ_{k−1} + exp(δ_k)。梯度解析可写，用 Adam 优化。
"""
from __future__ import annotations

import numpy as np

from .utils import sigmoid


def _unpack(params: np.ndarray, n_level: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """参数布局：[τ_0] + [δ_1…δ_{K−1}] + [β_1…β_q]。"""
    n_cut = n_level - 1
    theta0 = float(params[0])
    delta = params[1:n_cut]
    beta = params[n_cut:]
    inc = np.exp(np.clip(delta, -10, 10))
    tau = np.empty(n_cut)
    tau[0] = theta0
    for k in range(1, n_cut):
        tau[k] = tau[k - 1] + inc[k - 1]
    return theta0, delta, beta, tau


def _cum_survival(eta: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """S[:, k] = P(y ≥ k+1) = σ(η − τ_k)，形状 (n, K−1)。

    用等价式 σ(x) = 0.5·(1+tanh(x/2))：全程无分支、无布尔索引赋值，
    对 (n, K−1) 这种小数组比逐元素 sigmoid 快数倍（旧实现是本流程的头号热点）。
    """
    x = np.subtract(eta[:, None], tau[None, :])
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _nll_grad(params, Z, y, n_level, reg):
    n, q = Z.shape
    n_cut = n_level - 1
    theta0, delta, beta, tau = _unpack(params, n_level)

    eta = Z @ beta
    c = _cum_survival(eta, tau)                    # (n, K−1)

    idx = np.arange(n)
    y_lo = y - 1
    m1 = y >= 1
    m2 = y <= n_cut - 1
    # y = 0 时 lo 取 1（无下界），y = K−1 时 hi 取 0（无上界）
    lo = np.where(m1, c[idx, np.maximum(y_lo, 0)], 1.0)
    hi = np.where(m2, c[idx, np.minimum(y, n_cut - 1)], 0.0)

    P = np.clip(lo - hi, 1e-12, None)
    nll = -np.mean(np.log(P)) + reg * float(beta @ beta)

    invP = 1.0 / P
    # d nll / d eta = −1/n · invP · dP/deta
    dP_deta = m1 * (lo * (1 - lo)) - m2 * (hi * (1 - hi))
    g_eta = -invP * dP_deta / n

    # 累加器用 bincount（C 实现）替代 np.add.at（逐元素 Python 调用）
    w_lo = np.where(m1, invP * lo * (1 - lo), 0.0) / n
    w_hi = np.where(m2, invP * hi * (1 - hi), 0.0) / n
    # j = y−1：dP/dτ = −lo(1−lo)  →  g = +invP·lo(1−lo)/n
    # j = y  ：dP/dτ = +hi(1−hi)  →  g = −invP·hi(1−hi)/n
    g_tau = np.bincount(y_lo[m1], weights=w_lo[m1], minlength=n_cut)
    g_tau -= np.bincount(y[m2], weights=w_hi[m2], minlength=n_cut)

    g_beta = Z.T @ g_eta + 2 * reg * beta

    inc = np.exp(np.clip(delta, -10, 10))
    g_theta0 = g_tau.sum()
    tail = np.cumsum(g_tau[::-1])[::-1]            # tail[k] = g_tau[k:].sum()
    g_delta = tail[1:n_cut] * inc[:max(n_cut - 1, 0)]

    g = np.concatenate(([g_theta0], g_delta, g_beta))
    return nll, g


def fit_ordinal(Z: np.ndarray, y: np.ndarray, n_level: int = 7,
                reg: float = 1e-2, lr: float = 0.05, n_iter: int = 1500,
                seed: int = 0, tol: float = 1e-9, patience: int = 200) -> dict:
    """Adam 优化有序回归。

    `tol`/`patience` 为收敛早停：连续 `patience` 次迭代没有超过
    `tol·max(1,|nll|)` 的改善就停止。实测 nll 在 140–540 次迭代即到机器精度，
    跑满 1500 次纯属空转（问题一要在 800 次拟合上重复，这是主要耗时来源）。
    早停只影响"何时停"，返回的仍是 nll 最小的那组参数。
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=int)
    q = Z.shape[1]
    n_cut = n_level - 1

    params = np.concatenate(([-1.0], np.full(max(n_cut - 1, 0), np.log(0.5)), np.zeros(q)))
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best, best_p = np.inf, params.copy()
    last_gain = 0

    for t in range(1, n_iter + 1):
        nll, g = _nll_grad(params, Z, y, n_level, reg)
        if nll < best:
            if best - nll > tol * max(1.0, abs(nll)):
                last_gain = t
            best, best_p = nll, params.copy()
        if t - last_gain >= patience:
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh = m / (1 - beta1 ** t)
        vh = v / (1 - beta2 ** t)
        params = params - lr * mh / (np.sqrt(vh) + eps)

    theta0, delta, beta, tau = _unpack(best_p, n_level)
    return {"tau": tau, "beta": beta, "nll": float(best), "n_level": n_level,
            "n_iter_used": t}


def predict_proba(model: dict, Z: np.ndarray) -> np.ndarray:
    """返回 (n, K) 的等级概率矩阵。"""
    Z = np.asarray(Z, dtype=float)
    n_level = model["n_level"]
    eta = Z @ model["beta"]
    c = _cum_survival(eta, model["tau"])                   # P(y ≥ k+1)
    n = len(Z)
    S = np.column_stack([np.ones(n), c, np.zeros(n)])      # S[:,k] = P(y ≥ k)
    P = S[:, :-1] - S[:, 1:]
    P = np.clip(P, 1e-12, None)
    return P / P.sum(axis=1, keepdims=True)


def predict(model: dict, Z: np.ndarray) -> np.ndarray:
    return predict_proba(model, Z).argmax(axis=1)


def fit_all(Z: np.ndarray, G: np.ndarray, n_level: int = 7,
            reg: float = 1e-2, seed: int = 0) -> list[dict]:
    """对全部感官维度各拟合一个有序回归。"""
    return [fit_ordinal(Z, G[:, m], n_level=n_level, reg=reg, seed=seed)
            for m in range(G.shape[1])]


def predict_all(models: list[dict], Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (整数等级 n×q, 概率张量 n×q×K)。"""
    n, q = len(Z), len(models)
    n_level = models[0]["n_level"]
    G = np.empty((n, q), dtype=int)
    P = np.empty((n, q, n_level))
    for m, mod in enumerate(models):
        pm = predict_proba(mod, Z)
        P[:, m, :] = pm
        G[:, m] = pm.argmax(axis=1)
    return G, P

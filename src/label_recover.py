"""问题一 标签还原：把 4.csv 的标准化连续值还原为 0–6 等级。

讲解方案的思路与数据裁决
------------------------
原方案用两条锚定：零点锚定（列最小值对应 y=0）与格点间距锚定
（σ = 1/(nδ)，δ 为相邻取值差的最小公共间距）。

实测裁决：50 个维度的格点一致性最高仅 0.076，唯一值中位 229，
若格点成立需要 38 位以上评委，与感官品评实践不符。因此格点锚定不可用，
必须采用方案自带的后备路径。

本实现提供两种还原：
  anchor   —— 零点锚定 + 稳健极值锚定（1% 分位→0，99% 分位→6）
  quantile —— 等频 7 分位切分（无分布假设）

两者都不需要格点结构，且都给出确定的 ŷ ∈ [0,6]。anchor 保留"0 分代表不存在"
的评分卡语义，quantile 保证等级分布均匀，二者的一致率作为还原可靠性的证据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_LEVEL, ZERO_ANCHOR_QUANTILE
from .utils import quadratic_weighted_kappa


def recover_anchor(v: np.ndarray, q_lo: float = ZERO_ANCHOR_QUANTILE,
                   q_hi: float = 1.0 - ZERO_ANCHOR_QUANTILE) -> tuple[np.ndarray, dict]:
    """零点 + 极值锚定：ŷ = 6·(v − q_lo)/(q_hi − q_lo)，截断到 [0,6]。"""
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanquantile(v, q_lo), np.nanquantile(v, q_hi)
    if hi - lo <= 1e-12:
        return np.zeros_like(v), {"lo": lo, "hi": hi, "degenerate": True}
    y = (v - lo) / (hi - lo) * (N_LEVEL - 1)
    return np.clip(y, 0, N_LEVEL - 1), {"lo": float(lo), "hi": float(hi), "degenerate": False}


def recover_quantile(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """等频 7 分位切分，返回连续等级中心与整数等级。"""
    v = np.asarray(v, dtype=float)
    qs = np.nanquantile(v, np.linspace(0, 1, N_LEVEL + 1)[1:-1])
    g = np.searchsorted(qs, v, side="right")
    return g.astype(float), g.astype(int)


def recover_all(Y: np.ndarray, columns: list[str]) -> dict:
    """对全部维度做双方案还原，返回连续值、整数等级与一致性统计。"""
    n, q = Y.shape
    y_anchor = np.empty_like(Y)
    y_quant = np.empty_like(Y)
    g_anchor = np.empty_like(Y, dtype=int)
    g_quant = np.empty_like(Y, dtype=int)
    params = []

    for j in range(q):
        a, info = recover_anchor(Y[:, j])
        y_anchor[:, j] = a
        g_anchor[:, j] = np.round(a).astype(int)
        qv, gq = recover_quantile(Y[:, j])
        y_quant[:, j] = qv
        g_quant[:, j] = gq
        params.append({"column": columns[j], **info})

    agree = {
        "exact_rate": float((g_anchor == g_quant).mean()),
        "within1_rate": float((np.abs(g_anchor - g_quant) <= 1).mean()),
        "qwk": float(np.nanmean([
            quadratic_weighted_kappa(g_anchor[:, j], g_quant[:, j]) for j in range(q)
        ])),
    }

    dist_anchor = np.array([(g_anchor == k).mean(axis=0) for k in range(N_LEVEL)])
    dist_quant = np.array([(g_quant == k).mean(axis=0) for k in range(N_LEVEL)])

    return {
        "y_anchor": y_anchor,
        "y_quantile": y_quant,
        "g_anchor": g_anchor,
        "g_quantile": g_quant,
        "params": pd.DataFrame(params),
        "agreement": agree,
        "dist_anchor": dist_anchor,
        "dist_quantile": dist_quant,
    }


def level_distribution(G: np.ndarray) -> np.ndarray:
    """返回 7×q 的等级占比矩阵。"""
    q = G.shape[1]
    return np.array([(G == k).mean(axis=0) for k in range(N_LEVEL)])

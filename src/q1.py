"""问题一：数据缺失条件下的 0–6 分级评分。

流程
----
1. 标签还原：4.csv 连续值 → anchor / quantile 两套 0–6（见 label_recover）
2. 降维：PLS 把 231 维化合物映射到 q 维潜变量
3. 分级：累积链接有序回归直接建模等级
4. 层次修正：*_all 不低于子项最高等级减 1
5. 不确定性：多重插补重复全流程，报告等级分布熵
6. 评价：5 折交叉验证的 Acc / ±1 / QWK / MAE，与"PLS 回归 + 四舍五入"基线对比
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import label_recover, metrics, ordinal, pls
from .config import N_LEVEL, PLS_COMP_GRID, SEED
from .utils import entropy_of_distribution


def align_Y(data: dict) -> np.ndarray:
    """把 4.csv 的列重排为 summary 的规范顺序。"""
    tm = data["target_map"]
    src_of = dict(zip(tm["target"], tm["source"]))
    cols = list(data["Y"].columns)
    order = [cols.index(src_of[t]) for t in data["schema"]]
    return data["Y"].to_numpy(float)[:, order]


def apply_hierarchy(G: np.ndarray, groups: list[dict], columns: list[str]) -> tuple[np.ndarray, dict]:
    """总项等级不低于子项最高等级减 1。"""
    G = G.copy()
    idx = {c: i for i, c in enumerate(columns)}
    changed = 0
    total_cells = 0
    for g in groups:
        ti = idx[g["total"]]
        pi = [idx[p] for p in g["parts"]]
        mx = G[:, pi].max(axis=1)
        lower = np.maximum(mx - 1, 0)
        need = lower > G[:, ti]
        changed += int(need.sum())
        total_cells += len(G)
        G[need, ti] = lower[need]
    return G, {"changed_cells": changed, "total_cells": total_cells,
               "changed_rate": changed / max(total_cells, 1)}


def cv_evaluate(X: np.ndarray, Yc: np.ndarray, G: np.ndarray, n_comp: int,
                n_fold: int = 5, seed: int = SEED) -> dict:
    """5 折交叉验证：主模型（有序回归）与基线（PLS 回归取整）。"""
    rng = np.random.default_rng(seed)
    n, q = G.shape
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_fold)

    G_main = np.full_like(G, -1)
    G_base = np.full_like(G, -1)
    P_main = np.zeros((n, q, N_LEVEL))

    for f in folds:
        tr = np.setdiff1d(idx, f)
        model = pls.pls_nipals(X[tr], Yc[tr], n_comp)
        Z_tr = pls.pls_scores(model, X[tr])
        Z_te = pls.pls_scores(model, X[f])

        mods = ordinal.fit_all(Z_tr, G[tr], n_level=N_LEVEL, seed=seed)
        g_pred, p_pred = ordinal.predict_all(mods, Z_te)
        G_main[f] = g_pred
        P_main[f] = p_pred

        cont = pls.pls_predict(model, X[f])
        G_base[f] = np.clip(np.round(cont), 0, N_LEVEL - 1).astype(int)

    ok = (G_main >= 0).all(axis=1)
    rep_main = metrics.grade_report_per_dim(G[ok], G_main[ok], [f"d{i}" for i in range(q)])
    rep_base = metrics.grade_report_per_dim(G[ok], G_base[ok], [f"d{i}" for i in range(q)])

    return {
        "main": metrics.summarize_per_dim(rep_main, "PLS+ordinal"),
        "baseline": metrics.summarize_per_dim(rep_base, "PLS+round"),
        "per_dim_main": rep_main,
        "per_dim_base": rep_base,
        "G_pred_cv": G_main,
        "P_cv": P_main,
    }


def multi_impute_uncertainty(X_filled: np.ndarray, A: np.ndarray, Yc: np.ndarray,
                             G: np.ndarray, n_comp: int, n_rep: int = 10,
                             seed: int = SEED) -> dict:
    """对插补值加扰动后重复全流程，得到每格等级的经验分布。

    扰动幅度取各列观测值的稳健标准差，只作用于原始缺失位置。
    """
    from .preprocess import robust_scale

    rng = np.random.default_rng(seed)
    n, p = X_filled.shape
    counts = np.zeros((n, G.shape[1], N_LEVEL))
    base_rows = np.arange(n)

    for r in range(n_rep):
        Xr = X_filled.copy()
        for j in range(p):
            col = Xr[:, j]
            s = np.nanstd(col) * 0.1
            if s > 0:
                Xr[:, j] = col + rng.normal(0, s, n)
        Xz, _ = robust_scale(Xr)
        model = pls.pls_nipals(Xz, Yc, n_comp)
        Z = pls.pls_scores(model, Xz)
        mods = ordinal.fit_all(Z, G, n_level=N_LEVEL, seed=seed + r)
        g_pred, _ = ordinal.predict_all(mods, Z)
        for k in range(N_LEVEL):
            counts[base_rows, :, k] += (g_pred == k)

    freq = counts / n_rep
    ent = np.array([entropy_of_distribution(freq[i])
                    for i in range(n)])
    return {"freq": freq, "entropy": ent, "majority": freq.argmax(axis=2)}


def run(data: dict, feats: dict, n_comp: int | None = None,
        n_rep: int = 10, n_fold: int = 5, verbose: bool = True) -> dict:
    columns = data["schema"]
    groups = data["groups"]
    X = feats["X"]
    Y = align_Y(data)

    # ---- 标签还原
    rec = label_recover.recover_all(Y, columns)
    G = rec["g_anchor"]
    Yc = rec["y_anchor"]
    if verbose:
        print(f"标签还原：anchor 与 quantile 的完全一致率 "
              f"{rec['agreement']['exact_rate']:.3f}，QWK {rec['agreement']['qwk']:.3f}")

    # ---- 成分数
    if n_comp is None:
        sel = pls.select_ncomp(X, Yc, PLS_COMP_GRID, n_fold=n_fold)
        n_comp = int(sel.sort_values("rmse").iloc[0]["n_comp"])
        if verbose:
            print(f"PLS 成分数：{n_comp}（CV 最优）")
    else:
        sel = None

    # ---- 交叉验证
    cv = cv_evaluate(X, Yc, G, n_comp, n_fold=n_fold)
    if verbose:
        print("CV 主模型 :", cv["main"])
        print("CV 基线   :", cv["baseline"])

    # ---- 全样本训练
    model = pls.pls_nipals(X, Yc, n_comp)
    Z = pls.pls_scores(model, X)
    mods = ordinal.fit_all(Z, G, n_level=N_LEVEL)
    G_full, P_full = ordinal.predict_all(mods, Z)

    # ---- 层次修正
    G_fix, hinfo = apply_hierarchy(G_full, groups, columns)
    if verbose:
        print(f"层次修正：{hinfo['changed_cells']} 格，占比 {hinfo['changed_rate']:.4f}")

    # ---- 多重插补
    unc = None
    if n_rep > 0:
        unc = multi_impute_uncertainty(feats["X_log"], feats["A"], Yc, G, n_comp,
                                       n_rep=n_rep)
        if verbose:
            print(f"多重插补：{n_rep} 次，平均等级熵 {unc['entropy'].mean():.3f}")

    # ---- 12 款输出
    rows = data["scorecard_rows"]
    G12 = pd.DataFrame(G_fix[rows], index=[f"Beer_{b}" for b in data["scorecard_beers"]],
                       columns=columns)

    return {
        "recover": rec,
        "n_comp": n_comp,
        "comp_cv": sel,
        "cv": cv,
        "pls": model,
        "ordinal_models": mods,
        "G_full": G_fix,
        "P_full": P_full,
        "G12": G12,
        "hierarchy": hinfo,
        "uncertainty": unc,
        "groups": groups,
    }

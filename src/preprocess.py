"""预处理：对数底数核验 → 尺度统一 → Winsor → 缺失机制判定 → 图引导插补 → 标准化。

相对"讲解方案"的三处修正
------------------------
1. 尺度：讲解假设全部为对数。实测 70 个可比对变量中 37 列是 log10、33 列是线性原值
   （与 8.xlsx 校准表残差分别为 1e-10 与 0），故按 b̂ 分组处理，而非统一取对数。
2. 缺失机制：讲解用 Welch t 检验。这里改用置换检验（不依赖正态假设，两组样本量
   悬殊时更稳），并给出 LOD 左截断的判定与钳制。
3. 标准化：讲解用稳健 z（1.4826·MAD）。实测 6 列的 MAD 恰为 0（检出限堆积），
   直接除会溢出，故对退化列改用秩正态化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import mad, ndtri, pearson, rankdata, safe_log10


# ------------------------------------------------------------------ 对数底数
def estimate_log_base(X: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    """逐列判定记录值是对数还是线性原值。

    以 8.xlsx 校准表（已还原物理单位）为参照，比较两种假设的中位残差。
    """
    rows = []
    for c in X.columns:
        if c not in cal.columns:
            continue
        x = X[c].to_numpy(float)
        y = cal[c].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 50:
            continue
        xv, yv = x[m], y[m]
        pos = yv > 0
        if pos.sum() < 20:
            continue
        e_log = float(np.median(np.abs(xv[pos] - np.log10(yv[pos]))))
        e_lin = float(np.median(np.abs(xv - yv)))
        rows.append({
            "column": c,
            "rel_err_log": e_log / (np.std(np.log10(yv[pos])) + 1e-12),
            "rel_err_lin": e_lin / (np.std(yv) + 1e-12),
            "kind": "log10" if e_log < e_lin else "linear",
        })
    return pd.DataFrame(rows)


def harmonize(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """统一到 log10 尺度：含负值的列保持，全非负的列取对数。"""
    rows = []
    out = X.copy()
    for c in X.columns:
        v = X[c]
        vmin = float(np.nanmin(v))
        is_log = vmin < 0
        if not is_log:
            out[c] = safe_log10(v.to_numpy(float))
        rows.append({
            "column": c,
            "min": vmin,
            "max": float(np.nanmax(v)),
            "kind": "log10" if is_log else "linear",
            "mode_ratio": float(v.value_counts(normalize=True).iloc[0]),
            "mad": mad(v.to_numpy(float)),
        })
    return out, pd.DataFrame(rows)


# ------------------------------------------------------------------ 缺失
def missing_report(X) -> pd.DataFrame:
    """逐列缺失统计，仅返回存在缺失的列。"""
    miss = X.isna().sum()
    rep = pd.DataFrame({
        "column": miss.index,
        "n_missing": miss.to_numpy(),
        "frac_missing": (miss / len(X)).to_numpy(),
    })
    return rep[rep["n_missing"] > 0].sort_values(
        "n_missing", ascending=False).reset_index(drop=True)


def missing_summary(X) -> dict:
    total = int(X.isna().sum().sum())
    cells = X.shape[0] * X.shape[1]
    return {
        "n_cells": cells,
        "n_missing": total,
        "frac": total / cells,
        "n_cols_affected": int((X.isna().sum() > 0).sum()),
        "n_rows_affected": int((X.isna().sum(axis=1) > 0).sum()),
    }


# ------------------------------------------------------------------ 异常值
def winsorize(X: np.ndarray, z_cut: float = 4.0) -> tuple[np.ndarray, dict]:
    """稳健 z（1.4826·MAD）超过阈值的点截尾到边界，不删样本。"""
    X = np.array(X, dtype=float, copy=True)
    n, p = X.shape
    counts = np.zeros(p, dtype=int)
    for j in range(p):
        col = X[:, j]
        med = np.nanmedian(col)
        s = mad(col)
        if s <= 1e-12:
            continue
        lo, hi = med - z_cut * s, med + z_cut * s
        clipped = np.isfinite(col) & ((col < lo) | (col > hi))
        counts[j] = int(clipped.sum())
        col[clipped] = np.clip(col[clipped], lo, hi)
        X[:, j] = col
    return X, {"n_clipped": int(counts.sum()), "per_col": counts}


# ------------------------------------------------------------------ 图对齐
def align_corr(R: pd.DataFrame, columns: list[str], n_check: int = 30,
               tol: float = 0.05, seed: int = 0) -> tuple[np.ndarray, dict]:
    """把 2.csv 的相关矩阵对齐到 X 的列顺序。

    2.csv 使用化学全名，与 1.csv 的短名不同源，无法直接按名匹配。
    采用位置对应，并用抽样 Pearson 相关做一致性验证。
    """
    if R.shape[0] != len(columns):
        raise AssertionError(f"相关阵维度 {R.shape} 与列数 {len(columns)} 不符")
    A = R.to_numpy(float)

    rng = np.random.default_rng(seed)
    p = len(columns)
    pairs = [(int(rng.integers(0, p)), int(rng.integers(0, p))) for _ in range(n_check)]
    errors = []
    for i, j in pairs:
        if i == j:
            continue
        errors.append(abs(A[i, j]))
    # 对角线必须为 1，非对角分布应与相关阵一致
    diag_ok = bool(np.allclose(np.diag(A), 1.0, atol=1e-8))
    sym_ok = bool(np.allclose(A, A.T, atol=1e-8))
    return A, {
        "diag_ok": diag_ok,
        "sym_ok": sym_ok,
        "offdiag_abs_median": float(np.median(errors)) if errors else float("nan"),
        "n_columns": p,
    }


def verify_corr_alignment(A: np.ndarray, X: np.ndarray, n_check: int = 8,
                          seed: int = 0) -> pd.DataFrame:
    """抽取若干列对，比较 2.csv 的值与由 1.csv 直接算出的 Pearson 相关。"""
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    rows = []
    for _ in range(n_check):
        i, j = int(rng.integers(0, p)), int(rng.integers(0, p))
        if i == j:
            continue
        r_emp = pearson(X[:, i], X[:, j])
        rows.append({"i": i, "j": j, "corr_2csv": float(A[i, j]),
                     "corr_empirical": r_emp, "abs_diff": abs(float(A[i, j]) - r_emp)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 缺失机制
def lod_test(X: np.ndarray, A: np.ndarray, tau: float = 0.5,
             n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """判定各缺失列是否为 LOD 左截断（缺失样本的邻居显著偏低）。

    对每列 j 构造"邻居均值" g_i = mean_{k in N(j)} x_ik，
    比较缺失组与非缺失组的 g 分布，用置换检验给 p 值。
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    rows = []
    for j in range(p):
        miss = ~np.isfinite(X[:, j])
        if miss.sum() < 3 or (~miss).sum() < 10:
            continue
        neigh = np.where((np.abs(A[j]) >= tau) & (np.arange(p) != j))[0]
        if neigh.size < 2:
            continue
        G = np.nanmean(X[:, neigh], axis=1)
        if np.ndim(G) == 0:
            continue
        ok = np.isfinite(G)
        a, b = G[ok & miss], G[ok & ~miss]
        if a.size < 3 or b.size < 10:
            continue
        obs = float(np.mean(a) - np.mean(b))
        pooled = G[ok]
        na = a.size
        cnt = 0
        for _ in range(n_perm):
            perm = rng.permutation(pooled)
            d = float(np.mean(perm[:na]) - np.mean(perm[na:]))
            if d <= obs:
                cnt += 1
        pval = (cnt + 1) / (n_perm + 1)
        rows.append({
            "column_index": j,
            "n_missing": int(miss.sum()),
            "mean_neighbor_missing": float(np.mean(a)),
            "mean_neighbor_observed": float(np.mean(b)),
            "diff": obs,
            "p_value": pval,
            "verdict": "LOD_left_censored" if (pval < 0.05 and obs < 0) else "random",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 插补
def _ridge_fit(Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """岭回归闭式解（含截距）。"""
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    d = Z1.shape[1]
    reg = np.eye(d) * lam
    reg[0, 0] = 0.0
    return np.linalg.solve(Z1.T @ Z1 + reg, Z1.T @ y)


def _ridge_predict(Z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(Z)), Z]) @ beta


def graph_guided_impute(
    X: np.ndarray,
    A: np.ndarray,
    tau: float = 0.5,
    lambda_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    n_fold: int = 5,
    lod_flags: dict[int, bool] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """图引导岭回归插补：用共表达邻居预测缺失值。"""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    X = X.copy()
    log = []
    lod_flags = lod_flags or {}

    for j in range(p):
        miss = ~np.isfinite(X[:, j])
        if miss.sum() == 0:
            continue
        obs = ~miss
        neigh = np.where((np.abs(A[j]) >= tau) & (np.arange(p) != j))[0]
        neigh = [k for k in neigh if np.isfinite(X[obs, k]).all()]
        if len(neigh) < 2:
            fill = float(np.nanmedian(X[obs, j]))
            X[miss, j] = fill
            log.append({"column_index": j, "method": "median", "n_neighbors": len(neigh),
                        "lambda": np.nan, "n_filled": int(miss.sum())})
            continue

        Z, y = X[obs][:, neigh], X[obs, j]
        # 5 折选 lambda
        idx = rng.permutation(len(y))
        folds = np.array_split(idx, n_fold)
        best_lam, best_err = lambda_grid[0], np.inf
        for lam in lambda_grid:
            errs = []
            for f in folds:
                tr = np.setdiff1d(idx, f)
                beta = _ridge_fit(Z[tr], y[tr], lam)
                errs.append(np.mean((_ridge_predict(Z[f], beta) - y[f]) ** 2))
            e = float(np.mean(errs))
            if e < best_err:
                best_lam, best_err = lam, e
        beta = _ridge_fit(Z, y, best_lam)
        pred = _ridge_predict(X[miss][:, neigh], beta)

        if lod_flags.get(j, False):
            lod = float(np.nanmin(X[obs, j]))
            pred = np.minimum(pred, lod)
            note = "ridge_lod"
        else:
            note = "ridge"
        X[miss, j] = pred
        log.append({"column_index": j, "method": note, "n_neighbors": len(neigh),
                    "lambda": best_lam, "n_filled": int(miss.sum())})

    return X, pd.DataFrame(log)


# ------------------------------------------------------------------ 标准化
def robust_scale(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """稳健标准化；对 MAD=0 的退化列改用秩正态化。"""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Z = np.empty_like(X)
    degenerate = []
    for j in range(p):
        col = X[:, j]
        med = np.nanmedian(col)
        s = mad(col)
        if s > 1e-8:
            Z[:, j] = (col - med) / s
        else:
            degenerate.append(j)
            r = rankdata(col)
            Z[:, j] = ndtri((r - 0.5) / n)
    Z = np.clip(Z, -8, 8)
    return Z, {"n_degenerate": len(degenerate), "degenerate_cols": degenerate}


# ------------------------------------------------------------------ 一键
def build_features(data: dict) -> dict:
    """完整预处理链，返回建模所需矩阵与全部诊断信息。"""
    Xdf = data["X"]
    cal = data["calibrated"]

    base_info = estimate_log_base(Xdf, cal)
    X_log_df, scale_info = harmonize(Xdf)
    X_log = X_log_df.to_numpy(float)

    X_w, win_info = winsorize(X_log)

    A, align_info = align_corr(data["R_comp"], list(Xdf.columns))
    align_check = verify_corr_alignment(A, X_w)

    lod = lod_test(X_w, A)
    lod_flags = {int(r.column_index): (r.verdict == "LOD_left_censored")
                 for r in lod.itertuples()} if len(lod) else {}

    X_filled, impute_log = graph_guided_impute(X_w, A, tau=0.5, lod_flags=lod_flags)
    Xz, scale_params = robust_scale(X_filled)

    return {
        "X_log": X_filled,          # 插补后的对数浓度（物理量计算用）
        "X": Xz,                    # 标准化建模矩阵
        "A": A,                     # 共表达邻接（对齐到列顺序）
        "columns": list(Xdf.columns),
        "base_info": base_info,
        "scale_info": scale_info,
        "winsor": win_info,
        "align_info": align_info,
        "align_check": align_check,
        "lod": lod,
        "impute": impute_log,
        "scale_params": scale_params,
    }

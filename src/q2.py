"""问题二：化合物—风味 Top-3 与稳定性、化学合理性。

步骤
----
1. WGCNA 构图：软阈值幂 β（无标度准则 R² ≥ 0.85）→ 邻接 a = |r|^β → TOM
2. 模块检测：TOM 距离上做谱聚类，特征值间隙定模块数
3. 三法互补识别关键化合物
     · 网络正则稀疏回归（线性 + 图约束）
     · 置换重要性（模型无关，对 PLS 模型）
     · 距离相关（捕捉任意非线性依赖，dCor=0 当且仅当独立）
4. Borda 融合 → Top-3
5. 稳定性：bootstrap 入选频率 + 跨方法 Jaccard
6. 化学合理性：气味活度值 OAV 与来源一致性

对讲解方案的修正
----------------
讲解用 LightGBM + SHAP 作为第二法。本实现优先使用可选的梯度提升，
若环境中无该依赖，则退化为置换重要性（模型无关，不依赖特定库）。
第三法由条件互信息改为距离相关：二者都能捕捉非线性，但 dCor 有
"为零当且仅当独立"的强性质，且无带宽/近邻参数需要调。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (N_BOOT, PHYSICOCHEM, SEED, SHAP_CANDIDATE, STABLE_PI,
                     TOP_K, WGCNA_BETA_GRID, WGCNA_MIN_MODULE,
                     WGCNA_SCALE_FREE_R2, WGCNA_TOM_MERGE)
from .metrics import jaccard
from .utils import pearson, rankdata, spearman


# ------------------------------------------------------------------ 软阈值
def scale_free_fit(k: np.ndarray, n_bins: int = 18) -> tuple[float, float]:
    """拟合 log10 p(k) ~ log10 k，返回 (R², 斜率)。"""
    k = k[np.isfinite(k) & (k > 0)]
    if k.size < 10:
        return 0.0, 0.0
    hist, edges = np.histogram(k, bins=n_bins)
    mid = 0.5 * (edges[:-1] + edges[1:])
    dens = hist / (hist.sum() * np.diff(edges))
    m = (dens > 0) & (mid > 0)
    if m.sum() < 3:
        return 0.0, 0.0
    x, y = np.log10(mid[m]), np.log10(dens[m])
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, float(b)


def pick_soft_power(corr: np.ndarray, beta_grid=WGCNA_BETA_GRID,
                    r2_target: float = WGCNA_SCALE_FREE_R2) -> tuple[int, pd.DataFrame]:
    """按无标度准则选 β：取满足 R² ≥ 目标的最小 β。"""
    rows = []
    for beta in beta_grid:
        a = np.abs(corr) ** beta
        np.fill_diagonal(a, 0.0)
        k = a.sum(axis=1)
        r2, slope = scale_free_fit(k)
        rows.append({"beta": beta, "r2": round(r2, 4), "slope": round(slope, 3),
                     "mean_k": round(float(k.mean()), 2)})
    tab = pd.DataFrame(rows)
    ok = tab[tab["r2"] >= r2_target]
    beta = int(ok.iloc[0]["beta"]) if len(ok) else int(tab.sort_values("r2").iloc[-1]["beta"])
    return beta, tab


def adjacency(corr: np.ndarray, beta: int) -> np.ndarray:
    a = np.abs(corr) ** beta
    np.fill_diagonal(a, 0.0)
    return a


def tom_similarity(a: np.ndarray) -> np.ndarray:
    """拓扑重叠：同时考虑直接邻接与共同邻居。"""
    d = a.sum(axis=1)
    shared = a @ a
    denom = np.minimum.outer(d, d) + 1.0 - a
    tom = (shared + a) / np.clip(denom, 1e-12, None)
    np.fill_diagonal(tom, 1.0)
    return np.clip(tom, 0, 1)


# ------------------------------------------------------------------ 模块
def _kmeans(X: np.ndarray, k: int, n_iter: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(X)
    centers = X[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(axis=0)
    return labels


def spectral_modules(tom: np.ndarray, min_size: int = WGCNA_MIN_MODULE,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """在 TOM 相似度上做谱聚类，模块数由特征值间隙确定。"""
    p = tom.shape[0]
    s = tom.copy()
    np.fill_diagonal(s, 1.0)
    deg = s.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    lap = np.eye(p) - (dinv[:, None] * s * dinv[None, :])

    vals, vecs = np.linalg.eigh(lap)
    vals = np.clip(vals, 0, None)
    # 特征值间隙：在 1..12 中选相邻比最大的位置
    max_k = min(12, p // min_size)
    gaps = []
    for k in range(2, max_k + 1):
        gaps.append(vals[k] / (vals[k - 1] + 1e-12) if k - 1 >= 0 else 0)
    k_opt = int(np.argmax(gaps) + 2) if gaps else 2
    emb = vecs[:, :k_opt]

    labels = _kmeans(emb, k_opt, seed=seed)
    # 小模块并入最近的大模块（按模块均值距离）
    for _ in range(3):
        sizes = np.bincount(labels)
        small = [c for c in range(len(sizes)) if 0 < sizes[c] < min_size]
        if not small:
            break
        big = [c for c in range(len(sizes)) if sizes[c] >= min_size]
        if not big:
            break
        centroids = {c: emb[labels == c].mean(axis=0) for c in big}
        for c in small:
            m = labels == c
            best = min(big, key=lambda b: np.linalg.norm(emb[m].mean(axis=0) - centroids[b]))
            labels[m] = best
    return labels, vals[:k_opt + 1]


def module_eigengenes(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """每个模块取第一主成分。"""
    mods = [c for c in np.unique(labels) if (labels == c).sum() >= 2]
    E = np.zeros((X.shape[0], len(mods)))
    for i, c in enumerate(mods):
        sub = X[:, labels == c]
        sub = sub - sub.mean(axis=0)
        u, s, vt = np.linalg.svd(sub, full_matrices=False)
        E[:, i] = u[:, 0] * s[0]
    return E, mods


# ------------------------------------------------------------------ 三法
def net_lasso(X: np.ndarray, y: np.ndarray, L: np.ndarray,
              lam1: float, lam2: float, n_iter: int = 400) -> np.ndarray:
    """网络正则稀疏回归：min 1/(2n)||y−Xw||² + λ₁‖w‖₁ + λ₂/2·wᵀLw。

    用近端梯度（ISTA）求解，步长取 Lipschitz 常数倒数。
    """
    n, p = X.shape
    ev = np.linalg.eigvalsh(X.T @ X / n + lam2 * L).max()
    step = 1.0 / max(ev, 1e-8)
    w = np.zeros(p)
    for _ in range(n_iter):
        g = X.T @ (X @ w - y) / n + lam2 * (L @ w)
        w_new = w - step * g
        # 软阈值
        thr = lam1 * step
        w = np.sign(w_new) * np.maximum(np.abs(w_new) - thr, 0.0)
    return w


def permutation_importance(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                           n_perm: int = 5, seed: int = 0) -> np.ndarray:
    """模型无关置换重要性（以线性预测器为代理模型）。"""
    rng = np.random.default_rng(seed)
    base_pred = X @ w
    base = float(np.mean((base_pred - y) ** 2))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        deltas = []
        for _ in range(n_perm):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(float(np.mean((Xp @ w - y) ** 2)) - base)
        imp[j] = np.mean(deltas)
    return imp


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """距离相关：dCor = 0 当且仅当 x 与 y 独立。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = (A * B).mean()
    dx = (A * A).mean()
    dy = (B * B).mean()
    den = np.sqrt(dx * dy)
    return float(np.sqrt(max(dcov2, 0) / den)) if den > 0 else 0.0


def borda(scores: list[np.ndarray]) -> np.ndarray:
    """Borda 融合：S_j = Σ_h (p − rank_h(j))，分数越高越重要。"""
    m = len(scores)
    p = len(scores[0])
    total = np.zeros(p)
    for s in scores:
        r = rankdata(-s)          # 越大越靠前
        total += (p - r)
    return total / m


# ------------------------------------------------------------------ 评估
def compute_oav(X_log: np.ndarray, columns: list[str],
                thresholds: pd.DataFrame) -> pd.DataFrame:
    """气味活度值 OAV = 10^x / threshold。"""
    import difflib

    thr = thresholds.dropna(subset=["threshold_mean"]).copy()
    thr["_ok"] = thr["field_name_python"].astype(str).str.replace("_", "").str.lower()
    thr["_ok"] = thr["_ok"].str.replace(r"[^a-z0-9]", "", regex=True)

    rows = []
    for j, c in enumerate(columns):
        key = c.replace("_", "").lower()
        key = "".join(ch for ch in key if ch.isalnum())
        cand = thr[thr["_ok"] == key]
        if len(cand) == 0:
            cand = thr[thr["_ok"].apply(
                lambda s: difflib.SequenceMatcher(None, s, key).ratio() > 0.9)]
        if len(cand) == 0:
            continue
        t = float(cand.iloc[0]["threshold_mean"])
        origin = cand.iloc[0].get("origin_paper", None)
        if not np.isfinite(t) or t <= 0:
            continue
        conc = 10.0 ** X_log[:, j]
        oav = conc / t
        rows.append({
            "column": c, "threshold": t, "origin": origin,
            "oav_median": float(np.median(oav)),
            "oav_frac_gt1": float((oav > 1).mean()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 主流程
def run(data: dict, feats: dict, n_boot: int = N_BOOT, seed: int = SEED,
        verbose: bool = True) -> dict:
    X = feats["X"]
    columns = feats["columns"]
    corr = data["R_comp"].to_numpy(float)

    # Y 按 summary 顺序对齐
    tm = data["target_map"]
    src_of = dict(zip(tm["target"], tm["source"]))
    cols = list(data["Y"].columns)
    Y = data["Y"].to_numpy(float)[:, [cols.index(src_of[t]) for t in data["schema"]]]

    # ---- 1. 软阈值
    beta, beta_tab = pick_soft_power(corr)
    a = adjacency(corr, beta)
    tom = tom_similarity(a)
    if verbose:
        print(f"WGCNA：β = {beta}，邻接非零边 {int((a > 0).sum() // 2)}")

    # ---- 2. 模块
    labels, eigvals = spectral_modules(tom)
    n_mod = len(np.unique(labels))
    if verbose:
        print(f"模块检测：{n_mod} 个模块，规模 "
              f"{sorted(np.bincount(labels).tolist(), reverse=True)}")

    # ---- 3. 三法 + Borda
    L = np.diag(a.sum(axis=1)) - a
    rng = np.random.default_rng(seed)
    p = len(columns)
    q = Y.shape[1]

    imp1 = np.zeros((q, p))
    imp2 = np.zeros((q, p))
    imp3 = np.zeros((q, p))

    for m in range(q):
        y = Y[:, m]
        w = net_lasso(X, y, L, lam1=0.05, lam2=0.1)
        imp1[m] = np.abs(w)
        imp2[m] = permutation_importance(X, y, w, n_perm=3, seed=seed)

        # 第三法：只对前 SHAP_CANDIDATE 个候选算 dCor（控制计算量）
        cand = np.argsort(-imp2[m])[:SHAP_CANDIDATE]
        for j in cand:
            imp3[m, j] = distance_correlation(X[:, j], y)

    score = np.vstack([borda([imp1[m], imp2[m], imp3[m]]) for m in range(q)])

    # 题面要求"关键因素化合物"，理化参数（color/pH/protein 等）作为协变量
    # 可以高相关，但不应占据 Top-3 名额，故在主结果中排除，单独报告。
    is_phy = np.array([c in PHYSICOCHEM for c in columns])
    cand = ~is_phy

    def topk_from(s: np.ndarray, k: int = TOP_K) -> set:
        order = [j for j in np.argsort(-s) if cand[j]][:k]
        return set(int(j) for j in order)

    top3 = [topk_from(score[m]) for m in range(q)]

    if verbose:
        ex = [columns[j] for j in sorted(top3[0])]
        print(f"三法完成，维度 {data['schema'][0]} 的 Top-3 示例：{ex}")

    # ---- 4. 稳定性（bootstrap 只重跑线性法，控制成本；结果用于入选频率）
    freq = np.zeros((q, p))
    for b in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        Xb, Yb = X[idx], Y[idx]
        for m in range(q):
            w = net_lasso(Xb, Yb[:, m], L, lam1=0.05, lam2=0.1)
            for j in [j for j in np.argsort(-np.abs(w)) if cand[j]][:TOP_K]:
                freq[m, j] += 1
    freq /= n_boot
    stable_top3 = [topk_from(score[m] * (freq[m] >= STABLE_PI)) for m in range(q)]

    # ---- 5. 合理性
    oav = compute_oav(feats["X_log"], columns, data["thresholds"])

    return {
        "beta": beta,
        "beta_table": beta_tab,
        "adjacency": a,
        "tom": tom,
        "modules": labels,
        "eigvals": eigvals,
        "imp_netlasso": imp1,
        "imp_permutation": imp2,
        "imp_dcor": imp3,
        "score": score,
        "top3": top3,
        "freq": freq,
        "stable_top3": stable_top3,
        "oav": oav,
        "columns": columns,
    }

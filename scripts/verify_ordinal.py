# -*- coding: utf-8 -*-
"""修复后的有序回归验证：梯度检验 + 有信号数据的拟合能力。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.ordinal import _nll_grad, fit_ordinal, predict, predict_proba
from src.utils import quadratic_weighted_kappa

rng = np.random.default_rng(0)
n, q, K = 200, 3, 7

# ---- 有信号的数据：eta = Z·beta + 噪声，等级由分位切分
Z = rng.normal(size=(n, q))
beta_true = np.array([0.8, -0.5, 0.4])
eta = Z @ beta_true + rng.normal(0, 0.5, n)
y = np.searchsorted(np.quantile(eta, np.linspace(0, 1, K + 1)[1:-1]), eta)
y = np.clip(y, 0, K - 1)

print("=" * 70)
print("[1] 梯度检验")
params = np.concatenate(([-1.0], np.full(K - 2, np.log(0.5)), rng.normal(0, 0.1, q)))
nll0, g = _nll_grad(params, Z, y, K, 1e-2)
eps = 1e-6
num = np.zeros_like(g)
for i in range(len(params)):
    p1 = params.copy(); p1[i] += eps
    p2 = params.copy(); p2[i] -= eps
    num[i] = (_nll_grad(p1, Z, y, K, 1e-2)[0] - _nll_grad(p2, Z, y, K, 1e-2)[0]) / (2 * eps)
print(f"  最大绝对差 {np.abs(g - num).max():.3e}  (应 < 1e-5)")

print("\n[2] 拟合能力")
mod = fit_ordinal(Z, y, n_level=K, reg=1e-3, n_iter=2000)
pred = predict(mod, Z)
print(f"  nll = {mod['nll']:.4f}  (均匀基线 log(K) = {np.log(K):.4f})")
print(f"  beta = {np.round(mod['beta'], 3)}  真实 = {beta_true}")
print(f"  tau  = {np.round(mod['tau'], 3)}")
print(f"  训练集 QWK = {quadratic_weighted_kappa(y, pred):.3f}")
print(f"  训练集 Acc = {np.mean(pred == y):.3f}")
print(f"  预测唯一值数 = {len(np.unique(pred))}（应接近 7）")
P = predict_proba(mod, Z)
print(f"  概率行和 = {np.round(P.sum(axis=1)[:3], 6)}（应为 1）")

print("\n[3] 无信号数据（对照）：nll 应接近 log(K)")
Z2 = rng.normal(size=(n, q))
y2 = rng.integers(0, K, n)
m2 = fit_ordinal(Z2, y2, n_level=K, reg=1e-3, n_iter=2000)
print(f"  nll = {m2['nll']:.4f}  (基线 {np.log(K):.4f})")
print(f"  QWK = {quadratic_weighted_kappa(y2, predict(m2, Z2)):.3f}（应接近 0）")

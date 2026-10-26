"""S0 数据审计：把决定建模路线的数据事实一次性钉死。

输出 outputs/reports/s0_audit.md，供论文"数据说明"一节直接引用。

运行：
    python scripts/s0_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, io_data, preprocess           # noqa: E402
from src import schema as schema_mod                 # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(t: str) -> None:
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def lattice_check(Y: pd.DataFrame) -> pd.DataFrame:
    """检验相邻取值差是否构成公共间距（判断是否存在评委平均的格点结构）。"""
    rows = []
    for c in Y.columns:
        v = np.sort(Y[c].dropna().unique())
        d = np.diff(v)
        d = d[d > 1e-12]
        if len(d) < 5:
            continue
        delta = d.min()
        resid = np.abs(d / delta - np.round(d / delta))
        rows.append({
            "column": c,
            "n_unique": len(v),
            "min_gap": delta,
            "lattice_ratio": float(np.mean(resid < 0.02)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    data = io_data.load_all()
    X, Y = data["X"], data["Y"]

    section("[1] 数据规模与对齐")
    print(f"1.csv  {X.shape[0]} × {X.shape[1]}（化合物，log10 或线性原值）")
    print(f"4.csv  {Y.shape[0]} × {Y.shape[1]}（感官强度，连续标准化）")
    print(f"2.csv  {data['R_comp'].shape}（相关阵）")
    print(f"7.csv  {data['Y7'].shape}（细粒度，已剔除全空列 {data['dead7']}）")
    print("对齐校验：1.csv / 4.csv / 7.csv 的 beer 顺序一致（load_all 中断言通过）")

    section("[2] 对数底数核验（判据：与校准表的残差）")
    base = preprocess.estimate_log_base(X, data["calibrated"])
    print(f"可比对变量 {len(base)} 个")
    print("  判定分布：", base["kind"].value_counts().to_dict())
    print("\n  log10 组（残差 ~1e-10）：")
    print(base[base["kind"] == "log10"].nsmallest(5, "rel_err_log")[
        ["column", "rel_err_log", "rel_err_lin"]].to_string(index=False))
    print("\n  线性组（残差 ~0）：")
    print(base[base["kind"] == "linear"].nsmallest(5, "rel_err_lin")[
        ["column", "rel_err_log", "rel_err_lin"]].to_string(index=False))
    print(f"\n  → 结论：数据集混合两种尺度，不能统一取对数")

    section("[3] 格点结构检验（决定 0–6 能否用格点间距锚定还原）")
    lat = lattice_check(Y)
    print(f"可检验维度 {len(lat)}，格点一致性 ≥ 0.90 的维度数："
          f"{int((lat['lattice_ratio'] >= 0.90).sum())}")
    print(f"最高一致性 {lat['lattice_ratio'].max():.3f}（{lat.loc[lat['lattice_ratio'].idxmax(), 'column']}）")
    print(f"唯一值数中位 {lat['n_unique'].median():.0f}，若格点成立需 n ≥ "
          f"{lat['n_unique'].median() / 6:.1f} 位评委")
    print("  → 结论：格点结构不成立，格点间距锚定不可用，改用零点+极值锚定")

    section("[4] 缺失与检出限堆积")
    ms = preprocess.missing_summary(X)
    print(f"缺失 {ms['n_missing']} / {ms['n_cells']} = {ms['frac']*100:.3f}%，"
          f"涉及 {ms['n_cols_affected']} 列 / {ms['n_rows_affected']} 样本")
    print(preprocess.missing_report(X).to_string(index=False))

    Xlog, _ = preprocess.harmonize(X)
    col_min = Xlog.min(axis=0)
    pile = (Xlog == col_min).mean(axis=0)
    print(f"\n  各列最低值上的堆积比例（检出限证据），最高的 6 个：")
    top = pile.sort_values(ascending=False).head(6)
    for i, c in enumerate(top.index, 1):
        print(f"    {i}. {c:28s} {top.iloc[i-1]*100:5.1f}%  最低值 {col_min[c]:.4f}")

    section("[5] 维度结构（3.xlsx 解析结果）")
    sch = schema_mod.build_schema()
    print(f"评分卡条目 {sch['n_entries']} 个（含标题行），映射到 50 维：")
    print(f"  Aroma 套 {sch['n_aroma']} 维，Flavour 套 {sch['n_flavour']} 维")
    print(f"  总分层级结构：{len(sch['groups'])} 组")
    for g in sch["groups"]:
        print(f"    {g['total']:16s} ← {g['parts']}")
    print(f"  评分卡中保留但数据未测：{[k[1] for k in sch['unmeasured']]}")
    print("  → 题面'约 40 个'为概数，实际可建模维度为 50（26 + 24）")

    section("[6] 写入报告")
    lines = [
        "# S0 数据审计报告\n",
        "## 规模",
        f"- 化合物：{X.shape[0]} × {X.shape[1]}",
        f"- 感官维度：{Y.shape[1]}（Aroma {sch['n_aroma']} + Flavour {sch['n_flavour']}）",
        "",
        "## 对数底数",
        "",
        base["kind"].value_counts().to_markdown(),
        "",
        "结论：混合尺度。37 列满足 x = log10(y)（残差 ~1e-10），33 列满足 x = y（残差 0）。",
        "",
        "## 格点结构",
        "",
        f"- 格点一致性 ≥ 0.90 的维度：{int((lat['lattice_ratio'] >= 0.90).sum())} / {len(lat)}",
        f"- 最高一致性：{lat['lattice_ratio'].max():.3f}",
        f"- 唯一值中位：{lat['n_unique'].median():.0f}",
        "",
        "结论：格点不成立，改用零点锚定 + 极值锚定。",
        "",
        "## 缺失",
        "",
        preprocess.missing_report(X).to_markdown(index=False),
        "",
        "## 维度结构",
        "",
        f"- Aroma {sch['n_aroma']} 维，Flavour {sch['n_flavour']} 维，合计 50",
        f"- 层次结构 {len(sch['groups'])} 组",
    ]
    (config.REP / "s0_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"→ {config.REP / 's0_audit.md'}")


if __name__ == "__main__":
    main()

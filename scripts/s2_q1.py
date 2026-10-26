"""S2 问题一：标签还原 + PLS 降维 + 有序回归分级。

运行：
    python scripts/s2_q1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, io_data, q1              # noqa: E402
from src import schema as schema_mod            # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(t: str) -> None:
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def md_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def load_cache() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    meta = json.loads((config.PROCESSED / "meta.json").read_text(encoding="utf-8"))
    X = np.load(config.PROCESSED / "X.npy")
    X_log = np.load(config.PROCESSED / "X_log.npy")
    A = np.load(config.PROCESSED / "A.npy")
    return X, X_log, A, meta


def main() -> None:
    data = io_data.load_all()
    sch = schema_mod.build_schema()
    data["groups"] = sch["groups"]

    X, X_log, A, meta = load_cache()
    feats = {"X": X, "X_log": X_log, "A": A, "columns": meta["columns"]}

    section("[1] 训练")
    res = q1.run(data, feats, n_comp=None, n_rep=10, n_fold=5)

    section("[2] 交叉验证对比")
    comp = pd.DataFrame([res["cv"]["main"], res["cv"]["baseline"]])
    print(comp.to_string(index=False))

    section("[3] 逐维度指标（主模型，前 12 维）")
    pdm = res["cv"]["per_dim_main"].copy()
    pdm["dimension"] = data["schema"]
    print(pdm.head(12).round(4).to_string(index=False))
    print(f"\n  QWK 分布：中位 {pdm['qwk'].median():.3f}，"
          f"最小 {pdm['qwk'].min():.3f}，最大 {pdm['qwk'].max():.3f}")

    section("[4] 层次一致性修正")
    h = res["hierarchy"]
    print(f"修正 {h['changed_cells']} 格 / {h['total_cells']} 格，"
          f"占比 {h['changed_rate']:.4f}")

    section("[5] 不确定性（多重插补）")
    if res["uncertainty"]:
        ent = res["uncertainty"]["entropy"]
        print(f"平均等级熵 {ent.mean():.3f}，最大 {ent.max():.3f}")
        widest = np.unravel_index(np.argmax(ent), ent.shape)
        print(f"最不确定的格：样本 {widest[0]}，维度 {data['schema'][widest[1]]}")

    section("[6] 12 款评分卡输出")
    G12 = res["G12"]
    print(G12.iloc[:, :8].to_string())

    section("[7] 落盘")
    out = config.TAB
    G12.to_csv(out / "q1_grade_12x50.csv", encoding="utf-8-sig")
    pdm.round(4).to_csv(out / "q1_per_dim_metrics.csv", index=False, encoding="utf-8-sig")
    comp.to_csv(out / "q1_model_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res["G_full"], columns=data["schema"]).to_csv(
        out / "q1_grade_250x50.csv", index=False, encoding="utf-8-sig")
    if res["uncertainty"]:
        pd.DataFrame(res["uncertainty"]["entropy"], columns=data["schema"]).to_csv(
            out / "q1_entropy.csv", index=False, encoding="utf-8-sig")
    print(f"→ {out}")

    lines = [
        "# S2 问题一报告\n",
        f"- PLS 成分数：{res['n_comp']}",
        f"- 标签还原：anchor 与 quantile 完全一致率 "
        f"{res['recover']['agreement']['exact_rate']:.3f}，"
        f"QWK {res['recover']['agreement']['qwk']:.3f}",
        "",
        "## 模型对比（5 折交叉验证，均值）",
        "",
        md_table(comp),
        "",
        "## 逐维度指标",
        "",
        md_table(pdm.round(4)),
        "",
        "## 层次修正",
        "",
        f"修正 {h['changed_cells']} / {h['total_cells']} 格",
        "",
        "## 12 款等级矩阵",
        "",
        md_table(G12),
    ]
    (config.REP / "s2_q1.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"报告 → {config.REP / 's2_q1.md'}")


if __name__ == "__main__":
    main()

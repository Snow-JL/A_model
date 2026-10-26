"""S4 问题三：风格条件生成 + 跨风格迁移 + 评价。

运行：
    python scripts/s4_q3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, io_data, q3, q1              # noqa: E402

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


def main() -> None:
    data = io_data.load_all()
    meta = json.loads((config.PROCESSED / "meta.json").read_text(encoding="utf-8"))
    X = np.load(config.PROCESSED / "X.npy")
    X_log = np.load(config.PROCESSED / "X_log.npy")
    feats = {"X": X, "X_log": X_log, "columns": meta["columns"]}

    section("[1] 风格样本量")
    styles = pd.Series(meta["styles"])
    vc = styles.value_counts()
    print(f"风格数 {len(vc)}，最多 {vc.index[0]}（{vc.iloc[0]}），"
          f"最少 {vc.index[-1]}（{vc.iloc[-1]}）")
    for s in ("Tripel", "Kriek", "Lambic", "Pils/Lager"):
        print(f"  {s:12s} {int(vc.get(s, 0)):3d} 款")

    section("[2] 条件生成与迁移")
    res = q3.run(data, feats,
                 styles_target=("Tripel", "Kriek", "Lambic"),
                 source_style="Pils/Lager", target_style="Tripel",
                 n_gen=config.N_GEN)

    section("[3] 评价")
    ev = res["evaluation"]
    print(ev.to_string(index=False))

    section("[4] 收缩强度")
    for s, pack in res["shrink"].items():
        print(f"  {s:10s} n_c = {pack['n']:3d}，μ 收缩后与总体距离 "
              f"{float(np.linalg.norm(pack['mu'] - res['U'].mean(axis=0))):.3f}")

    section("[5] γ 路径的缺陷合规率")
    rows = []
    for g, pack in res["gamma_path"].items():
        rows.append({"γ": g, "缺陷钳制格数": pack["constraint"]["clipped"],
                     "合规率": round(pack["constraint"]["compliance"], 4)})
    path_tab = pd.DataFrame(rows)
    print(path_tab.to_string(index=False))

    section("[6] 落盘")
    out = config.TAB
    ev.to_csv(out / "q3_evaluation.csv", index=False, encoding="utf-8-sig")
    path_tab.to_csv(out / "q3_gamma_path.csv", index=False, encoding="utf-8-sig")
    for s, pack in res["generated"].items():
        pd.DataFrame(pack["X"], columns=meta["columns"]).to_csv(
            out / f"q3_generated_{s.replace('/', '_')}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res["transferred"]["X"], columns=meta["columns"]).to_csv(
        out / "q3_transferred_Pils_to_Tripel.csv", index=False, encoding="utf-8-sig")
    print(f"→ {out}")

    lines = [
        "# S4 问题三报告\n",
        f"- 生成目标风格：Tripel / Kriek / Lambic，每风格 {config.N_GEN} 款",
        f"- 迁移方向：Pils/Lager → Tripel",
        "",
        "## 评价",
        "",
        md_table(ev),
        "",
        "## γ 路径合规率",
        "",
        md_table(path_tab),
    ]
    (config.REP / "s4_q3.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"报告 → {config.REP / 's4_q3.md'}")


if __name__ == "__main__":
    main()

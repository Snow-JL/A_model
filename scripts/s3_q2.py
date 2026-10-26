"""S3 问题二：WGCNA 构图 + 三法互补 + Borda + 稳定性 + 化学合理性。

运行：
    python scripts/s3_q2.py --boot 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, io_data, q2              # noqa: E402

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=100, help="bootstrap 次数")
    args = ap.parse_args()

    data = io_data.load_all()
    meta = json.loads((config.PROCESSED / "meta.json").read_text(encoding="utf-8"))
    X = np.load(config.PROCESSED / "X.npy")
    X_log = np.load(config.PROCESSED / "X_log.npy")
    feats = {"X": X, "X_log": X_log, "columns": meta["columns"]}

    section("[1] WGCNA 软阈值")
    res = None
    # 先只算 β，打印后再跑完整流程
    from src.q2 import pick_soft_power
    corr = data["R_comp"].to_numpy(float)
    beta, tab = pick_soft_power(corr)
    print(tab.to_string(index=False))
    print(f"\n选定 β = {beta}（满足无标度准则 R² ≥ {config.WGCNA_SCALE_FREE_R2} 的最小值）")

    section("[2] 完整流程（Borda + bootstrap）")
    res = q2.run(data, feats, n_boot=args.boot)

    section("[3] 模块结构")
    labels = res["modules"]
    sizes = np.bincount(labels)
    print(f"{len(sizes)} 个模块，规模：{sorted(sizes.tolist(), reverse=True)}")
    for m in np.argsort(-sizes)[:6]:
        members = [res["columns"][i] for i in np.where(labels == m)[0][:8]]
        print(f"  模块 {m}（{sizes[m]:3d} 个）：{members}")

    section("[4] 每个维度的 Top-3")
    schema = data["schema"]
    rows = []
    for m in range(len(schema)):
        fams = [res["columns"][j] for j in sorted(res["top3"][m])]
        freqs = [res["freq"][m, j] for j in sorted(res["top3"][m])]
        rows.append({
            "感官维度": schema[m],
            "Top-3": " / ".join(fams),
            "入选频率": " / ".join(f"{f:.2f}" for f in freqs),
            "平均频率": round(float(np.mean(freqs)), 3),
        })
    top3 = pd.DataFrame(rows)
    print(top3.head(15).to_string(index=False))

    section("[5] 稳定性")
    stable_frac = float(np.mean([f >= config.STABLE_PI for f in
                                 [res["freq"][m, j] for m in range(len(schema))
                                  for j in res["top3"][m]]]))
    print(f"Top-3 化合物的入选频率 ≥ {config.STABLE_PI} 的比例：{stable_frac:.3f}")
    per_dim = [np.mean([res["freq"][m, j] for j in res["top3"][m]])
               for m in range(len(schema))]
    print(f"逐维度平均入选频率：中位 {np.median(per_dim):.3f}，"
          f"低于 0.6 的维度 {int(np.sum(np.array(per_dim) < 0.6))} 个")

    section("[6] 化学合理性（OAV 与来源）")
    oav = res["oav"]
    print(f"可计算 OAV 的化合物 {len(oav)} / {len(res['columns'])}")
    if len(oav):
        print(f"OAV 中位数 > 1 的化合物占比：{(oav['oav_median'] > 1).mean():.3f}")
        top_oav = oav.nlargest(8, "oav_median")[
            ["column", "threshold", "origin", "oav_median", "oav_frac_gt1"]]
        print("\n  OAV 最高 8 个化合物：")
        print(top_oav.round(3).to_string(index=False))

        print("\n  Top-3 化合物的 OAV 覆盖率：")
        top_cols = {res["columns"][j] for m in range(len(schema)) for j in res["top3"][m]}
        hit = oav[oav["column"].isin(top_cols)]
        other = oav[~oav["column"].isin(top_cols)]
        if len(hit) and len(other):
            print(f"    Top-3 中可算 OAV 的 {len(hit)} 个，"
                  f"OAV 中位数 {hit['oav_median'].median():.3f}")
            print(f"    非 Top-3 的 {len(other)} 个，"
                  f"OAV 中位数 {other['oav_median'].median():.3f}")

    section("[7] 落盘")
    out = config.TAB
    top3.to_csv(out / "q2_top3.csv", index=False, encoding="utf-8-sig")
    tab.to_csv(out / "q2_soft_power.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"column": res["columns"], "module": labels}).to_csv(
        out / "q2_modules.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(res["score"], columns=res["columns"], index=schema).to_csv(
        out / "q2_borda_score.csv", encoding="utf-8-sig")
    pd.DataFrame(res["freq"], columns=res["columns"], index=schema).to_csv(
        out / "q2_selection_freq.csv", encoding="utf-8-sig")
    if len(oav):
        oav.round(4).to_csv(out / "q2_oav.csv", index=False, encoding="utf-8-sig")
    print(f"→ {out}")

    lines = [
        "# S3 问题二报告\n",
        f"- WGCNA β = {beta}",
        f"- 模块数：{len(sizes)}",
        f"- bootstrap 次数：{args.boot}",
        f"- Top-3 入选频率 ≥ {config.STABLE_PI} 的比例：{stable_frac:.3f}",
        "",
        "## 软阈值选择",
        "",
        md_table(tab),
        "",
        "## 每维度 Top-3",
        "",
        md_table(top3),
        "",
        "## 化学合理性",
        "",
        (md_table(oav.nlargest(15, "oav_median").round(3)) if len(oav) else "无"),
    ]
    (config.REP / "s3_q2.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"报告 → {config.REP / 's3_q2.md'}")


if __name__ == "__main__":
    main()

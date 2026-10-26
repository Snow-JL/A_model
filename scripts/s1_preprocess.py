"""S1 预处理：尺度统一 → Winsor → LOD 判定 → 图引导插补 → 标准化。

产物写入 data/processed/，后续脚本直接读取，不重复计算。

运行：
    python scripts/s1_preprocess.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, io_data, preprocess          # noqa: E402
from src import schema as schema_mod                # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(t: str) -> None:
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


def main() -> None:
    data = io_data.load_all()
    sch = schema_mod.build_schema()
    section("[1] 预处理链")
    feats = preprocess.build_features(data)

    print(f"尺度判定：{(feats['scale_info']['kind'] == 'log10').sum()} 列对数 / "
          f"{(feats['scale_info']['kind'] == 'linear').sum()} 列线性原值")
    print(f"Winsor：截尾 {feats['winsor']['n_clipped']} 格"
          f"（阈值 |z| > {config.WINSOR_Z}，MAD 换算 1.4826）")

    ai = feats["align_info"]
    print(f"2.csv 对齐：对角 {ai['diag_ok']}，对称 {ai['sym_ok']}，"
          f"非对角 |r| 中位 {ai['offdiag_abs_median']:.3f}")

    section("[2] 2.csv 与 1.csv 的相关一致性抽检")
    chk = feats["align_check"]
    print(chk.round(4).to_string(index=False))
    print(f"  平均绝对偏差 {chk['abs_diff'].mean():.4f}（显著大于 0 说明两者顺序不同源）")

    section("[3] 缺失机制判定（置换检验，1000 次）")
    lod = feats["lod"]
    if len(lod):
        print(lod.round(4).to_string(index=False))
        n_lod = int((lod["verdict"] == "LOD_left_censored").sum())
        print(f"\n  判定为 LOD 左截断：{n_lod} / {len(lod)} 列")
    else:
        print("  无满足检验条件的缺失列")

    section("[4] 图引导插补")
    im = feats["impute"]
    print(im.round(4).to_string(index=False))
    print(f"合计填充 {im['n_filled'].sum()} 格，"
          f"其中含 LOD 钳制 {int((im['method'] == 'ridge_lod').sum())} 列")

    section("[5] 标准化")
    X = feats["X"]
    sp = feats["scale_params"]
    print(f"X 范围 [{X.min():.2f}, {X.max():.2f}]，中位 |z| = {np.median(np.abs(X)):.3f}")
    print(f"退化列（MAD=0 改用秩正态化）：{sp['n_degenerate']} 个")

    section("[6] 落盘")
    np.save(config.PROCESSED / "X.npy", feats["X"])
    np.save(config.PROCESSED / "X_log.npy", feats["X_log"])
    np.save(config.PROCESSED / "A.npy", feats["A"])
    np.save(config.PROCESSED / "Y.npy", data["Y"].to_numpy(float))

    meta = {
        "seed": config.SEED,
        "columns": feats["columns"],
        "schema": sch["columns"],
        "block_of": sch["block_of"],
        "groups": sch["groups"],
        "aroma": schema_mod.aroma_and_flavour(sch)[0],
        "flavour": schema_mod.aroma_and_flavour(sch)[1],
        "scorecard_rows": data["scorecard_rows"].tolist(),
        "scorecard_beers": data["scorecard_beers"].tolist(),
        "styles": data["ids"]["tasting_category_fine"].tolist(),
        "target_map": data["target_map"].to_dict("records"),
        "n_degenerate": sp["n_degenerate"],
        "impute": im.to_dict("records"),
        "lod": lod.to_dict("records") if len(lod) else [],
    }
    (config.PROCESSED / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for f in sorted(config.PROCESSED.glob("*")):
        print(f"  {f.name:16s} {f.stat().st_size/1024:8.1f} KB")

    lines = [
        "# S1 预处理报告\n",
        f"- 尺度：对数 {(feats['scale_info']['kind'] == 'log10').sum()} 列 / 线性 "
        f"{(feats['scale_info']['kind'] == 'linear').sum()} 列",
        f"- Winsor 截尾：{feats['winsor']['n_clipped']} 格（|z| > {config.WINSOR_Z}）",
        "",
        "## 相关阵对齐抽检",
        "",
        chk.round(4).to_markdown(index=False),
        "",
        f"平均绝对偏差 {chk['abs_diff'].mean():.4f}",
        "",
        "## 缺失机制",
        "",
        (lod.round(4).to_markdown(index=False) if len(lod) else "无"),
        "",
        "## 插补",
        "",
        im.round(4).to_markdown(index=False),
        "",
        f"合计填充 {im['n_filled'].sum()} 格",
        "",
        "## 标准化",
        "",
        f"- X 范围 [{X.min():.2f}, {X.max():.2f}]",
        f"- 退化列 {sp['n_degenerate']} 个（改用秩正态化）",
    ]
    (config.REP / "s1_preprocess.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"报告 → {config.REP / 's1_preprocess.md'}")


if __name__ == "__main__":
    main()

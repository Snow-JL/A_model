"""数据读取与对齐校验。

约定：所有随机变量以 beer 序号（1..250）为对齐键。
4.csv 的 beer_id 列有损坏（250 行仅 96 个唯一值，与 1.csv 一致率 14.8%），
因此一律按 beer 对齐，beer_id 只作核对，不参与任何 join。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ID_COLS, RAW, SCORECARD_BEERS


# ------------------------------------------------------------------ 工具
def _norm(name: str) -> str:
    """列名规范化：去 X 前缀、统一小写、去下划线，用于跨文件模糊匹配。"""
    s = name.strip().lower()
    if s.startswith("x") and len(s) > 1 and s[1].isdigit():
        s = s[1:]
    return s.replace("_", "").replace(" ", "")


def _check_beer_order(a: np.ndarray, b: np.ndarray, what: str) -> None:
    if not np.array_equal(a, b):
        raise AssertionError(f"{what}: beer 顺序不一致，禁止继续")
    if not np.array_equal(np.sort(a), np.arange(1, len(a) + 1)):
        raise AssertionError(f"{what}: beer 不是 1..n 的排列")


# ------------------------------------------------------------------ 读取
def load_compounds() -> tuple[pd.DataFrame, pd.DataFrame]:
    """1.csv → (标识表, 化合物矩阵)。化合物为 log10 浓度。"""
    df = pd.read_csv(RAW / "1.csv")
    ids = df[ID_COLS].copy()
    X = df.drop(columns=ID_COLS)
    assert X.shape[1] == 231, f"化合物列数异常: {X.shape[1]}"
    return ids, X


def load_sensory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """4.csv → (标识表, 50 维连续强度矩阵)。逐列已标准化。"""
    df = pd.read_csv(RAW / "4.csv")
    ids = df[["beer", "tasting_category_fine"]].copy()
    Y = df.drop(columns=["beer", "beer_id", "tasting_category_fine"])
    assert Y.shape[1] == 50, f"感官列数异常: {Y.shape[1]}"
    return ids, Y


def load_sensory_fine() -> tuple[pd.DataFrame, pd.DataFrame]:
    """7.csv → 186 维细粒度感官（62 描述符 × 3 模态），作为辅助标签。"""
    df = pd.read_csv(RAW / "7.csv")
    ids = df[["beer", "tasting_category_fine"]].copy()
    Y = df.drop(columns=ID_COLS)
    # 3 个全空列直接剔除，避免污染辅助任务
    dead = [c for c in Y.columns if Y[c].isna().all()]
    Y = Y.drop(columns=dead)
    return ids, Y, dead


def load_corr_compounds() -> pd.DataFrame:
    """2.csv → 231×231 Pearson 相关阵。"""
    R = pd.read_csv(RAW / "2.csv", index_col=0)
    assert R.shape == (231, 231), f"2.csv 规模异常: {R.shape}"
    assert np.allclose(R.values, R.values.T, atol=1e-10), "2.csv 不对称"
    assert np.allclose(np.diag(R.values), 1.0, atol=1e-10), "2.csv 对角非 1"
    return R


def load_thresholds() -> pd.DataFrame:
    """8.xlsx 感官阈值表（261 行），提供 OAV 判据。"""
    return pd.read_excel(RAW / "8.xlsx", sheet_name="Sensory thresholds")


def load_sensory_names() -> pd.DataFrame:
    """8.xlsx 感官名称映射（101 行，Trained panel 50 + Ratebeer 51）。"""
    return pd.read_excel(RAW / "8.xlsx", sheet_name="Sensory names")


def load_calibrated() -> pd.DataFrame:
    """8.xlsx 校准化合物表（250×73），用于核验 log 底数与物理单位。"""
    return pd.read_excel(RAW / "8.xlsx", sheet_name="Calibrated chemical dataset")


def load_scorecard_schema() -> tuple[list[str], pd.DataFrame]:
    """3.xlsx summary → (50 个输出维度的规范列名, 空白模板)。"""
    sm = pd.read_excel(RAW / "3.xlsx", sheet_name="summary")
    drop = {"Beer", "comments"}
    cols = [c for c in sm.columns if c not in drop]
    assert len(cols) == 50, f"summary 维度数异常: {len(cols)}"
    return cols, sm


def load_scorecard_sheets() -> dict[int, pd.DataFrame]:
    """3.xlsx 的 12 张评分卡（空白模板，仅含 0-6 刻度标签）。"""
    out = {}
    for k in SCORECARD_BEERS:
        out[k] = pd.read_excel(RAW / "3.xlsx", sheet_name=f"beer_{k}", header=None)
    return out


# ------------------------------------------------------------------ 列映射
def target_mapping() -> pd.DataFrame:
    """建立 4.csv 列名 ↔ summary 输出名 的一一映射。

    三级匹配：精确 → 规范化（去 X 前缀/下划线/大小写）→ 模糊（相似度 > 0.85）。
    已知差异：4.csv 的 'A_esters_fruity' 对应 summary 的 'A_ester_fruity'（少一个 s），
    'X4vg' 对应 '4vg'。
    返回 DataFrame[source, target, match]。
    """
    import difflib

    cols, _ = load_scorecard_schema()
    _, Y = load_sensory()
    src = list(Y.columns)

    rows: list[dict] = []
    used: set[str] = set()

    for c in src:                                   # 1. 精确
        if c in cols and c not in used:
            rows.append({"source": c, "target": c, "match": "exact"})
            used.add(c)

    done = {r["source"] for r in rows}
    for c in [x for x in src if x not in done]:     # 2. 规范化相等
        for t in cols:
            if t not in used and _norm(c) == _norm(t):
                rows.append({"source": c, "target": t, "match": "normalized"})
                used.add(t)
                break

    done = {r["source"] for r in rows}
    for c in [x for x in src if x not in done]:     # 3. 模糊
        cand = sorted(
            ((t, difflib.SequenceMatcher(None, _norm(c), _norm(t)).ratio())
             for t in cols if t not in used),
            key=lambda x: -x[1],
        )
        if not cand or cand[0][1] < 0.85:
            raise AssertionError(f"无法映射的列: {c}，最佳候选 {cand[:2]}")
        rows.append({"source": c, "target": cand[0][0], "match": "fuzzy"})
        used.add(cand[0][0])

    m = pd.DataFrame(rows)
    assert len(m) == 50, f"映射对数异常: {len(m)}"
    assert m["source"].is_unique and m["target"].is_unique
    return m


# ------------------------------------------------------------------ 汇总
def load_all() -> dict:
    """一次性载入并完成全部对齐校验。任何断言失败即中止。"""
    ids_c, X = load_compounds()
    ids_s, Y = load_sensory()
    ids_f, Y7, dead7 = load_sensory_fine()

    _check_beer_order(ids_c["beer"].to_numpy(), ids_s["beer"].to_numpy(), "1.csv vs 4.csv")
    _check_beer_order(ids_c["beer"].to_numpy(), ids_f["beer"].to_numpy(), "1.csv vs 7.csv")

    score_beers = ids_c[ids_c["beer"].isin(SCORECARD_BEERS)].sort_values("beer")
    assert len(score_beers) == 12

    return {
        "ids": ids_c,
        "X": X,                     # 250×231，log10 浓度
        "Y": Y,                     # 250×50，连续标准化强度
        "Y7": Y7,                   # 250×183（剔除 3 个全空列）
        "dead7": dead7,
        "R_comp": load_corr_compounds(),
        "thresholds": load_thresholds(),
        "sensory_names": load_sensory_names(),
        "calibrated": load_calibrated(),
        "schema": load_scorecard_schema()[0],
        "target_map": target_mapping(),
        "scorecard_rows": score_beers.index.to_numpy(),
        "scorecard_beers": score_beers["beer"].to_numpy(),
        "styles": score_beers["tasting_category_fine"].tolist(),
    }

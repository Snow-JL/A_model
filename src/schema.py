"""3.xlsx 评分卡解析：确定 50 个感官维度的块归属与层次结构。

实测结构（beer_1，101×33）
--------------------------
左侧 Aroma 块（标签列 1 与 4，得分列 16），右侧 Flavour 块（标签列 17 与 20，得分列 32）。
每张卡含 0–6 刻度表头若干行，但所有得分格均为 0，即空白模板，不含任何真值。

块归属结果
----------
Aroma 侧 26 维 = A_ 前缀 15 维（Malt 5 + Hops 5 + Esters 5）+ 通用 11 维
Flavour 侧 24 维 = F_ 前缀 15 维 + 通用 9 维
合计 50 维，与 4.csv 的 50 列一一对应。

注意：评分卡中 Acetaldehyde 在两侧各出现一次，但 4.csv 无此列，
说明该维度在设计表中保留、在实测数据中未测。论文需说明。
"""
from __future__ import annotations

import pandas as pd

from .config import RAW, SECTION_TITLES

# 评分卡标签 → summary 列名。同名标签在 Aroma / Flavour 两侧指向不同列，
# 因此键为 (block, label)。
LABEL_TO_COLUMN: dict[tuple[str, str], str] = {
    # ---- Aroma 侧：A_ 前缀 15 维
    ("Aroma", "Malt"): "A_malt_all",
    ("Aroma", "Cereals, flour, grains"): "A_malt_grain",
    ("Aroma", "Bread crust"): "A_malt_bread",
    ("Aroma", "Caramel"): "A_malt_cara",
    ("Aroma", "Smoked, burnt toast"): "A_malt_burn",
    ("Aroma", "Hops"): "A_hops_all",
    ("Aroma", "Citrus"): "A_hops_citrus",
    ("Aroma", "Tropical fruit"): "A_hops_tropical",
    ("Aroma", "Noble hops (dried herbs, dried grass, hay)"): "A_hops_noble",
    ("Aroma", "Pine, woody, resinous"): "A_hops_woody",
    ("Aroma", "Esters"): "A_esters_all",
    ("Aroma", "Ethyl acetate, solvent, nail polish"): "A_esters_ethac",
    ("Aroma", "Banana, pear, candy"): "A_esters_isoaa",
    ("Aroma", "Flowers, roses"): "A_esters_flower",
    ("Aroma", "Sweet, fruity, apples"): "A_ester_fruity",
    # ---- Aroma 侧：通用 11 维
    ("Aroma", "Body"): "body",
    ("Aroma", "Bitterness"): "bitternes",
    ("Aroma", "Carbonation"): "co2",
    ("Aroma", "Aftertaste"): "aftertaste",
    ("Aroma", "DMS (cooked cabbage, corn)"): "dms",
    ("Aroma", "Metallic"): "metallic",
    ("Aroma", "4VG (smoky, clove, medicinal, dentist)"): "4vg",
    ("Aroma", "Orange"): "orange",
    ("Aroma", "Coriander"): "coriander",
    ("Aroma", "Clove/anise"): "clove",
    ("Aroma", "Overall"): "overall",
    # ---- Flavour 侧：F_ 前缀 15 维
    ("Flavour", "Malt"): "F_malt_all",
    ("Flavour", "Cereals, flour, grains"): "F_malt_grain",
    ("Flavour", "Bread crust"): "F_malt_bread",
    ("Flavour", "Caramel"): "F_malt_cara",
    ("Flavour", "Smoked, burnt toast"): "F_malt_burn",
    ("Flavour", "Hops"): "F_hops_all",
    ("Flavour", "Citrus"): "F_hops_citrus",
    ("Flavour", "Tropical fruit"): "F_hops_tropical",
    ("Flavour", "Noble hops (dried herbs, dried grass, hay)"): "F_hops_noble",
    ("Flavour", "Pine, woody, resinous"): "F_hops_woody",
    ("Flavour", "Esters"): "F_esters_all",
    ("Flavour", "Ethyl acetate, solvent, nail polish"): "F_esters_ethac",
    ("Flavour", "Banana, pear, candy"): "F_esters_isoaa",
    ("Flavour", "Flowers, roses"): "F_esters_flower",
    ("Flavour", "Sweet, fruity, apples"): "F_ester_fruity",
    # ---- Flavour 侧：通用 9 维
    ("Flavour", "Alcohol"): "alcohol",
    ("Flavour", "Sweetness"): "sweetness",
    ("Flavour", "Acidity/sourness"): "acidity",
    ("Flavour", "Trans-2-nonenal (wet cardboard, old attic, papery)"): "t2n",
    ("Flavour", "Diacetyl (buttery, caramel/butterscotch)"): "diacetyl",
    ("Flavour", "Stale hops (cheesy, roquefort)"): "stale_hops",
    ("Flavour", "Lactic acid"): "lactic",
    ("Flavour", "Acetic acid"): "acetic",
    ("Flavour", "Barnyard/horsey"): "barnyard",
}

# 评分卡中存在但 4.csv 未测量的维度
UNMEASURED = {
    ("Aroma", "Acetaldehyde (green apples, peel)"),
    ("Flavour", "Acetaldehyde (green apples, peel)"),
}

_SKIP_LABELS = {"Date", "Name:", "Beer#:", "Aroma", "Flavour", "Comments:",
                "Not present", "Predominant/overwhelming"}

# 与感官来源分类对应的泛化维度归属（问题二化学合理性使用）
FAULT_COLUMNS = ["dms", "diacetyl", "4vg", "t2n", "stale_hops", "metallic",
                 "lactic", "acetic", "barnyard"]


def extract_entries(sheet: pd.DataFrame) -> list[dict]:
    """按出现顺序抽取评分卡中的维度条目。"""
    entries: list[dict] = []
    for i in range(sheet.shape[0]):
        for block, cols in (("Aroma", (1, 4)), ("Flavour", (17, 20))):
            for c in cols:
                if c >= sheet.shape[1]:
                    continue
                v = sheet.iat[i, c]
                if not isinstance(v, str):
                    continue
                s = v.strip()
                if not s or s in _SKIP_LABELS:
                    continue
                entries.append({
                    "row": i,
                    "block": block,
                    "label": s,
                    "is_section": s in SECTION_TITLES,
                })
                break
    return entries


def build_schema() -> dict:
    """构建 50 维的块归属、层次结构与校验信息。"""
    sheet = pd.read_excel(RAW / "3.xlsx", sheet_name="beer_1", header=None)
    entries = extract_entries(sheet)

    pairs: list[tuple[str, str]] = []
    unmeasured: list[tuple[str, str]] = []
    unmapped: list[tuple[str, str]] = []
    for e in entries:
        if e["is_section"]:
            continue
        key = (e["block"], e["label"])
        if key in LABEL_TO_COLUMN:
            pairs.append((e["block"], LABEL_TO_COLUMN[key]))
        elif key in UNMEASURED:
            unmeasured.append(key)
        else:
            unmapped.append(key)

    if unmapped:
        raise AssertionError(f"评分卡存在无法映射的维度：{unmapped}")

    # 校验：与 4.csv 的列集合一致
    from .io_data import load_sensory, load_scorecard_schema
    _, Y = load_sensory()
    summary, _ = load_scorecard_schema()
    mapped_cols = [col for _, col in pairs]
    if sorted(mapped_cols) != sorted(summary):
        missing = set(summary) - set(mapped_cols)
        extra = set(mapped_cols) - set(summary)
        raise AssertionError(f"块归属与 summary 不一致：缺 {missing}，多 {extra}")

    # 重排到 summary 顺序
    order = {c: i for i, c in enumerate(summary)}
    pairs.sort(key=lambda t: order[t[1]])

    block_of = {col: blk for blk, col in pairs}
    groups = _hierarchy(summary)

    return {
        "columns": summary,
        "block_of": block_of,
        "n_aroma": sum(1 for c in summary if block_of[c] == "Aroma"),
        "n_flavour": sum(1 for c in summary if block_of[c] == "Flavour"),
        "groups": groups,
        "unmeasured": unmeasured,
        "n_entries": len(entries),
        "pairs": pairs,
    }


def _hierarchy(schema: list[str]) -> list[dict]:
    """总—分层结构：Malt / Hops / Esters 的 overall 与其四个子项。"""
    groups = []
    for block in ("A", "F"):
        for fam in ("malt", "hops", "ester"):
            members = [c for c in schema if c.startswith(f"{block}_{fam}")]
            totals = [c for c in members if c.endswith("_all")]
            parts = [c for c in members if not c.endswith("_all")]
            if len(totals) == 1 and parts:
                groups.append({"total": totals[0], "parts": parts})
    return groups


def aroma_and_flavour(schema: dict) -> tuple[list[str], list[str]]:
    """返回 Aroma 套与 Flavour 套的列名列表。"""
    cols = schema["columns"]
    blk = schema["block_of"]
    return ([c for c in cols if blk[c] == "Aroma"],
            [c for c in cols if blk[c] == "Flavour"])

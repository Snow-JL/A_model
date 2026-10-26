"""全局配置：路径、随机种子、超参数、领域常量。

所有模块只从这里取路径与超参数，不硬编码。
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- 路径
ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT.parent
RAW = PROJ / "2026年MathorCup大数据挑战赛-赛道A初赛" / "附件"

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
REP = OUT / "reports"
for _d in (PROCESSED, FIG, TAB, REP):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 随机性
SEED = 20260826

# ---------------------------------------------------------------- 数据规模
ID_COLS = ["beer", "beer_id", "tasting_category_fine"]
N_SAMPLES = 250
N_COMPOUND = 231
N_SENSORY = 50
N_SCORECARD = 12
SCORECARD_BEERS = list(range(1, N_SCORECARD + 1))

# ---------------------------------------------------------------- 预处理
# 缺失机制判定：Welch t 检验的显著性水平
LOD_ALPHA = 0.05
# 图引导插补：邻居阈值 τ 与岭回归正则网格
IMPUTE_TAU = 0.5
IMPUTE_LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
IMPUTE_CV_FOLD = 5
# 异常值：稳健 z 分数阈值（MAD 换算系数 1.4826）
WINSOR_Z = 4.0
MAD_TO_SD = 1.4826

# ---------------------------------------------------------------- 标签还原
# 零点锚定所用的低分位（避免极端最小值受噪声影响）
ZERO_ANCHOR_QUANTILE = 0.02
# 等级数
N_LEVEL = 7

# ---------------------------------------------------------------- PLS
PLS_COMP_GRID = tuple(range(2, 21))

# ---------------------------------------------------------------- 有序回归
ORD_REG_L2 = 1e-2       # 系数 L2 正则
ORD_MAX_ITER = 500

# ---------------------------------------------------------------- WGCNA
WGCNA_BETA_GRID = tuple(range(1, 21))
WGCNA_SCALE_FREE_R2 = 0.85     # 无标度准则阈值
WGCNA_TOM_MERGE = 0.80         # 模块特征值相关合并阈值
WGCNA_MIN_MODULE = 8           # 最小模块规模

# ---------------------------------------------------------------- 三法 + Borda
N_BOOT = 500                   # bootstrap 次数
TOP_K = 3                      # 每维度取前 K
STABLE_PI = 0.6                # 入选频率稳定阈值
SHAP_CANDIDATE = 20            # 进入互信息计算的候选数

# ---------------------------------------------------------------- 问题三
COPULA_SHRINK_KAPPA = 5.0      # 均值收缩强度
COPULA_SHRINK_ALPHA = 5.0      # 协方差收缩强度
N_GEN = 200                    # 每风格生成样本数
GAMMA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# ---------------------------------------------------------------- 领域常量
# 题面指定的缺陷化合物：标签 → 1.csv 列名。
# Stale hops 在 1.csv 中无独立化合物列（它只作为感官维度存在于 4.csv），
# 因此只进入感官层面讨论，不参与化合物钳制。
DEFECTS = {
    "DMS": "DMS",
    "Diacetyl": "X2.3.butanedione",
    "4VG": "X4VG",
    "trans-2-nonenal": "X.E..2.nonenal",
}
DEFECTS_SENSORY_ONLY = {"Stale hops": "stale_hops"}

# 缺陷约束的经验上限分位：低于 0.5 会把大部分正常样本也钳掉
DEFECT_QUANTILE = 0.75

# 理化参数（非挥发性风味化合物）。题面称之为"保留原始量纲"的少数列。
# 问题二要求的是"关键因素化合物"，故 Top-3 主结果从风味化合物中选，
# 理化参数单独报告，避免 color/pH 这类协变量占据名额。
PHYSICOCHEM = {
    "pH", "color.EBC.", "bitterness", "protein.g.L.", "iron", "SO2.mg.L.",
    "lactic_acid.mg.L.", "acetic_acid.g.L.", "ammonia.mg.L.",
    "betaglucan.mg.L.", "glycerol", "GFS", "kcalperc", "unfermentables",
    "CO2.PSI.", "ethanol..v.v.", "esters_sum", "aroma_hops_sum",
    "acids_sum", "sulfur_sum",
}

# 3.xlsx 中 Aroma / Flavour 两块共用的 "Faults" 与 "Spiced/Lambic" 标题行
SECTION_TITLES = {"Faults", "Spiced beers", "Lambic/Geuze"}

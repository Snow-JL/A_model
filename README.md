# A 题建模工作区

赛道 A：啤酒酿造化合物分析与啤酒风味预测。

代码以 numpy 自实现全部算法，仅依赖 pandas / openpyxl（读表）与 matplotlib（可选出图）。
不依赖 scikit-learn / scipy。

## 运行顺序

本目录可整体改名或移动：`src/config.py` 与各 `scripts/*.py` 的路径全部由
`Path(__file__).resolve().parents[1]` 反推，不含硬编码绝对路径。前提是
`A_model/` 与 `2026年MathorCup大数据挑战赛-赛道A初赛/附件/` 保持同级。

在 `A_model/` 目录下依次执行（PowerShell，本机实测可用）：

```powershell
cd D:\SnowJL\2026_MathorCup\A_model
$env:PYTHONIOENCODING = 'utf-8'     # 否则中文报告输出可能编码报错
D:\miniconde\python.exe scripts\s1_preprocess.py   # 必须先跑，产物被 s2–s4 复用
D:\miniconde\python.exe scripts\s2_q1.py
D:\miniconde\python.exe scripts\s3_q2.py --boot 40 # 正式出图建议 ≥200
D:\miniconde\python.exe scripts\s4_q3.py
D:\miniconde\python.exe scripts\s0_audit.py        # 审计，随时可跑
D:\miniconde\python.exe scripts\verify_ordinal.py  # 方法正确性证据
```

s1 的产物被 s2–s4 复用，所以 s1 必须先跑一次。

**解释器必须显式指定。** 本机直接敲 `python` 走的是 `E:\SnowJL\python.exe`（3.13），
里面没有 numpy/pandas，会直接 `ModuleNotFoundError`。`D:\miniconde\python.exe` 已装
numpy / pandas / openpyxl，是当前可用的解释器；若换机器，换成任何装了这三件套的解释器即可。

参考耗时（本机 i9-14900HX，`--boot 40`）：s0 约 2 s、s1 约 1 s、s2 约 110 s、
s3 约 40 s（`--boot 100` 约 79 s）、s4 约 1.5 s、verify 约 0.5 s。

依赖：numpy / pandas / openpyxl（读 xlsx）。Python 需 3.10+；
Windows 控制台建议先 `set PYTHONIOENCODING=utf-8`（或 PowerShell
`$env:PYTHONIOENCODING='utf-8'`）以免中文报告输出报编码错。

## 模块职责

### src/ —— 算法与流程

| 文件 | 职责 |
| --- | --- |
| `config.py` | 唯一的配置源：路径、随机种子、全部超参数、缺陷化合物与理化参数名单 |
| `utils.py` | 基础数值：平均秩、逆正态分位、Spearman、二次加权 Kappa、sigmoid、Winsor 用 MAD |
| `io_data.py` | 读 8 个附件；按 beer 序号对齐（禁用损坏的 beer_id）；三级列名匹配（精确→规范化→模糊） |
| `schema.py` | 解析 3.xlsx 评分卡，得到 Aroma 26 维 / Flavour 24 维的块归属与 6 组总—分层级 |
| `preprocess.py` | 尺度判定（对数 vs 线性）、Winsor、LOD 置换检验、图引导岭回归插补、稳健标准化 |
| `label_recover.py` | 问题一的 0–6 标签还原：零点+极值锚定 / 等频 7 分位两套方案 |
| `pls.py` | NIPALS 偏最小二乘，含成分数交叉验证选择 |
| `ordinal.py` | 累积链接有序回归（proportional odds），解析梯度 + Adam |
| `metrics.py` | MMD²、能量距离、Jaccard、k-NN 覆盖率、记忆化率、逐维度分级报告 |
| `q1.py` | 问题一全流程：还原→PLS→有序回归→层次修正→多重插补→5 折 CV |
| `q2.py` | 问题二全流程：WGCNA 构图→谱聚类模块→三法互补→Borda→bootstrap→OAV |
| `q3.py` | 问题三全流程：经验 CDF→收缩高斯 copula→缺陷钳制→最优传输迁移→γ 路径→评价 |

### scripts/ —— 可执行入口

| 文件 | 产出 |
| --- | --- |
| `s0_audit.py` | `reports/s0_audit.md`：混合尺度、格点不成立、LOD 堆积、维度结构的证据 |
| `s1_preprocess.py` | `data/processed/*.npy, meta.json`；`reports/s1_preprocess.md` |
| `s2_q1.py` | `tables/q1_grade_12x50.csv`、`q1_grade_250x50.csv`、`q1_per_dim_metrics.csv`、`q1_entropy.csv` |
| `s3_q2.py` | `tables/q2_top3.csv`、`q2_modules.csv`、`q2_borda_score.csv`、`q2_selection_freq.csv`、`q2_oav.csv` |
| `s4_q3.py` | `tables/q3_evaluation.csv`、`q3_generated_*.csv`、`q3_transferred_*.csv` |
| `verify_ordinal.py` | 有序回归的解析梯度与数值梯度对比、有/无信号数据下的拟合检验 |

## 三条不可违反的数据约定

1. **合并键是 beer 序号，不是 beer_id。** 4.csv 的 beer_id 250 行仅 96 个唯一值，与 1.csv 一致率 14.8%，已在 `io_data.py` 中弃用。
2. **1.csv 是混合尺度。** 175 列含负值（log10 浓度），56 列全非负（线性原值）。已验证：与 8.xlsx 校准表比对，37 列满足 x=log10(y)、33 列满足 x=y。任何新增代码不得假设统一取对数。
3. **4.csv 不存在格点结构。** 50 个维度的相邻差值不构成公共间距（一致性最高 0.076），因此不能用"评委平均分的格点间距"反解标准差，只能用零点锚定 + 分位/极值锚定。

## 当前已知局限

- 标签还原无真值可校验，anchor 与 quantile 两种方案的完全一致率仅 0.28（QWK 0.67）。这是全篇最大的不确定性来源。
- 问题二的 Top-3 稳定性偏低：入选频率 ≥ 0.6 的化合物仅占 15%，48 个维度的平均频率低于 0.6。属真实结论，需在论文中作为主要发现之一呈现，而非掩盖。
- OAV 检验未通过预期：Top-3 化合物的 OAV 中位数 0.20，非 Top-3 为 0.12，区分度不足。可能原因：1.csv 浓度单位与阈值表单位未能对齐（DMS 为 ug/L、4VG 为 mg/L、Diacetyl 无单位）。
- 缺陷约束的阈值改用训练集 75% 分位，而非 8.xlsx 阈值，原因同上（单位不可对齐）。

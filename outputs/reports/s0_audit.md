# S0 数据审计报告

## 规模
- 化合物：250 × 231
- 感官维度：50（Aroma 26 + Flavour 24）

## 对数底数

| kind   |   count |
|:-------|--------:|
| log10  |      37 |
| linear |      33 |

结论：混合尺度。37 列满足 x = log10(y)（残差 ~1e-10），33 列满足 x = y（残差 0）。

## 格点结构

- 格点一致性 ≥ 0.90 的维度：0 / 50
- 最高一致性：0.076
- 唯一值中位：229

结论：格点不成立，改用零点锚定 + 极值锚定。

## 缺失

| column             |   n_missing |   frac_missing |
|:-------------------|------------:|---------------:|
| amyl.acetate       |          27 |          0.108 |
| aroma_hops_sum     |          13 |          0.052 |
| methylpyrazine     |          12 |          0.048 |
| X2.3.butanedione   |           9 |          0.036 |
| X3.methylbutanal   |           7 |          0.028 |
| ethyl.nicotinate   |           2 |          0.008 |
| ethyl.isobutyrate  |           1 |          0.004 |
| furfural           |           1 |          0.004 |
| phenylacetaldehyde |           1 |          0.004 |

## 维度结构

- Aroma 26 维，Flavour 24 维，合计 50
- 层次结构 6 组
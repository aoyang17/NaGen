# NaGen Data Card v1 — Na–TM–P–O 阴极数据集

版本：v1（Phase 0，2026-08-18）
原始输入：`/mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl`（严格只读）
机器可读审计：`data/audit/audit_summary.json`、`data/audit/label_provenance.json`、`data/audit/records.csv`

## 1. 数据规模与解析

- 记录数：**17,774**；解析失败：**0**
- 文件大小：约 506 MB；每行一个 JSON 对象
- 唯一 reduced formula：**564**
- 唯一结构（指纹去重 + StructureMatcher 抽样复核）：**17,774**（无精确重复）
- 结构家族（按 `material_id` 前缀）：pyro（焦磷酸盐 P2O7）8,115、mix 3,765、fluoro（氟磷酸盐）2,481、ortho（正磷酸盐 PO4）1,926、nitro（氮磷酸盐）971、NaSICON 234、meta（偏磷酸盐 PO3）206、oxy 76

## 2. 字段与缺失率

| 字段 | 存在率 | 说明 |
|---|---|---|
| `material_id` / `formula` / `structure` | 100% | 标识、化学式、pymatgen 结构 |
| `optimized_energy_per_atom` | 100% | UMA MLFF 势能/原子，**非 DFT**（见标签来源表） |
| `energy_above_hull` | 0% | 全部缺失 |
| `energy_per_atom` / `formation_energy_per_atom` | 0% | 全部缺失 |
| `num_NaO3..12`、`num_TMO3..12`、`num_PO3..12`（30 个） | 键存在但 **100% 为 null** | 配位计数字段为空，需自行重算 |
| `error_msg` / `traceback` / `optimized_structure` | 35.5%（6,304 条） | MLFF 优化中轨迹写入 I/O 错误（见 §7） |

## 3. 化学组成

- 元素总计数：O 1,073,957；P 300,888；Na 156,439；Fe 44,898；Mn 42,982；Ti 30,061；V 30,061；F 17,269；N 3,884；C 176
- 元素覆盖（记录数）：O/P 17,774（100%）；Na 17,557（98.8%，**217 条无 Na**，即脱钠相）；Ti 7,831；V 7,831；Fe 5,103；Mn 4,840；F 2,481；N 971；C 88
- 每条记录至少含 Fe/Mn/Ti/V 之一（TM 缺失记录 = 0）
- 阴离子骨架（reduced formula 模式）：P2O7 8,115 条/135 式；P4O15 2,417/51；PO4 2,251/108；P2O8 174/15；其余 4,817/255（含 NO3/CO3/F 混合）
- **重要**：数据集并非纯 PO4 正磷酸盐，**焦磷酸盐（P2O7）占多数**；PO4 完整性（每个 P 均 4 配位 O）覆盖 16,798 条（94.5%）

## 4. 结构与尺寸

- 原子数（原始 = primitive ≈ niggli，三组分布几乎一致）：min 20，median **91**，p10 43，p90 174，max 184
- primitive（spglib `find_primitive`，symprec 1e-3）：中位 91；**>20 原子占 99.9%，>40 占 91.8%，>80 占 57.5%**
- symprec 敏感性（1e-3/1e-2/1e-1，120 条抽样）：约简量不变，SG P1 占比 94%/94%/92% → **大晶胞是真实的 Na 空位有序超胞**，不是容差假象
- 空间群：P1 16,256（91.5%）；其余少量 P21/P2/C2/R3/R32 等
- 体积/原子：min 11.0，median 13.4，p90 14.8，max 19.7 Å³/atom
- 无序/部分占位：**0 条**
- 化学式字段与结构组成不一致：**0 条**

## 5. 配位与局部环境（自算，替代全部为 null 的 num_* 字段）

- P–O 配位（cutoff 2.15 Å）：Q0（孤立 PO4）4,805；Q1（单桥氧，P2O7）8,115；Q2 206；mixed_Q 3,677；irregular_PO 971
- TM–O 配位（cutoff 2.4 Å）：CN=6 占主导（110,059 个位点）、CN=4 25,337、CN=5 12,185、CN=3 128（94 条记录）、CN=7 293（184 条记录）
- 最短键距（全量，Å）：P–O min 1.426 / median 1.499；TM–O min 1.592 / median 1.946；Na–O min 2.108 / median 2.271；O–O min 1.364（异常）/ median 2.452
- 共价半径比（min_cov_ratio，每记录最紧 pair）：min 0.675，p10 0.816，median 0.860

## 6. 能量

- `optimized_energy_per_atom`：全部 17,774 条有值，范围 **−7.841 ~ −6.061 eV/atom**，mean −6.944，median −6.922
- 能量口径：UMA（FAIRChem `omat`）MLFF 两阶段松弛（FIRE 晶胞+原子 fmax 0.10、BFGS 原子 fmax 0.01）后的势能/原子。**不是 DFT，无泛函/赝势/U 元数据；不能直接用于凸包或形成能**

## 7. 已知标签问题（关键）

- **6,304 条（35.5%）带 I/O 错误**：`error_msg = [Errno 5] Input/output error`，`traceback` 显示崩溃发生在 ASE 轨迹写盘（`ulm.py fd.seek`），即 MLFF 松弛过程中轨迹写入失败
- 这些记录仍含 `optimized_energy_per_atom` 与 `optimized_structure`（崩溃时几何，相对 `structure` 平均 RMSD 0.39 Å、体积 −1.2%），**收敛状态不可验证**
- 11,470 条无错误记录：`structure` = 松弛后几何（已与原始 parent 文件比对确认 ≠ parent），但文件内**无残差力/收敛标志**，fmax≤0.01 的收敛仅为管线设定假设
- 结论：`optimized_energy_per_atom` 是"MLFF 近似能量"，其中 35.5% 为中断松弛的近似值；`energy_above_hull` 全缺 → 稳定性标签（hull）不存在，是 Phase 1 首要标签缺口

## 8. 异常清单

| 异常 | 记录数 | 说明 |
|---|---|---|
| 含 C/N（超出 Na–TM–P–O 目标集） | 1,059（C 88，N 971） | 建议第一阶段剔除或单列 |
| 含 F | 2,481 | 氟磷酸盐家族，保留与否需领域决策 |
| 无电荷中性氧化态猜测 | 4,010（22.6%） | 多为 F/N 及特殊 Na:TM 配比 |
| irregular_PO（P 4 配位比例 <0.75） | 971 | 与 nitro/混合骨架重叠 |
| O 未成键比例 >5% | 761 | 需人工复核 |
| 异常近邻（<0.75×共价半径和） | 199（215 对） | 抽样显示几乎全部为短 V=O/Ti=O（~1.6 Å，化学上合理），非重叠错误 |
| 非正体积 / formula 不匹配 / 无序 | 0 / 0 / 0 | — |

## 9. 划分（split v1）

- 分组键：`reduced_formula`（564 组；prototype 级指纹全唯一，无法再细分）
- 比例：train 14,219（80.0%）、val 1,777（10.0%）、test 1,778（10.0%）
- **化学式泄漏 = 0**（同一 formula 的所有记录只出现在一个 split）
- 每个 split 的 Q 标签分布见 `data/splits/split_v1.json`
- 文件：`data/splits/split_v1.json`、`data/splits/split_by_idx.csv`

## 10. 派生数据与谱系

- `data/audit/records.csv`：17,774 行 × 63 列逐条特征（约 92 MB）
- `data/audit/canonical_v1.jsonl`：canonical 记录（raw_id + reduced_formula + original_structure + primitive_structure + flags，约 724 MB）
- `data/audit/audit_summary.json` / `provenance.json` / `label_provenance.json` / `duplicates.json` / `roundtrip_report.json`
- canonical_id：`NaGen-00000..NaGen-17773`（= 行号 idx，稳定映射 raw_id ↔ canonical_id）

## 11. 对后续阶段的直接含义

1. **MatterGen 原子数约束失效**：99.9% 的 primitive cell >20 原子（MP-20 类预训练上限），不能直接按官方缓存流程微调；需要决策：a) 以约简后的母体框架小晶胞为目标生成，再枚举 Na 空位；b) 放宽原子数上限与大 batch 显存预算
2. **能量标签不可用于 hull 条件训练**；35.5% 的 MLFF 能量来自中断松弛
3. **配位标签全部需自算**（本审计已实现等价计算）
4. **化学体系决定**：是否保留 F（2,481 条）、是否剔除 C/N（1,059 条）是 Phase 1 数据集的第一个冻结决策
5. 焦磷酸盐/混合骨架占主导 → 生成目标不应只定义 PO4

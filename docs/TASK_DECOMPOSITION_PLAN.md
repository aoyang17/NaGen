# NaGen 任务分解计划（Task Decomposition）— MVP 版

日期：2026-08-18
状态：Phase 1 计划草案（待评审后执行）
关联：`DATA_CARD_v1.md`、`HARD_CONSTRAINTS_SPEC.md`、`PHASE0_REPORT.md`、
`ALGORITHM_PLAN.md`、`docs/OPTIMIZATION.md`

## 1. 背景与目标

Phase 0 审计确认：17,774 条结构 primitive 后中位 91 原子、91.5% 为 P1，
是真实的 Na 空位有序超胞。直接把它当作生成模型的训练分布，flow/diffusion
能学到的流形会很窄，逆设计只能在窄流形内优化——这是整套方案最致命的前提。

本计划的**唯一目标**：把"91 原子 P1 超胞的无约束联合生成"改写为
**"理想化母体（小晶胞）× Na 占据模式 × TM 装饰 × 畸变"的分层生成**，
把逆设计空间从高维连续流形压缩为"小集合 × 组合 × 低维畸变"。

**MVP 原则**：不追求一次覆盖全数据；先在数据量最大的一个骨架家族上把
分解→验证→生成管线完整跑通，再推广。可做可不做的全部不做。

## 2. Phase 1 冻结决策

| 决策 | 结论 | 影响 |
|---|---|---|
| F 是否保留 | **保留，作为独立条件**（氟磷酸盐家族单列分支，不进主焦磷酸/正磷酸生成目标） | 氟磷酸盐 2,481 条/102 式保留在数据集中，生成条件含 `fluoro` 标志 |
| C/N 是否保留 | **剔除出主生成集，单列备查**（nitro 971 条、C 88 条单独清单，不参与训练/评测） | 主生成集 = 无 C/N 的 Na–TM–P–O(–F) |
| Na=0 脱钠相 | **保留，作为全空占据端点**（217 条是占据枚举的合法端点，也是容量定义的另一端） | 母体 Na 位点占据向量允许全 0；脱钠相进入家族验证集 |

其余决策沿用 `HARD_CONSTRAINTS_SPEC.md` v1：TM={Fe,Mn,Ti,V}、P 仅 PO4、
TM 配位 ∈{4,5,6}、键型距离下限、体积窗口 10.5–20.5 Å³/atom。

## 3. 数据证据（分解为什么成立）

以下为 2026-08-18 在 `canonical_v1.jsonl` + `records.csv` 上的只读探针结果：

| 探针 | 结果 | 含义 |
|---|---|---|
| 564 式 → P–O–TM 骨架拓扑聚类（元素相关 WL hash） | **108 个家族**；top 10 覆盖 8,215 条（46%） | 家族轴只有百量级，远小于 564 式 |
| TiV 焦磷酸盐 Na1–Na7 系列 | 共享同一 12 节点框架（8P+4TM、20 边）；Na2↔Na7 晶格矩阵差 <0.04（M≈I） | 不同 Na 含量 = 同一母体的占据变体 |
| NaFeP2O7 vs Na2TiV(P2O7)2 | 元素无关图同构 = True | 母体框架跨 TM 通用；Fe/Mn/TiV 焦磷酸三个家族可合并（≥2,345 条/25 式） |
| Na2 vs Na7 的 Na 位点（对齐框架原点后） | 仍差 ~0.7 Å，框架原子差 ~0.5 Å | 观测结构是松弛后的衍生物；**母体是潜变量，不能逐对叠加直接读** |

第四行决定了方法论：任务分解不是聚类预处理，而是一个**反问题**——
从松弛、空位有序、带畸变的超胞族中推断（母体格、Wyckoff 位点、占据向量、
超格矩阵、畸变场）。

## 4. 分解定义（四层）

1. **构筑单元（motif）**：PO4 单体、P2O7（Q1）二聚体、P4O15、PO3 链（Q2）、
   TMO6/5/4 多面体、F 取代规则。Q0/Q1/Q2/mixed 统计已有，词典很小。
2. **理想化母体（parent）**：单元按角共享规则组装的最小对称晶胞（目标
   12–40 原子，有明确空间群与 Wyckoff 位点）。已知原型优先
   （Na2MP2O7 三斜 P-1、maricite NaFePO4、NASICON、alluaudite），
   新家族用"家族内所有晶格的整数化最小公共超格 + spglib 对称性"反推。
3. **占据模式（decoration）**：Na 位点占据向量 o∈{0,1}^K（K=母体 Na 位点数，
   目标 2–8）；TM 位点装饰（Fe/Mn/Ti/V）；F 标志。脱钠相 = 全 0 端点。
4. **超胞 + 畸变**：超格矩阵 M + 松弛位移场 Δx。这是生成管线的最后一步，
   也是 MLFF/DFT 的输入。

## 5. MVP 范围

### 做（MVP 必做）

- **单一目标家族**：焦磷酸 P2O7 12 节点框架（Fe/Mn/TiV 合并，≥2,345 条/25 式），
  母体候选 = Na2MP2O7 型 40 原子晶胞。整条管线只在这个家族上闭环。
- 第 1–3 步实现：家族聚类 → 母体搜索 → Na 亚晶格推断 → 占据向量表。
- 分解 registry v2（schema 见 §7）与验收门（§8）。
- 固定组成 CSP 生成演示：给定母体 + 指定 Na 含量，枚举/采样占据 → MLFF 松弛。

### 不做（明确排除，避免复杂化）

- 不推广到全部 108 个家族（先在 1 个家族验证方法论）。
- 不做 hull/形成能标签：`docs/OPTIMIZATION.md` 的 ΔE_hull 约束需要外部
  surrogate 或 DFT 标定，当前实验室数据不完备，**该约束在 MVP 中不可评估、
  不进入生成条件**（沿用 `HARD_CONSTRAINTS_SPEC.md` §3 的"不可评估"状态）。
- 不做电压/NEB/声子、不做多目标 Pareto 优化循环、不做 active learning 回训。
- 不做 D-Flow 源空间优化推理（见 §9，MultipleShooting 仅作后续备选）。
- 不做 novelty 指纹库与 novelty 目标（其优化依赖可微指纹代理，属 Phase 2+）。

MVP 的约束仅保留数据可支撑的解析项：电中性、P/TM 配位、键型距离、
体积窗口、容量（ΔNa 语义，§9）。

## 6. 操作流程（七步，MVP 版）

1. **骨架图聚类**：P–O–TM 连通图 + 元素无关 WL hash → family_id。
   冻结连接规则：P–O<2.15 Å、TM–O<2.55 Å、角共享边、P–O–P 桥。
   产出：家族清单 + 每家族成员 idx 列表（探针代码已跑通，固化为
   `src/nagen/decomposition/family_cluster.py`）。
2. **母体搜索**：先查已知原型表（StructureMatcher，容差按原型级）；
   未命中则做共同超格基搜索 + 对称性约简。产出：候选母体晶胞 + 格向量 +
   空间群/Wyckoff 表。
3. **Na 亚晶格推断**：成员去畸变（见 4）后映射到母体格，取 Na 位点**并集**，
   对称性约简为 K 个 Wyckoff 位点；每个成员 → 占据向量 o。注意：松弛使 Na
   位点偏移 ~0.7 Å，必须在母体格重建后投影，不能逐对叠加。
4. **残差分解**：成员原子位置 − 母体框架位置 → 位移场 Δx；
   报告框架 RMSD 分布，区分"占据改变"与"畸变"。
5. **反衍生物验证**：parent + M + occupancy 重建 → 与观测比对
   （占据模式用枚举器/穷举子集，受电中性、Na–Na 距离、容量窗口过滤）。
6. **registry v2**：接 canonical_id↔split 谱系，产出分解数据卡。
7. **门控**：§8 四项指标全部达标后冻结 dataset v2，进入生成实验。

## 7. 输出与 registry 格式

```
data/decomposition/
├── families_v2.json            # family_id -> {motif 组成, 成员 idx, 母体候选}
├── parents_v2.json             # parent_id -> {晶胞/格向量/Wyckoff 位点/空间群}
├── occupancy_v2.jsonl          # canonical_id -> {family, parent, o, TM_dec, M, rmsd}
├── dataset_v2.jsonl            # 分解后记录（原始结构 + 分解字段 + 能量）
└── decomposition_report.md     # 覆盖率/简洁性/重建保真度/能量有序性
```

每条分解记录至少含：`canonical_id`、`family_id`、`parent_id`、`occupancy`、
`tm_decoration`、`superlattice_M`、`distortion_rmsd`、`n_na`、`energy_mlff`。

## 8. 验收门（MVP）

| 指标 | 阈值（焦磷酸家族） | 说明 |
|---|---|---|
| 分解覆盖率 | ≥ 95% 成员重建 RMSD ≤ 0.3 Å（对去畸变母体格） | 表示层拒绝率 = 1 − 覆盖率，分家族报告 |
| 母体简洁性 | 1 个母体 + K ≤ 8 个 Na 位点 | 家族数/母体数随推广阶段监控 |
| 重建保真度 | 采样占据子集重建结构与观测 StructureMatcher 通过率 ≥ 90% | 用与第 5 步相同协议 |
| 能量有序性 | 家族内 MLFF 能量随 Na 含量单调（Spearman > 0.8） | hull 前的最弱一致性检查 |

两个全局硬指标沿用既定口径：**生成有效性拒绝率**（分层报告：家族采样失败 /
占据违反硬约束 / 畸变重建失败）与**代理–DFT 一致性（Kendall τ）**；
后者在家族×Na 含量网格上小样标定，MVP 阶段只需冻结协议，不强制跑 DFT。

## 9. 与生成 / 逆设计 / Dflow 的衔接

- **术语固定**：本文档及后续代码统一区分
  **D-Flow**（可控生成：对初始噪声 x0 求梯度、穿过 flow 反传的源空间优化）
  与 **Dflow**（工作流编排：MLFF/DFT 验证与主动学习回流的 DAG）。
- **生成分层**：先采样 family/parent（≤40 原子，MatterGen/FlowMM 均覆盖），
  再枚举/生成占据（组合空间，电中性、Na–Na 距离、容量窗口过滤），
  最后 MLFF 松弛 + 畸变。91 原子 P1 问题被拆掉。
- **容量定义（修正）**：`Q = ΔNa·F/(3.6·M)`，ΔNa 按**同一母体相邻占据状态
  的占据向量差**定义（如 Na2↔Na1），不是绝对 Na 数。占据枚举需保留
  "相邻状态对"关系，天然给出容量与电压差分对象。
- **D-Flow 推理（后续，非 MVP）**：当生成模型与可微代理就绪后，源空间
  优化的连续部分是母体晶格 + 畸变模式（低维），组合部分走枚举。
  长 ODE 反传出现梯度退化/不稳定时，**可切换 MultipleShootingFlow**
  （分段积分 + defect loss，`DFLOW_STRATEGIES.md` §2.3），段数通过实测确定。
- **split 注意**：现有 split v1 按 reduced_formula 分组（化学式泄漏 = 0）。
  同一母体跨多个 Na 含量（不同化学式），故分解后需在评测时按 family 汇总；
  是否升级为"母体级 split"推迟到生成评测阶段再决策，MVP 沿用 v1。

## 10. 风险与回退

- **母体搜索失败**：若某家族找不到公共小晶胞（M 非整数化/框架 RMSD 过大），
  该家族标记 `parent_fail` 并剔除出 MVP 覆盖率统计，不阻塞焦磷酸家族。
- **Na 位点并集歧义**：K 与对称性约简依赖 spglib 容差；
  以去畸变母体格为基准冻结 symprec，并记录在 families_v2 元数据中。
- **能量有序性不达标**：MLFF 能量本身有 35.5% 中断松弛记录，
  先只用无 error 标记记录（11,470 条）做有序性检查；不达标则降级为
  "同家族同 Na 含量能量方差"报告，不阻塞分解交付。

## 11. 参考

- Phase 0：`PHASE0_REPORT.md`、`DATA_CARD_v1.md`、`HARD_CONSTRAINTS_SPEC.md`
- 优化问题：`docs/OPTIMIZATION.md`（MVP 仅取其可解析约束）
- 生成/逆设计：`PROJECT_PLAN.md`、`ALGORITHM_PLAN.md`、`DFLOW_STRATEGIES.md`

## 12. 执行状态（2026-08-18）

MVP 分解管线已实现并跑通，产物与完整结果见
`data/decomposition/decomposition_report.md`。执行对计划的修正：

- **全量聚类**：17,774 → 498 元素相关家族 / 47 框架家族；
  纯焦磷酸 12 节点框架（Q1、无 F/C/N）实际 210 条 / 27 式
  （计划预估 2,345 条有误，来源是把含 F/其他 q_label 的记录混入）。
- **单母体分解在几何一致核心上成立**：TiV/Fe/Mn 三个家族的核心
  （align<0.35 Å）占据向量 100% 与观测一致（含 Na=0 端点），
  K=8/9/14，重建 RMSD 中位 0.07–0.41 Å，能量有序 Spearman 0.73–0.96。
- **家族内含多个几何多形体**：同一 el-aware 家族中 60–79% 成员无法对齐到
  单一母体（暴力网格验证真实差异可达 1.6 Å）→ "一个母体"需扩展为
  "按几何一致核心迭代多母体"，这是第 1 步之后的直接扩展项。
- **实现经验**（已固化进代码）：canonical primitive 格基不统一需用
  `frac @ (Lm @ inv(Lp))` 基变换；Na 位点并集需 PBC 感知平均；
  Na 亚晶格不做框架对称性约简；TM（Ti/V）装饰互换需用 TM 无关几何度量。
- **门控结果**：能量有序性达标（Spearman>0.8 于多数装饰组）；
  母体简洁性基本达标；覆盖率（21–40%）未达 95%，因多形体，待多母体扩展。
- **代码**：`src/nagen/decomposition/`（family_cluster / parent_search /
  na_sublattice / validate / run_decomposition），CLI 支持 `--family-id` 与
  `--records-family`。

## 13. 多母体扩展执行状态（2026-08-18，第 9 节"生成"之前）

已实现 `src/nagen/decomposition/multi_parent.py` 并在纯焦磷酸 12 节点框架
（210 条）上跑通，产物见 `data/decomposition/multi_parent/`：

- **11 个母体**：核心覆盖率 67.1%（141/210），registry 覆盖率 97.6%（205/210；
  64 条尾部就近指派，5 条无法指派）。所有母体 matcher ≥0.94（多数 1.00），
  重建 RMSD 中位 0.000–0.41 Å，核心占据与观测一致（1 条边界例外）。
- **能量网格（hull 基础）**：每母体 × Na 含量 → MLFF 能量/原子（mean/median/n/
  n_relax_ok）；能量随 Na 含量单调（Spearman 0.67–1.00）。144xx 批次全部带
  I/O error 标记（中断松弛能量），网格已给出并标注 relax_ok=0，DFT 标定时需
  优先用 relax_ok 样本。
- **容量（ΔNa 语义）**：每母体相邻 Na 含量对已给出 Q = ΔNa·F/(3.6·M_lowNa)
  （例 0→4 = 120.0 mAh/g），生成约束可直接引用。
- **覆盖率提升路径**：MIN_CORE 降到 3、或尾部"母体 + 定向畸变"建模；
  未指派 5 条保留在 registry 外。

## 14. 第 9 阶段执行状态（2026-08-18）：Flow Matching + D-Flow 完成

已实现 `src/nagen/generation/`（flow_data / flow_model / surrogate / dflow_infer），
完整报告见 `data/generation/GENERATION_REPORT.md`：

- **数据**：205 条母体畸变场（模板 + delta 坐标/晶格，条件 51 维）。
- **Flow matching 训练**：900 epoch，loss 1.80→1.15；权重 `models/flow.pt`。
- **可微能量代理**：平滑 RDF + MLP，训练集一致性 MAE=0.013 eV/atom、R²=0.994；
  权重 `models/surrogate.pt`。
- **生成分层**：flow 采样 + 几何过滤 + 代理排序，12/12 目标获得有效低能候选
  （min_dist≥1.2 Å、体积窗口内、P 配位多数合格）。
- **D-Flow 推理**：x0 源空间优化（12 步 ODE 反传 + Adam 30 步），4 个目标
  平均降能 -0.227 eV/atom 且结构保持有效；结果 `dflow_results.json`。
- **术语**：本阶段完成的是 **D-Flow（可控生成）**；**Dflow（工作流编排）**
  作为后续 MLFF/DFT 验证与主动学习回流层，尚未执行。
- **待办**：Dflow 验证（代理-DFT Kendall τ、生成有效性拒绝率两硬指标）、
  flow 模型升级（领域适配/更大模型）、MultipleShooting 提效。

## 15. Dflow-SUR 可控生成执行状态（2026-08-18，200 候选）

实现 `src/nagen/generation/dflow_sur_generate.py`，按
`docs/OPTIMIZATION.md` 约束生成并过滤候选：

- **流程**：目标（母体 × Na 含量，Q≥130）→ 源空间优化（8 步 ODE + 20 步 Adam，
  代理能量 + 距离软惩罚）→ 硬过滤（电中性/键距/P·TM 配位/体积/容量/ΔE_proxy）
  → RDF 指纹去重。
- **结果**：920 优化候选 → **200 个通过全部约束**（拒绝率约 66%）；
  容量 133–199 mAh/g、ΔE_proxy≤0.150 eV/atom、min_dist 1.40–1.57 Å、
  体积 11.3–20.0 Å³/atom、电荷/配位全部通过。
- **覆盖**：5 个母体 / 7 个化学式（Na3–Na8 焦磷酸盐）；Mn 母体候选未通过硬过滤。
- **文件**：`data/generation/candidates_200.jsonl`、
  `data/generation/candidates_200_cif/`、`data/generation/candidates_summary.md`。
- **边界**：ΔE_proxy 为代理口径（E_surr − 母体能量网格最小），非 DFT hull；
  代理-DFT Kendall τ 与最终有效性拒绝率待 Dflow 验证层执行。

## 16. 与原数据不同的 200 个新结构（占据枚举，2026-08-18）

针对"生成数据与原数据几乎相同"的评审意见，改为**占据枚举**路线：

- 在母体 K 个 Na 位点上枚举数据中未出现的占据模式（新 Na 排序），
  纯母体模板几何 + 约束过滤。
- **结构级新颖性硬门槛**：200/200 与全部 205 个训练结构
  StructureMatcher 不匹配（426/500 池化候选通过）；占据向量全部不在
  registry 内。
- 约束：电中性、键距、P/TM 配位、体积、容量 ≥130 mAh/g（133–178）、
  Na–Na ≥2.0 Å，全部满足。
- 构成：5dfa299c-1（TiV，K=12）168 条、43badee5-1（Fe，K=11）29 条、
  43badee5-0 3 条；化学式仍属原数据集合（新颖在 Na 排序层），
  组成级新化学式（TM 装饰/配比）尚未枚举。
- 文件：`data/generation/novel_200.jsonl`、`novel_200_cif/`、
  `novel_200_summary.md`。

## 17. N/A/L 全变的 200 个新结构（组成级新颖，2026-08-18）

按用户要求（N、A、L 至少 A、L 必须变化），实现
`src/nagen/generation/novel_composition.py`：

- **A**：TM 装饰配比枚举（Ti/V、Fe/Mn 互掺及纯 Ti/V）→ **42 个新化学式**，
  200/200 化学式不在原 564 集、晶胞组成不在训练集。
- **L**：单胞晶格长度扰动 ±3%；另有 2× 超胞（120 条）晶胞加倍。
- **N**：Na 含量变化 + 超胞 → N 覆盖 45–48 与 86–89（8 种取值）。
- 验证：200/200 与 205 个训练结构 StructureMatcher 不匹配；
  约束（电中性/键距/配位/体积/容量≥130）全部通过。
- 示例新化学式：Na7Mn2Fe6(P2O7)8、Na4MnFe3(P2O7)4、Na7V8(P2O7)8 等。
- 文件：`data/generation/nova_200.jsonl`、`nova_200_cif/`、
  `nova_200_summary.md`。
- 边界：能量为代理口径（非 DFT）；L 扰动幅度受硬过滤约束，
  更大变形需 MLFF 松弛支撑。

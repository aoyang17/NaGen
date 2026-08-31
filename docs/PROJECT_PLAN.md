# NaGen 整体技术布置

## 1. 任务定义

目标不是仅生成“长得像磷酸盐”的晶体，而是在明确的化学与物理约束下，产生可计算、可松弛、稳定、独特且有电池价值的 Na–TM–P–O 候选，并保留从原始样本、模型、采样参数到 DFT 结果的完整谱系。

建议把可控目标分为三层：

1. **硬约束**：必须含 Na/P/O；允许的过渡金属集合；元素计量范围；电中性/合理氧化态；原子数上限；最小原子间距；可选固定组成或空间群。
2. **生成条件**：Na 含量、TM 组合、空间群、密度、体积/原子、P–O 与 TM–O 配位、骨架类型、目标能量区间。
3. **优化目标**：能量高于凸包、形成能、工作电压、理论容量、Na 脱嵌能、体积变化、Na 迁移势垒、电子结构与合成可行性。这里应采用 Pareto 优化，不能把所有目标粗暴压成单一分数。

“高精度”的定义必须落在验证层级上：生成网络负责提案，MLFF 负责快速松弛/筛选，最终结论由一致设置的 DFT 验证；生成模型本身不能替代 DFT 精度。

## 2. 当前数据审计结论

只读检查结果：

- 17,774 行，约 506 MB，每行为 JSON；
- 含 `material_id`、`formula`、pymatgen 风格 `structure`、能量与 Na/TM/P 的 O 配位计数字段；
- `structure` 含完整晶格矩阵、周期边界和位点的元素/分数坐标/笛卡尔坐标；首条为 36 个位点；
- 抽查前三条时，`energy_above_hull`、`energy_per_atom` 和 `num_*` 配位列均为空，`optimized_energy_per_atom` 有值。

这意味着第一项工程不是训练，而是生成正式的 `data_audit.json`/数据卡，统计全量缺失率、元素覆盖、结构大小、晶格分布、重复结构、异常近邻、无序/部分占位、氧化态可解性及能量来源一致性。

### 必做数据治理

- 流式读取 JSONL，禁止一次性加载 506 MB；转换产物写入项目目录；
- 用 pymatgen/ASE 做 schema 校验与规范化，统一 primitive/conventional 选择，但保留原结构；
- 以 composition hash + StructureMatcher/结构指纹分组去重；
- 按结构原型/化学体系分组划分 train/val/test，防止同源结构泄漏；随机逐行切分不合格；
- 保存 `raw_id → canonical_id → split → derived_features → calculation_id`；
- 数据集和特征都带版本、代码提交、参数和内容 hash；
- 区分缺失值、零值、计算失败，绝不将 `null` 默认为 0；
- 确认 `optimized_energy_per_atom` 的计算方法、赝势、泛函、U 值和参考态。不同能量口径不可直接训练或构造凸包。

## 3. 数据表示与特征工程

### 3.1 生成模型的主表示

参考 MatterGen，将晶体表示为 `(A, X, L, N)`：元素类型、周期分数坐标、晶格和原子数。网络必须满足原子置换不变性、平移不变性、周期性，并在坐标/晶格分支具备合理的旋转等变或不变处理。

- `A`：离散元素 token；可附周期表族、周期、电负性、共价/离子半径、常见氧化态嵌入；
- `X`：`[0,1)^3` 分数坐标，周期最短镜像建图；
- `L`：建议使用旋转不变的对称晶格参数化或极分解/Gram 表示，避免直接扩散任意 3×3 矩阵导致退化晶胞；
- `N`：对磷酸盐建立经验分布或条件化原子数先验；大超胞先统一约简；
- 图边：周期邻居图，cutoff 与最大邻居数需在数据统计后确定。

### 3.2 条件与领域特征

特征分三类，避免把所有手工特征塞进生成主干：

- **条件特征**：化学体系、化学式/Na:TM:P 比、空间群、目标能量、目标电压/容量、结构族；
- **代理模型特征**：组成统计、密度、体积/原子、局域环境、RDF/ADF、SOAP/结构指纹、配位多面体畸变、键价和；
- **后验筛选特征**：电中性、氧化态、P–O 四面体完整性、TM–O 配位、O–O/P–P 异常距离、Na 通道连通性。

现有 `num_NaO3...num_PO12` 可重新计算并作为条件或评估描述符，但不是原子级生成表示。磷酸盐任务尤其应增加：PO4 四面体比例/畸变、TM 多面体连接方式、Na 位点网络维数、Na–Na 瓶颈半径，以及脱钠前后结构变化。

## 4. 生成模型架构

### 4.1 MatterGen 基线

MatterGen 联合去噪元素、坐标和晶格，并通过属性适配器及 classifier-free guidance (CFG) 控制化学、对称性和性质。NaGen 应先复现三种可比较模式：

1. **CSP 模式**：固定具体化学式，仅生成晶格和坐标；最稳健，适合已知组成探索多晶型。
2. **chemical-system 条件微调**：限定 Na–TM–P–O 元素集合，联合生成组成与结构。
3. **多属性适配器**：在有可信标签后加入稳定性、电压、容量、结构族等条件。

不要第一步就做多属性“全能模型”。先做 CSP 与化学体系条件，验证结构有效率和稳定率，再增加性能条件。

### 4.2 自研生成线

统一数据接口下保留研究接口，候选包括周期等变图扩散、flow matching/rectified flow、或分层生成（先组成与空间群，后 Wyckoff/坐标与晶格）。推荐的自研增量不是复刻 MatterGen，而是：

- 将 PO4 作为可选的软结构先验/多尺度 motif 表示；
- 对离散元素使用 categorical diffusion/masked modeling，对坐标和晶格用连续流；
- 将硬化学约束放入采样器，将软性质目标交给 CFG/代理梯度/Pareto 重排序；
- 为固定组成、可变 Na 含量与开放组成分别设计任务头。

## 5. 路线选择

| 路线 | 优点 | 主要风险 | 建议 |
|---|---|---|---|
| 直接用预训练 MatterGen 采样 | 最快建立基准 | 磷酸盐命中率和领域精度有限 | 必做 sanity baseline |
| MatterGen CSP/领域微调 | 利用大规模通用晶体先验，17k 数据可用 | 数据格式适配、标签缺失、域偏移 | **主线，优先** |
| 从 MatterGen 架构重新训练 | 可控制训练分布 | 17k 对联合生成偏小，容易过拟合 | 不作为第一阶段主线 |
| 自研 motif/flow 模型 | 可体现磷酸盐先验与方法创新 | 工程和验证成本最高 | 统一 benchmark 后并行推进 |
| 生成 + Dflow 闭环 | 能形成高精度证据和主动学习 | 算力、失败恢复、DFT 口径管理复杂 | **必需的验证/优化层** |

推荐采用两阶段迁移：先从通用 `mattergen_base` 进行磷酸盐无条件/化学域继续训练，保持通用几何先验；再用小型 adapter 做具体性质控制。可比较全量微调、冻结主干 adapter、以及逐层解冻三组消融，依据验证集有效性/稳定性和灾难性遗忘选择，而不是预先认定某一种微调方式。

## 6. 生成后优化与 Dflow 闭环

Dflow 是工作流编排与数据回流层，不是生成模型替代品。建议 DAG：

```text
conditional sampling
  → schema/geometry/charge hard filters
  → structure deduplication + novelty check
  → surrogate property prediction + uncertainty
  → MLFF relaxation and energy screening
  → Pareto/active-learning selection
  → standardized DFT relaxation
  → static energy / hull / voltage
  → optional NEB, phonon, AIMD
  → registry + retraining dataset
```

工作流必须具备幂等任务 ID、重试、失败分类、资源配置、断点续算和 artifact hash。DFT 参数（泛函、U、赝势、磁性初值、k 点密度、截断能、收敛阈值）版本化；同一排名只使用兼容口径。

### 多保真候选分配

- Level 0：价态/计量/距离/体积/PO4 完整性硬过滤；
- Level 1：一个或多个材料代理模型，输出均值与不确定性；
- Level 2：MLFF 松弛，检查坍塌、RMSD、能量与力；
- Level 3：DFT 几何优化和静态能，重建相图计算 `E_hull`；
- Level 4：对少量 Pareto 候选计算电压、容量、Na vacancy、NEB、声子/AIMD。

主动学习获取函数同时考虑目标性能、模型不确定性、新颖性和多样性。DFT 结果回流时保留“生成前结构”和“松弛后结构”，可进一步训练可松弛性/成功率模型，减少无效 DFT。

## 7. 评测体系

### 生成质量

- validity：可解析、合理晶格、无严重原子重叠、组成/电荷合法；
- uniqueness/novelty：结构去重且与训练库不匹配；
- stability：MLFF 与 DFT 两套阈值分开报告；核心指标可用 SUN（stable, unique, novel）；
- relaxability：松弛成功率、能量下降、晶格/坐标 RMSD、结构是否坍塌；
- diversity/coverage：组成、空间群、原型和指纹覆盖。

### 可控性

- 硬条件满足率；
- 连续性质 MAE、目标窗口命中率及 calibration；
- CFG guidance 扫描下的控制–多样性–稳定性曲线；
- 多目标 hypervolume/Pareto front，而不仅是平均分。

### 对照与消融

- 训练集检索/元素替换与随机结构搜索；
- MatterGen 直接采样、CSP、领域继续训练、adapter 微调；
- 有/无磷酸盐 motif 先验；
- 随机切分与原型隔离切分的差异；
- MLFF 排名和 DFT 排名相关性。

所有模型使用固定 blind test、固定采样预算和固定 DFT 预算，报告置信区间与失败数。

## 8. 工程模块与实验管理

1. `data`：schema、流式 ETL、标准化、去重、split、data card。
2. `representations/features`：周期图、晶格参数化、条件编码、领域描述符。
3. `models`：MatterGen wrapper/adapters、代理模型、自研生成模型。
4. `generation`：条件规范、采样、CFG、硬约束、批量导出 CIF/extxyz。
5. `evaluation`：结构匹配、SUN、控制命中率、松弛前后指标。
6. `optimization`：Pareto 排序、Bayesian/active learning、批次多样性选择。
7. `workflows`：Dflow DAG、MLFF/DFT/NEB 模板、失败恢复。
8. `registry`：数据版本、模型卡、运行配置、候选谱系、计算证据。

配置建议使用 Hydra；每次运行保存解析后的完整配置、随机种子、环境锁文件、git commit、数据 hash、checkpoint 和指标。训练/验证/测试结构 ID 清单必须落盘，避免后续实验静默换 split。

## 9. 分阶段实施与验收门

### Phase 0：定义与审计（1–2 周）

- 全量数据报告、标签来源表、目标 TM 集合和硬约束规范；
- canonical dataset v1、group split v1、结构重复与异常清单；
- 明确 DFT 口径与“高精度/稳定”的阈值。

**Gate**：缺失值和泄漏可解释，抽样结构可无损往返 JSON→Structure→CIF。

### Phase 1：基线（2–4 周）

- 直接 MatterGen、固定组成 CSP、化学体系条件三条基线；
- 统一生成/去重/MLFF 评测；
- 训练轻量性质代理模型并校准不确定性。

**Gate**：在固定预算下报告 validity、SUN、RMSD、条件命中率及相对检索/替换基线的增益。

### Phase 2：领域迁移与控制（4–8 周）

- 磷酸盐继续训练；adapter/逐层解冻消融；
- 加入配位/结构族和可信物性条件；
- CFG 与硬约束组合调优。

**Gate**：blind 原型测试集和首批 DFT 上同时提升条件命中与稳定性，且多样性没有不可接受地坍塌。

### Phase 3：Dflow 主动学习（持续）

- 自动化多保真工作流；批量 DFT；失败恢复和 registry；
- 按不确定性 + Pareto + 多样性选择候选并周期回训；
- 对头部候选做电压、脱钠、NEB、声子/AIMD。

**Gate**：以“每 100 次 DFT 获得的 DFT-stable、novel、target-hit 候选数”为核心效率指标。

### Phase 4：自研生成器

- 在完全相同的数据 split、采样数和 DFT 预算下比较 motif-aware flow/diffusion；
- 只有在稳定性、控制性或计算效率至少一项有明确增益时才扩展。

## 10. 最近的执行顺序

1. 编写只读流式审计器，生成完整缺失率、元素/原子数/晶格/重复统计；
2. 与领域专家冻结 v1 控制变量和 DFT 计算口径，尤其是 TM 元素范围、价态、目标电压/容量与稳定阈值；
3. 建立 canonical schema 与 group split；
4. 部署 MatterGen 原版小批量采样并验证评测链；
5. 先做固定组成 CSP，再做 Na–TM–P–O 域微调；
6. 接入 MLFF，最后接 Dflow/DFT；不要在数据口径未定时先发起大量 DFT。

## 参考

- Microsoft MatterGen 官方仓库：https://github.com/microsoft/mattergen
- MatterGen 论文（Nature）：https://doi.org/10.1038/s41586-025-08628-5
- Microsoft Research 技术介绍：https://www.microsoft.com/en-us/research/blog/mattergen-a-new-paradigm-of-materials-design-with-generative-ai/

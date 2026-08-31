# NaGen 新一轮磷酸盐钠正极生成项目执行计划

> 状态：待 `/goal` 执行
> 制定日期：2026-08-26
> 项目目录：`/home/aobo/NaGen`
> 优化问题唯一依据：`docs/OPTIMIZATION.md`
> 本文档自发布起取代 `docs/ALGORITHM_PLAN.md`，后者仅作为历史记录，不再作为本轮实验的执行依据。

## 1. 项目目标

训练一个面向 Na–Fe–P–O 晶体的无条件 Flow Matching 生成模型，并在推理阶段使用 ShootingFlow/D-Flow 思路，将多目标、多约束优化加载到生成 ODE 上：冻结模型参数，通过终点目标对初始噪声反向传播，寻找优化后的初始噪声，再从该噪声完整积分生成新晶体。

本轮实验的最终目标不是只得到“损失下降”的轨迹，而是得到一批：

1. 满足 `docs/OPTIMIZATION.md` 全部解析硬约束；
2. 经 UMA + FIRE/BFGS 完整结构弛豫后仍满足硬约束；
3. 项目定义的 `E_hull <= 0.150 eV/atom`；
4. 相对训练集和已知结构具有新颖性；
5. 具备完整来源、评分、轨迹和可复现实验记录的 Na–Fe–P–O 候选结构。

本轮不以 DFT 稳定性作为结论。最终结果中的 `E_hull` 是同一 UMA 能标下、相对 MP 竞争相结构集合构建的模型凸包距离。

## 2. 核心算法定义

### 2.1 生成变量

晶体表示为：

- `N`：原胞原子数；
- `A`：元素类型；
- `X`：分数坐标；
- `L`：晶格。

生成分解按以下方式理解：

```text
N ~ p_N
A ~ p_A(A | N)
(X, L) = Phi_theta(z_X, z_L | N, A)
```

第一版 ShootingFlow 中固定离散变量 `N, A`，只优化连续源变量 `z_X, z_L`。原因是 `N` 和离散元素类型不适合直接用普通梯度优化，同时固定组成后，电荷、容量及 UMA 凸包参考能可以提前计算或缓存。

后续若需要联合优化组成，必须另立实验，引入离散松弛、枚举或组合优化；不得在本轮主实验中把连续原子 logits 当作已经解决的离散组成优化。

### 2.2 无条件 Flow Matching

先在 Na–Fe–P–O 主数据上训练无条件 Flow Matching 模型：

```text
dz/dt = v_theta(z_t, t; N, A)
```

训练结束后冻结 `theta`。本轮推理优化不更新网络参数，不重新训练代理模型。

### 2.3 ShootingFlow 推理优化

对给定可行的 `N,A` 和初始源噪声 `z_0=(z_X,z_L)`：

```text
C_1 = Phi_theta(z_0 | N, A)
```

在终点晶体 `C_1` 上计算目标与约束，通过完整 ODE 积分图反向传播到 `z_0`：

```text
z_0* = argmin_z  J(Phi_theta(z | N,A)) + source_prior(z)
```

获得 `z_0*` 后，从头使用冻结的 Flow Matching 模型重新积分一次，生成最终晶体：

```text
C_1* = Phi_theta(z_0* | N,A)
```

“优化轨迹”在这里指通过优化初始条件间接改变整条 ODE 轨迹。主实验不把每个时间点当作独立自由变量，也不在生成结束后直接移动终点坐标冒充 ShootingFlow。

### 2.4 多目标与约束

两个主目标为：

1. 最大化结构新颖性；
2. 最小化 `E_hull`。

主实验使用 MGDA 合并两个目标梯度，硬约束使用增广拉格朗日或平滑 hinge 罚项，源分布偏移使用 source-prior/trust-region 正则限制。

两目标 MGDA 应计算使组合梯度范数最小的权重：

```text
g = alpha * g_hull + (1-alpha) * g_novelty,  alpha in [0,1]
```

必须记录两个原始梯度、夹角、范数和最终权重，避免某一目标因量纲或梯度幅值长期失效。固定加权和仅作为消融或偏好扫描，不作为唯一主方法。

## 3. 指标和约束的唯一口径

具体数学定义以 `docs/OPTIMIZATION.md` 为唯一依据。执行代码、配置和报告不得自行采用旧口径。

### 3.1 E_hull

本项目只使用以下定义：

```text
E_hull(C) = E_UMA(C) - E_ref^UMA(comp(C))
```

- `E_UMA`：生成晶体由 UMA 预测的每原子能量；
- `E_ref^UMA`：Materials Project 提供竞争相结构和组成，全部竞争相由同一 UMA checkpoint 重算后构建相图所得的组成参考能；
- 优化方向：越小越好；
- 强约束：`E_hull <= 0.150 eV/atom`。

该量不是 DFT 凸包距离；它是 UMA 模型能标下、以 MP 竞争相结构集合为支持域的凸包距离。所有输出必须保存 UMA checkpoint 哈希、任务头、竞争相结构来源、结构处理协议、缓存日期和缓存哈希。

2026-08-27 的 12 结构校准已经证明原 `E_UMA-E_MP` 跨能标差值不可用于绝对门控；其中位绝对差为 `0.683 eV/atom`。该旧量只保留为历史校准诊断，禁止再作为优化目标、`E_hull` 或最终约束，也不得通过拟合或扣除常数偏移恢复使用。

候选和竞争相必须使用相同 UMA checkpoint、任务头和能量归一化。竞争相须先按与最终候选一致的 UMA+FIRE/BFGS 协议完整弛豫，再缓存其每原子能量；推理期候选可使用可微 UMA 单点终点能引导，最终门控必须在相同协议的 UMA 完整弛豫后重算。固定 `N,A` 时组成不变，`E_ref^UMA` 在一次 shooting 优化中为缓存常数，因此对 `X,L` 的梯度来自 `E_UMA`。

禁止在每个梯度步调用 Materials Project API 或重算竞争相。必须预先建立“MP 竞争相结构→UMA 能量→UMA 相图”的离线缓存；缓存缺失、checkpoint 哈希不一致或无法构建参考值的组成不能通过 `E_hull` 强约束。

### 3.2 组成、电荷和容量

只允许 Na、Fe、P、O，且四种元素都必须出现。组成归一化为：

```text
Na_m Fe_n P O_(4-y)
```

并满足：

- `0 <= y <= 0.5`；
- `m >= 1`；
- `m >= n`；
- `n <= 1`；
- Fe 平均价态 `z_Fe = (3 - 2y - m) / n`，且 `2 <= z_Fe <= 4.5`；
- Fe 可氧化当量 `kappa_Fe = m + 4.5n + 2y - 3`；
- `k = min(m, kappa_Fe)`；
- 理论比容量不低于 `130 mAh/g`。

这些条件在本轮中作为 `N,A` 生成/枚举后的精确预筛选，不依赖坐标梯度来修复。

### 3.3 几何和配位

- 原胞原子数：`23 <= N <= 184`；
- 体积/原子：`10.5 <= V/N <= 20.5 Å^3/atom`；
- P–O 配位截断：`2.00 Å`，P 配位数为 4；
- Fe–O 配位截断：`2.50 Å`，Fe 配位数为 4、5 或 6；
- 最小原子间距等要求以 `OPTIMIZATION.md` 为准；
- 已删除“桥氧与骨架连通性”，不得重新加入约束或评分。

推理梯度阶段使用可微近似：平滑最短距离、sigmoid 配位计数、体积 hinge。生成结束后必须再用精确规则复核。

### 3.4 新颖性

推理阶段使用当前可微晶体描述符，并以 smooth-min 或 top-k soft-min 近似到参考库的最近距离，避免硬 `min` 导致梯度不稳定。

最终新颖性必须使用精确结构匹配和去重流程确认。代理新颖性分数只用于引导，不能取代最终结构等价判断。

### 3.5 结构弛豫

结构弛豫不进入 ShootingFlow 边值问题，也不在每个梯度步运行。

它是完整生成结束后的独立强验证阶段：

```text
生成结构 -> UMA 能量/力/应力 -> FIRE/BFGS -> 收敛结构 -> 全部硬约束复核
```

UMA 是本轮统一使用的预训练原子势，不再以泛称“MLFF”描述该步骤。

## 4. 项目目录和数据边界

### 4.1 只读输入

原始数据：

```text
/mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl
```

`/mnt/data2/shared` 严格只读。不得在该目录写缓存、索引、修复文件或日志。

参考脚本：

```text
/mnt/data2/LiuSW/shared_space/NaCathode/CodingSpace/relaxation_n_ehull.py
```

该脚本只作为 UMA 弛豫和 MP 竞争相结构查询方式的参考，不直接修改；其跨能标能量函数不进入本项目正式 `E_hull` 口径。

### 4.2 代码和文档

所有代码、测试、配置及文档保存在：

```text
/home/aobo/NaGen
```

### 4.3 本轮耐久实验目录

默认实验 ID：

```text
exp_2026-08-26
```

耐久输出位于：

```text
/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26
```

若目录已存在，依次使用 `_02`、`_03`，不得覆盖旧实验。在仓库中建立只指向该目录的相对入口：

```text
/home/aobo/NaGen/experiments/<experiment_id>
```

建议目录结构：

```text
<experiment_id>/
  manifest/
  data/
  checkpoints/
  mp_cache/
  guidance/
  generated/
  relaxed/
  candidates/
  logs/
  reports/
```

临时文件只放 `/tmp`，环境、模型、缓存和日志不得写入 `/mnt/data2/shared`。

## 5. 分阶段执行计划

所有阶段必须按顺序执行。每一阶段只有在验收门通过后才能进入下一阶段；失败时先修复当前阶段，不得用放宽最终硬约束的方式绕过。

### Phase 0：冻结规范并同步可执行定义

目标：消除 `OPTIMIZATION.md` 与当前代码的旧口径冲突。

执行项：

- [ ] 记录代码 commit、dirty 状态、Python/CUDA/依赖版本；
- [ ] 给 `OPTIMIZATION.md` 生成内容哈希，写入实验 manifest；
- [ ] 更新 `src/nagen/inverse/spec.py`，只保留 Na/Fe/P/O；
- [ ] 移除 Mn/Ti/V/F/N/C 等旧元素和旧 TM 通用逻辑；
- [ ] 将 Fe 价态逻辑同步为平均价态 `[2,4.5]`；
- [ ] 将 P–O 截断同步为 `2.00 Å`；
- [ ] 将 Fe–O 截断同步为 `2.50 Å`；
- [ ] 同步组成、电荷、容量、体积、原子数和配位规则；
- [ ] 删除代码及配置中“桥氧与骨架连通性”入口；
- [ ] 为每条硬约束增加边界值单元测试；
- [ ] 搜索仓库，列出并消除仍参与执行的旧常数。

基线测试命令：

```bash
PYTHONPATH=src:.pylibs /mnt/data2/aobo/envs/NaGen/bin/python -m unittest discover -s tests -v
```

验收门 G0：

- `spec.py`、数据筛选、生成验证和后处理使用同一套定义；
- 所有约束边界测试通过；
- 仓库中不存在会进入本轮运行的旧元素集合、`2.15/2.40 Å` 截断或 Fe `[2,3]` 价态配置；
- 测试无回归。

### Phase 1：构建 Na–Fe–P–O 主数据集

目标：从只读原始数据构建可复现、无泄漏、质量可追踪的训练数据。

执行项：

- [ ] 精确提取四元 Na–Fe–P–O 且四种元素均存在的结构；
- [ ] 规范化元素顺序、周期坐标和晶格表示；
- [ ] 计算组成、原子数、体积/原子及解析约束标签；
- [ ] 保存上游 UMA 弛豫状态和中断标记，禁止静默丢弃；
- [ ] 主训练集优先采用成功弛豫且几何质量合格的结构；
- [ ] 将包含未成功弛豫样本的版本作为数据消融，不与主集混用；
- [ ] 按约化化学式或结构族进行 group split，禁止同化学式跨 train/val/test 泄漏；
- [ ] 生成数据审计报告、样本索引和文件哈希。

预期原始四元子集规模约为 4,196 个结构、116 个约化化学式，原胞 `N` 范围约为 23–184；实际数字必须由本轮审计重新确认，不可直接把预期值写成结果。

输出：

- `data/naxl_train.pt`、`naxl_val.pt`、`naxl_test.pt`；
- `manifest/dataset_manifest.json`；
- `reports/data_audit.md`；
- 训练集结构的新颖性参考索引。

验收门 G1：

- 所有样本仅含 Na/Fe/P/O；
- split 无公式级泄漏；
- 数据哈希、筛选原因和弛豫状态可追踪；
- 随机抽样反序列化后晶格、元素、坐标与源记录一致。

### Phase 2：训练无条件 Flow Matching 基线

目标：得到可独立生成合理 `N,A,X,L` 的冻结基模型。

执行项：

- [ ] 从训练集拟合经验原子数分布 `p_N`；
- [ ] 实现或校准 `p_A(A|N)`，并保留组成预筛选接口；
- [ ] 训练 `(X,L)` 条件于固定 `N,A` 的 Flow Matching；
- [ ] 保存 best/last checkpoint、训练曲线和完整配置；
- [ ] 固定验证 seed，周期性生成无条件样本；
- [ ] 评估流损失、数值稳定性、晶格正定性、坐标周期性和基础几何有效率；
- [ ] 比较 ODE 积分步数 12/24/48 的结果收敛性；
- [ ] 生成未引导 baseline，作为后续所有比较的共同对照。

不得只根据训练 loss 选择模型。checkpoint 选择必须同时考虑验证集 loss 和无条件生成有效率。

输出：

- `checkpoints/flow_best.pt`；
- `manifest/train_config.yaml`；
- `generated/baseline/`；
- `reports/unconditional_baseline.md`。

验收门 G2：

- 训练和验证无 NaN/Inf；
- 固定 seed 可复现；
- 生成的晶格有效，分数坐标和 PBC 处理正确；
- 24 与 48 步关键指标差异在预设容差内，否则提高主实验积分精度；
- baseline 指标和失败类型已完整记录。

### Phase 3：集成终点评估器和可微梯度桥

目标：在不训练新代理模型的前提下，让终点目标能对 `X,L` 反向传播。

#### 3.1 UMA 能量梯度桥

优先使用 UMA/fairchem 原生张量前向路径计算能量并保留自动微分图。若 `FAIRChemCalculator` 将数据转成 NumPy 或截断梯度，则实现自定义 autograd/VJP：

- 坐标导数由 UMA 力提供；
- 晶格导数由 UMA 应力及一致的晶胞变换关系提供；
- 前向值仍是同一个 UMA 模型的能量；
- 不训练新的能量代理模型。

对坐标和晶格分别进行中心有限差分验证，记录绝对误差、相对误差和方向一致性。

#### 3.2 UMA 竞争相凸包缓存

- [ ] 按化学体系查询并缓存 Materials Project 竞争相结构与组成；
- [ ] 用同一 UMA checkpoint、任务头和 UMA+FIRE/BFGS 完整弛豫协议计算全部竞争相每原子能量；
- [ ] 由 UMA 能量构建 PhaseDiagram，并预计算 `E_ref^UMA(comp)`；
- [ ] 保存查询时间、MP 条目 ID、UMA 模型哈希、API/库版本、结构处理协议和缓存哈希；
- [ ] API key 不写入代码、日志或产物。

#### 3.3 新颖性和软约束

- [ ] 复用 `novelty.py` 的可微描述符；
- [ ] 增加 smooth-min/top-k soft-min；
- [ ] 实现 P–O、Fe–O 配位的 sigmoid 软计数；
- [ ] 实现最短距离、体积/原子等平滑罚项；
- [ ] 将精确约束与软约束分别命名，防止报告混淆。

建议实现位置：

```text
src/nagen/inverse/uma_guidance.py
src/nagen/inverse/uma_hull.py
src/nagen/inverse/novelty.py
src/nagen/inverse/constraints.py
```

验收门 G3：

- UMA 坐标和晶格梯度通过有限差分测试；
- 同一结构的 forward 能量与标准 UMA 调用一致；
- MP 缓存重复读取不触发网络请求且结果一致；
- 新颖性与全部软约束对 `X,L` 有有限、非零、方向合理的梯度；
- 若 UMA 梯度桥不通过，不得开始大规模 ShootingFlow，也不得未经批准改用新代理模型。

### Phase 4：实现多目标 ShootingFlow

目标：将现有 source optimization 扩展为完整的多目标、强约束推理优化。

执行项：

- [ ] 冻结 Flow Matching 全部参数并加入断言；
- [ ] 固定已通过组成预筛选的 `N,A`；
- [ ] 只把 `z_X,z_L` 注册为优化变量；
- [ ] 每一步从源噪声完整积分到终点；
- [ ] 在终点计算 `E_hull`、新颖性和软约束；
- [ ] 分别计算两个主目标对源噪声的梯度；
- [ ] 用 MGDA 合并主目标梯度；
- [ ] 用增广拉格朗日/平滑 hinge 处理硬约束；
- [ ] 加入 source-prior 或 trust-region，限制源噪声偏离基分布；
- [ ] 加入梯度裁剪、非有限值检查和学习率回退；
- [ ] 保存 `z_0`、每步指标、`z_0*` 和最终重积分结构；
- [ ] 支持中断恢复和按样本独立重试。

建议新模块：

```text
src/nagen/inverse/multiobjective_shooting.py
```

现有 `guidance.py::optimize_source` 可作为单 shooting 骨架，但必须补齐 UMA、`E_hull`、新颖性、MGDA 和增广约束。现有终点直接修正 `optimize_terminal` 不进入主流水线，只允许作为明确标注的消融实验。

每个样本至少保存：

```text
sample_id, seed, N, A, composition
z0_hash, z0_star_hash
E_UMA, E_ref_UMA, E_hull
novelty_proxy
soft/exact constraint values
objective gradient norms and cosine
MGDA weights
source drift norm
ODE/guidance steps, optimizer state
failure/retry status
```

验收门 G4：

- 模型权重在优化前后逐位或按哈希一致；
- 更新发生在源噪声而非终点坐标缓存；
- 从 `z_0*` 重积分可复现保存的最终结构；
- 双目标梯度和 MGDA 权重可审计；
- 小样本运行无图断裂、NaN、显存持续增长或隐式模型更新。

### Phase 5：Smoke test 与小规模 pilot

目标：在昂贵主实验前验证端到端收益和失败模式。

推荐规模：

- smoke：2–4 个可行 `N,A`，每个 1–2 个源噪声；
- pilot：至少 32 个独立源噪声；
- 初始 ODE 步数 12，确认稳定后用 24；关键样本用 48 复核；
- guidance 步数、学习率和增广系数通过 pilot 决定，不在结果未知前写死。

对每个相同 `z_0,N,A` 比较：

1. 无条件生成；
2. 仅约束 source optimization；
3. 约束 + 新颖性；
4. 完整 MGDA：新颖性 + `E_hull` + 约束。

可选消融：固定加权和、终点 PostOpt、不同 source-prior 强度。

验收门 G5：

- 完整方法相对相同源噪声 baseline 在至少一个主目标上有稳定改善，且另一目标没有系统性崩溃；
- 软约束改善能转化为更高的精确约束通过率；
- source drift、梯度范数和显存/耗时处于可控范围；
- 重复 seed 结果一致；
- 已根据 pilot 冻结主实验配置。

若 `E_hull` 降低仅来自异常晶格、原子碰撞或噪声远离先验，视为失败，不得进入主实验。

### Phase 6：主生成实验

目标：在冻结配置下批量获得 optimized-source 生成结构。

起始建议为 256 个独立源噪声，采用分片、幂等和可恢复执行。是否扩容由 pilot 通过率、单位 GPU 小时有效产出和用户预算决定，不能通过反复试随机种子只汇报最好结果。

执行顺序：

1. 从 `p_N` 和 `p_A(A|N)` 采样，或从可行组成池抽取 `N,A`；
2. 精确执行组成、电荷、容量和原子数预筛；
3. 确认对应 `E_ref^UMA` 已由匹配 checkpoint 的 UMA 凸包缓存提供；
4. 采样并保存原始 `z_0`；
5. 运行无条件对照；
6. 运行多目标 ShootingFlow 得到 `z_0*`；
7. 从 `z_0*` 完整重积分；
8. 保存原始、对照、优化后结构和完整轨迹日志；
9. 对失败样本按预定义规则重试，不覆盖首次失败记录。

验收门 G6：

- 所有分片有确定状态：成功、失败或明确原因；
- 不存在缺失 seed、配置、源噪声或评分来源的结构；
- baseline 与 optimized 使用相同的 `N,A,z_0` 配对比较；
- 生成阶段结束后配置保持冻结。

### Phase 7：生成后精确筛选与 UMA 完整结构弛豫

目标：把可微引导结果转换为经过物理几何验证的候选。

先进行弛豫前精确筛选，剔除明显无效、重复、组成不符或严重碰撞的结构，避免浪费 UMA 弛豫资源。随后对通过者执行：

```text
UMA + FIRE（优先消除大力） -> BFGS（精细收敛）
```

具体优化器切换、最大步数、`fmax`、应力/晶格优化方式必须写入配置。每个结构保存：

- 初始和最终结构；
- 每步能量、最大力、应力、体积；
- 优化器和收敛阈值；
- 收敛、超步数、数值失败或异常终止状态；
- UMA 模型和 fairchem/ASE 版本。

弛豫后重新执行全部精确硬约束。弛豫失败或弛豫后约束失效的结构不得进入最终候选集。

验收门 G7：

- 每个候选有完整弛豫日志和明确终态；
- 收敛标准实际满足，而非仅因达到最大步数停止；
- 弛豫后组成保持不变，晶格有效，无严重原子重叠；
- 全部解析硬约束重新通过。

### Phase 8：最终 E_hull、新颖性、去重和 Pareto 分析

目标：形成可交付候选集和可审计结论。

对所有弛豫通过结构：

- [ ] 用弛豫后结构重新计算 `E_UMA`；
- [ ] 从固定且模型哈希匹配的 UMA 凸包缓存读取 `E_ref^UMA`；
- [ ] 计算并强制 `E_hull <= 0.150 eV/atom`；
- [ ] 对训练集、验证集、测试集及同批生成物做精确结构匹配；
- [ ] 去除等价结构和重复结构；
- [ ] 计算最终新颖性指标；
- [ ] 构建 `E_hull`–新颖性 Pareto 前沿；
- [ ] 对候选按硬约束、新颖性、`E_hull` 和弛豫质量综合分层，而非只给单一总分。

输出：

```text
candidates/final_candidates.jsonl
candidates/structures/*.cif
candidates/pareto_front.jsonl
reports/final_report.md
reports/failure_analysis.md
reports/reproducibility_manifest.json
```

验收门 G8：

- 最终列表中的每个结构均通过弛豫后全部硬约束；
- 每个结构 `E_hull <= 0.150 eV/atom`；
- 无训练集等价结构和候选间重复；
- 所有数值能追溯到模型、MP 条目、配置和结构文件；
- 报告同时给出成功数量、分母、通过率和计算成本，不只展示最佳样本。

## 6. 实验矩阵

主报告至少包含以下配对实验：

| 实验 | 源噪声优化 | 新颖性 | E_hull | 软约束 | 用途 |
|---|---:|---:|---:|---:|---|
| B0 | 否 | 否 | 否 | 否 | 无条件基线 |
| B1 | 是 | 否 | 否 | 是 | 约束引导基线 |
| B2 | 是 | 是 | 否 | 是 | 新颖性贡献 |
| M0 | 是 | 是 | 是 | 是 | 主方法：MGDA + 增广约束 |

可选消融：

- M0 与固定标量加权和；
- 不同 source-prior 强度；
- ODE 12/24/48 步；
- endpoint PostOpt 与 source optimization；
- 成功弛豫主数据与包含中断样本的数据版本。

所有方法必须使用相同的 `N,A,z_0` 配对比较，并报告均值、中位数、分位数、成功率和置信区间或 bootstrap 区间。

## 7. 关键评价指标

### 7.1 训练和无条件生成

- train/val flow loss；
- 晶格有效率、坐标/PBC 有效率；
- 元素和 `N` 分布覆盖；
- 精确解析约束通过率；
- 无条件结构重复率和训练集匹配率。

### 7.2 ShootingFlow

- 配对前后的 `E_UMA`、`E_hull`、新颖性；
- 两目标梯度范数、夹角和 MGDA 权重分布；
- 软约束和精确约束通过率；
- `||z_0* - z_0||` 及相对先验偏移；
- NaN、梯度爆炸、重试和失败率；
- 单样本耗时、显存峰值和 GPU 小时。

### 7.3 后处理和最终产出

- UMA 弛豫收敛率；
- 弛豫后硬约束存活率；
- `E_hull <= 0.150` 通过率；
- 最终唯一且新颖的结构数；
- Pareto 候选数；
- 每个源噪声、每 GPU 小时的最终有效产出。

## 8. 可复现性和断点恢复

每次运行必须保存：

- 实验 ID、阶段、时间和主机/GPU；
- git commit 和 dirty diff 摘要；
- 数据集、split、规范文件和 checkpoint 哈希；
- Python、PyTorch、CUDA、ASE、pymatgen、fairchem/UMA 版本；
- 随机种子和确定性设置；
- 完整 YAML/JSON 配置；
- MP 缓存元数据及条目 ID；
- 每个样本的 `z_0`、`z_0*`、优化历史和结构文件；
- 错误堆栈、重试次数和最终状态。

批处理采用独立 sample ID 和原子写入；JSONL 每条或小批 flush。重新启动时跳过已完成且哈希一致的样本，只重跑失败或缺失样本，不覆盖旧日志。

## 9. 失败处理和停止条件

以下情况必须停止进入下一阶段：

1. `OPTIMIZATION.md` 与可执行 spec 不一致；
2. 数据 split 发生公式或结构泄漏；
3. UMA 坐标或晶格梯度未通过有限差分；
4. Flow 参数在 inference optimization 中发生变化；
5. 目标下降主要依靠原子碰撞、异常体积或源噪声严重离开先验；
6. UMA 凸包参考能缺失或模型哈希不匹配却仍将结构标为 hull-pass；
7. UMA 弛豫未收敛却作为最终候选；
8. 最终报告无法追溯模型、数据、MP 条目或 seed。
9. 竞争相与候选未使用同一 UMA checkpoint、任务头、归一化或结构处理协议。

对应回退策略：

- 梯度爆炸/NaN：降低学习率、梯度裁剪、加强 trust region、缩短反传区间；必要时评估 multiple shooting；
- 无可行 `N,A`：重新采样或从满足精确组成规则的枚举池抽取，不用坐标优化修复组成；
- 目标冲突严重：检查梯度尺度和 MGDA 实现，再进行偏好权重扫描；
- UMA 凸包缓存缺失：标记 `E_ref_UMA unavailable` 并移出强约束候选，不猜测数值；
- UMA 梯度桥失败：修复原生张量路径或 VJP，不擅自训练替代代理；
- 后弛豫存活率低：分析软约束与精确约束偏差，再调整可微近似并重新 pilot，不能直接放宽最终规则。

## 10. 预计代码变更范围

后续 `/goal` 执行时，预计涉及但不限于：

```text
src/nagen/inverse/spec.py
src/nagen/inverse/dataset.py
src/nagen/inverse/train.py
src/nagen/inverse/generate.py
src/nagen/inverse/guidance.py
src/nagen/inverse/novelty.py
src/nagen/inverse/constraints.py
src/nagen/inverse/uma_guidance.py        # 新增
src/nagen/inverse/uma_hull.py            # 新增
src/nagen/inverse/multiobjective_shooting.py  # 新增
src/nagen/inverse/relax_uma.py           # 新增或整合
tests/                                   # 同步扩充
```

所有修改均应先有小测试，再运行昂贵任务。不得修改 `/mnt/data2/shared` 或参考脚本原件。

## 11. 最终交付物

项目完成时至少交付：

1. 与 `OPTIMIZATION.md` 一致且有测试覆盖的可执行规范；
2. 经审计的 Na–Fe–P–O 数据集和无泄漏 split；
3. 无条件 Flow Matching checkpoint 和 baseline 报告；
4. 经有限差分验证的 UMA 梯度桥；
5. 可复用的 MP 竞争相结构、UMA 能量与 UMA 凸包缓存及来源清单；
6. 多目标 MGDA + 增广约束 ShootingFlow 实现；
7. 配对 baseline、消融和主实验结果；
8. UMA + FIRE/BFGS 弛豫前后结构及日志；
9. 满足全部最终硬约束的唯一新颖候选 CIF/JSONL；
10. 最终报告、失败分析和完整可复现 manifest。

## 12. 完成定义

只有同时满足以下条件，整个 `/goal` 才可标记为完成：

- G0–G8 全部通过；
- 主实验和规定基线完成配对比较；
- 最终候选完成 UMA 结构弛豫、弛豫后精确复核、`E_hull` 复算和结构去重；
- 最终候选全部满足 `E_hull <= 0.150 eV/atom`；
- 结果未被表述为 DFT 级稳定性证明；
- 所有中间和最终产物保存在规定目录并可断点复现；
- `reports/final_report.md` 清楚报告成功、失败、分母、成本和局限性；
- 不存在未解释的跳过阶段、放宽约束或缺失来源。

若最终没有候选通过，实验仍可在所有阶段和分析完整结束后视为“实验执行完成”，但必须明确结论为零产出，给出失败分解；不得通过改变既定约束制造成功候选。

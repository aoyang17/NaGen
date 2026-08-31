# ALGORITHM_PLAN: 面向 NaGen 多目标晶体优化的梯度优化方案

日期：2026-08-14

关联问题定义：[OPTIMIZATION.md](OPTIMIZATION.md)

## 1. 目标与范围

本文档汇总 NaGen 在 Na–TM–P–O 磷酸盐正极材料发现中的多目标优化算法方案，并明确：

- 不使用以 qNEHVI/贝叶斯优化为主的外层黑盒优化；
- 优化器必须与 diffusion / flow matching 生成模型耦合；
- 以梯度优化为主线，通过可微代理计算目标，反传更新生成模型的低维输入；
- 约束分为软约束惩罚和硬过滤两层；
- 后续 MatterGen 只作为 diffusion baseline 或领域适配起点，梯度方案应与条件 flow matching 兼容。

## 2. 问题定义

候选晶体：

\[
\mathcal C=(N,\mathbf A,\mathbf X,\mathbf L)
\]

其中：

- \(N\)：原子数；
- \(\mathbf A=(a_i)_{i=1}^N\)：元素序列；
- \(\mathbf X=(\mathbf x_i)_{i=1}^N,\ \mathbf x_i\in[0,1)^3\)：分数坐标；
- \(\mathbf L\in\mathbb R^{3\times3},\ \det(\mathbf L)>0\)：晶格矩阵。

优化目标：

\[
\max_{\mathcal C}\left(
f_{\mathrm{nov}}(\mathcal C),
-\Delta E_{\mathrm{hull}}(\mathcal C)
\right)
\]

\[
f_{\mathrm{nov}}(\mathcal C)
=
\min_{\mathcal C_k\in\mathcal D_{\mathrm{ref}}}
d(\phi(\mathcal C),\phi(\mathcal C_k))
\]

主要约束：

- 电中性：\(\sum_i q(a_i)=0\)
- PBC 原子最小距离
- 比容量 \(Q_{\mathrm{th}}\ge130\ \mathrm{mAh\,g^{-1}}\)
- 热力学稳定：\(\Delta E_{\mathrm{hull}}\le0.150\ \mathrm{eV\,atom^{-1}}\)
- 磷酸盐配位：P 只以 PO4 存在，TM 只以 TMO4/TMO5/TMO6 存在

## 3. 核心设计原则

1. 原始设计变量 \(N,\mathbf A,\mathbf X,\mathbf L\) 不适合直接做梯度优化；
2. 将生成模型视为可微映射，优化其组成条件、性质条件和初始 latent/noise；
3. 所有参与梯度的目标都使用可微代理或平滑形式；
4. 多目标不使用固定权重单分数，优先使用多梯度下降获得 Pareto stationary；
5. 离散变量通过连续松弛、整数化和重新采样处理；
6. 生成后仍保留精确硬约束过滤，避免把硬化学规则完全交给软惩罚。

## 4. 可微生成映射

假设条件 flow matching / rectified flow：

\[
x_t=(1-t)x_0+t x_1
\]

\[
\frac{dx_t}{dt}=v_\phi(t,x_t,c)
\]

训练目标：

\[
\mathcal L_{\mathrm{FM}}
=
\mathbb E_{t,x_0,x_1}
\left[
\|v_\phi(t,x_t,c)-(x_1-x_0)\|^2
\right]
\]

采样：

\[
\mathcal C=G_\phi(c)
=
\mathrm{ODESolve}(v_\phi,\ x_0,\ t\in[0,1])
\]

变量 \(c\) 包括：

- 组成 logits，softmax 后表示 Na/TM/P/O/F 比例；
- 目标性质条件；
- 可选初始 latent/noise。

优化时冻结 \(\phi\)，对 \(c\) 求梯度。Flow matching / rectified flow 优先，因为轨迹更直、随机性更少，反传更稳定。

## 5. 可微目标构造

### 5.1 稳定性目标

\[
f_{\mathrm{stab}}(\mathcal C)
=
-\left[
E_\theta(\mathcal C)
-
E_{\mathrm{hull}}^{(\theta)}(\mathrm{comp}(\mathcal C))
\right]
\]

其中：

- \(E_\theta\)：可微 MLIP，如 MACE、M3GNet、CHGNet 或等变 GNN；
- \(E_{\mathrm{hull}}^{(\theta)}\)：组成到凸包能量的可微代理。

### 5.2 新颖性目标

原始定义为最近参考距离。为了可导，使用 soft-min：

\[
f_{\mathrm{nov}}^{(\beta)}(\mathcal C)
=
-\frac{1}{\beta}
\log
\frac{1}{|\mathcal D_{\mathrm{ref}}|}
\sum_{k}
e^{-\beta d(\psi(\mathcal C),\psi(\mathcal C_k))}
\]

\(\psi\) 可使用：

- 多壳层 RDF/ADF；
- SOAP 风格径向/角向展开；
- 图网络结构 embedding。

参考集指纹应预计算并建立近邻索引，每次只对候选结构回传梯度。

## 6. 约束处理

### 6.1 软约束惩罚

- 最小原子距离：

\[
\mathcal L_{\mathrm{dist}}
=
\sum_{i<j}
\mathrm{softplus}(d_{\min}(a_i,a_j)-d_{ij}^{\mathrm{PBC}})
\]

- 配位数软约束：

\[
\hat n_i=\sum_j\sigma(\kappa(r_{\mathrm{bond}}(a_i,a_j)-d_{ij}))
\]

\[
\mathcal L_{\mathrm{coord}}
=
\sum_{i:P}(\hat n_i-4)^2
+
\sum_{i:TM}\min_{k\in\{4,5,6\}}(\hat n_i-k)^2
\]

- 容量：

\[
\mathcal L_{\mathrm{cap}}
=
\mathrm{softplus}(130-Q_{\mathrm{th}}(\mathcal C))
\]

- 电中性，连续组成下：

\[
\mathcal L_{\mathrm{charge}}
=
\left(\sum_a q(a)p_a\right)^2
\]

### 6.2 硬过滤

梯度优化结束后，离散化组成并精确检查：

- 电中性；
- PBC 最小距离；
- P 配位严格为 4；
- TM 配位严格属于 \(\{4,5,6\}\)；
- 容量和 \(\Delta E_{\mathrm{hull}}\) 阈值；
- 结构去重和 novelty。

## 7. 多目标梯度聚合

### 7.1 主方案：MGDA

对每个候选求：

\[
g_1=\nabla_c f_{\mathrm{nov}},\qquad
g_2=\nabla_c f_{\mathrm{stab}}
\]

求 Pareto stationary 方向：

\[
\min_{w\in[0,1]}
\left\|
w g_1+(1-w)g_2
\right\|^2
\]

更新：

\[
c\leftarrow c-\eta(w^*g_1+(1-w^*)g_2)
-\eta_{\mathrm{pen}}\nabla_c\mathcal L_{\mathrm{pen}}
\]

两目标时权重子问题是一维二次问题，可闭式求解。

### 7.2 辅助方案：标量化

可并行多个标量权重扫描：

\[
\mathcal L_\alpha
=
(1-\alpha)\hat f_{\mathrm{nov}}
+
\alpha\hat f_{\mathrm{stab}}
-
\lambda\mathcal L_{\mathrm{pen}}
\]

注意线性加权只覆盖凸包部分 Pareto front，最终仍需非支配排序筛选。

## 8. 离散变量处理

- 固定组成模式：\(N,\mathbf A\) 固定，只优化 \(c\) 和生成几何；
- 开放组成模式：用 softmax 组成 logits 表示组成，优化后整数化；
- 元素序列 \(\mathbf A\) 不逐位优化，由 conditional flow model 根据组成采样；
- 需要时可用 Gumbel-Softmax 做元素松弛，但不作为第一阶段重点；
- \(N\) 不直接对整数求导，通过连续组成、整数化和重新采样处理。

## 9. 算法主循环

```text
输入：
  - 预训练条件 flow matching 模型 v_phi
  - 可微 MLIP E_theta
  - hull surrogate
  - 参考指纹库

输出：
  - Pareto 晶体档案

for round in 1..R:
    # 1. 低维变量
    c = (composition_logits, property_conditions, latent/noise)

    # 2. 可微生成
    C = FlowSample(v_phi, c)

    # 3. 目标
    f1 = softmin_novelty(C, reference_fingerprints)
    f2 = -E_hull(C, E_theta, hull_surrogate)

    # 4. 软约束
    penalty = distance + coordination + capacity + charge

    # 5. 多目标梯度
    g1 = grad(f1, c)
    g2 = grad(f2, c)
    w  = solve_MGDA(g1, g2)
    c  = c - lr * (w*g1 + (1-w)*g2 + lambda_pen*grad(penalty,c))

    # 6. 投影与离散化
    c = project_to_domain(c)
    formula = discretize(composition_logits)

    # 7. 硬过滤与归档
    candidates = hard_filter(C)
    archive = update_pareto(archive, candidates)

    # 8. 可选回流
    optionally_retrain_flow_or_surrogate(archive)
```

## 10. 模型依赖与当前缺口

### 必须建立的模型

1. **Conditional flow matching generator**

   从组成、性质和 latent 生成连续几何。可先用 MatterGen 作为 diffusion baseline，后续切换或并行自研 flow matching。

2. **可微 MLIP**

   计算结构总能量和几何梯度。优先采用 MACE、M3GNet、CHGNet 等预训练模型，后续用 Na 磷酸盐数据微调。

3. **E_hull(composition) surrogate**

   用于计算：

   \[
   \Delta E_{\mathrm{hull}}
   =
   E(\mathcal C)
   -
   E_{\mathrm{hull}}(\mathrm{comp}(\mathcal C))
   \]

   当前原始数据没有 `energy_above_hull`，这是首要标签缺口。

### 可暂不训练的组件

- novelty 指纹：先用 RDF/ADF/SOAP 等确定性特征；
- 硬约束过滤：先用规则和结构工具精确计算；
- 容量：组成和氧化态明确时可解析计算；
- feasibility classifier：软约束不足时再训练。

## 11. 当前数据观察

来源：`/mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl`

- 记录数：17,774；
- formula 数：564；
- 原子数：min 20，median 91，max 184，p90 174；
- 主要元素：Na/P/O 和 TM=Fe/Mn/Ti/V，另有 F，少量 N/C；
- `energy_above_hull`：全空；
- `optimized_energy_per_atom`：17,774 条均有值，范围约 -7.84 到 -6.06 eV/atom；
- Q/P–O 聚类：Q1 最多，Q0 次之；PO4 完整性总体较高。

含义：

- 原始结构多为 conventional/supercell，需要 primitive reduction 后再评估训练尺度；
- `optimized_energy_per_atom` 不能直接当 hull 使用；
- 数据规模适合 domain adapter 或领域微调，不适合从零训练通用 MatterGen。

## 12. MatterGen 作为 diffusion baseline 的可行性

### 可行路径

1. Zero-shot MatterGen 采样作为 baseline；
2. 用 domain adapter fine-tune 固定主骨干，训练 chemical-system / 组成条件；
3. adapter 不足时，低学习率 full fine-tune 消融；
4. 有 hull 标签后，再考虑 property-conditioned fine-tune。

### 不推荐

- 从零训练 MatterGen；
- 在 hull 标签未定义前直接做 `energy_above_hull` 条件微调；
- 原始大超胞不做 primitive reduction 直接训练。

### 算力参考

| 任务 | 推荐算力 |
|---|---|
| 仅采样 baseline | NVIDIA T4 16GB 或 A100 40GB |
| Adapter fine-tune | 单 A100 80GB，或双 A40 48GB |
| Full fine-tune | 2–8 张 A100 80GB DDP |
| 生成后 MLFF 松弛 | 单 A100 40GB + MatterSim 或其他 MLFF |
| 数据预处理 | 16–64 CPU cores，64–128GB RAM |

训练尺度：

- 有效 batch size 约 512；
- 推荐 bf16/mixed precision；
- primitive 后原子数仍偏大时，降低 per-GPU batch，增加 gradient accumulation；
- adapter fine-tune 17.7k、50–100 epoch，单 A100 80GB 预估数小时到 1–2 天；
- full fine-tune 预估 1–4 天或更长，具体取决于原子数和配置。

## 13. 分阶段实施

### Phase 0：数据治理与 primitive 统计

- primitive reduction + Niggli；
- 元素、原子数、重复结构和缺失值统计；
- 生成 MatterGen CSV/cache；
- 按 composition/prototype 划分 train/val/test。

### Phase 1：MatterGen 在我们数据上的 fine-tune 步骤

这一步专门回答“怎么用 MatterGen 在自己的 NaGen 数据上做 fine-tune”。先做 domain/chemical-system 适配，不做 `energy_above_hull` 条件，因为当前数据没有 hull 标签。

#### Step 1：建立可写环境与依赖

- 在 `/mnt/data2/aobo` 下建 Conda 环境，项目内只放 symlink；
- 安装 MatterGen、pymatgen、PyTorch CUDA、MatterSim 或其他 MLFF；
- 下载官方 checkpoint，例如 `mattergen_base`。

#### Step 2：把 JSONL 转成 MatterGen 数据格式

- 流式读取 `/mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl`；
- 每个 `structure` 转成 `pymatgen.Structure`；
- 做 primitive reduction + Niggli reduction；
- 元素先限制到 Na/TM/P/O，并决定是否保留 F、剔除 C/N；
- 输出训练 CSV，字段至少包含：
  - `structure_id`
  - `atomic_numbers`
  - `pos`
  - `cell`
  - `num_atoms`
  - `chemical_system`
  - 可选 `composition`
- 用 MatterGen 的 `csv-to-dataset` 或自写 converter 生成 cache。

#### Step 3：划分与去重

- 按 composition/prototype 分组划分 train/val/test；
- 禁止随机按行切分；
- 记录 split 文件，方便后续评测复用。

#### Step 4：跑 zero-shot baseline

固定一个或几个目标组成，先用未微调的 `mattergen_base` 采样：

```bash
mattergen-generate \
  --pretrained_name mattergen_base \
  --batch_size 64 \
  --target_compositions '[{"Na":1,"Fe":1,"P":1,"O":4}]'
```

保存 baseline 结构，用于和微调后模型比较 validity、SUN、RMSD、配位满足率。

#### Step 5：做 adapter fine-tune

优先使用官方 adapter 路径，冻结主骨干，只训 property/domain adapter：

```text
mattergen-finetune
  adapter.pretrained_name=mattergen_base
  data_module=<nagen_dataset_config>
  trainer.devices=1
  trainer.accelerator=gpu
  trainer.precision=bf16-mixed
  full_finetuning=false
```

需要改的关键训练参数：

- effective batch size 目标约 512；
- per-GPU batch 视 80GB 显存和 primitive 后原子数调整；
- gradient accumulation 不够再增大；
- 学习率先用 MatterGen 默认，再在 `1e-5` 到 `1e-4` 左右小范围扫；
- epoch 先设 50，用 validation loss 和结构有效性选 checkpoint。

#### Step 6：评测微调结果

用和 zero-shot 完全相同的采样预算，报告：

- validity；
- stability；
- uniqueness；
- novelty；
- 固定组成命中率；
- P 配位 =4、TM 配位 ∈ {4,5,6} 的通过率；
- 生成结构松弛前后的 RMSD。

#### Step 7：决定是否 full fine-tune

adapter 如果明显不够，再做 `full_finetuning=true`，用更低学习率更新主骨干，对比：

- 结构有效性；
- 领域分布覆盖；
- 是否丢失通用晶体先验。

full fine-tune 不要作为第一步，避免 17.7k 小数据上过拟合。

#### Step 8：条件能力扩展

只有等 `energy_above_hull` 标签、容量标签可信后，再训练 property-conditioned adapter；本阶段先只做 chemical-system / 固定组成 CSP 条件。

### Phase 2：单目标梯度验证

- 固定一个组成；
- 用可微 MLIP 做稳定性梯度优化；
- 验证 flow sample 反传、软约束和硬过滤。

### Phase 3：双目标 Pareto 梯度优化

- 加入 soft novelty；
- 使用 MGDA 或标量化扫描；
- 输出 Pareto 档案。

### Phase 4：开放组成优化

- 连续组成 logits 优化；
- 整数化 formula；
- 固定 formula 后重新采样和精修。

### Phase 5：MatterGen / flow model 微调闭环

- 先用 MatterGen adapter 做 domain baseline；
- 用 Pareto 档案和高价值候选回流微调生成模型。

## 14. 计划中的代码模块

```text
NaGen/src/nagen/
├── models/
│   └── flow.py                     # 条件 flow matching 采样与可微 FlowSample
├── objectives/
│   ├── stability.py                # 可微 MLIP + hull surrogate
│   └── novelty.py                  # 指纹 + 参考索引 + soft-min
├── optimization/
│   ├── mgda.py                     # 双目标 MGDA
│   ├── constraints.py              # 软约束与硬投影
│   └── pareto_flow_opt.py          # 主循环
└── evaluation/
    └── pareto_metrics.py           # 非支配排序、hypervolume、SUN 指标
```

## 15. 待确认决策

1. `D_ref` 具体指训练集、MP/ICSD，还是已知 Na 磷酸盐库；
2. novelty 指纹和重复阈值；
3. `E_hull` 标签来源和能量口径；
4. 是否保留 F，是否剔除 C/N；
5. 第一阶段使用 MatterGen diffusion，还是直接训练条件 flow matching；
6. 组成优化模式：固定组成优先，还是直接开放组成。

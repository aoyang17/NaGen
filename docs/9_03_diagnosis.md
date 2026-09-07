# NaGen 2026-09-03 诊断计划与路线修正

> 记录时间：2026-09-03
> 状态：本文取代 9/1–9/3 的求解器横扫路线，作为下一阶段的正式执行顺序
> 本文是诊断计划；结论以各步产出的 `reports/*.json` 和逐样本记录为准

## 1. 核心结论

**瓶颈在生成模型，不在优化器，也不在凸包基础设施。**

支撑证据来自 `experiments/exp_2026-08-26/reports/pilot_summary.json`，无引导 B0 的失败构成不是能量，是几何：

| 指标 | config 0 | config 1 | config 2 | config 3 |
|---|---|---|---|---|
| `median_minimum_distance_A` | 0.335 | 0.333 | 0.484 | 0.362 |
| `minimum_pbc_distances` 通过 | 0/8 | 0/8 | 0/8 | 0/8 |
| `p_coordination_eq_4` 通过 | 0/8 | 0/8 | 0/8 | 0/8 |
| `median_E_hull_eV_atom` | 22.7 | 23.0 | 26.2 | 29.5 |

> **口径警告（9/3 追加）**：上表 `E_hull` 一行是**被禁的跨能标口径**（见 2.10），绝对值不可用于与 `0.15` 阈值比较。但本节结论**不依赖**它——结论建立在 `median_minimum_distance_A` 与各项几何通过率上，这些是纯几何量，与能量口径无关。配对 delta `E_hull` 同样不受影响（口径偏移在固定组成下是常数，作差时抵消）。

最近原子间距中位数 `0.33 Å`，而正常成键在 `1.4–2.5 Å`。`22 eV/atom` 不是化学信息，是原子重叠产生的 Pauli 排斥。**这些输出不是"离凸包很远的结构"，它们不构成结构。**

引导之后（B1/B2/M0，4 组配置共 96 条引导路径）：

- 最近距离推到 `1.2–1.39 Å`，`minimum_pbc_distances` 仍然 **0/8，无一例外**
- `p_coordination_eq_4` 在 96 条路径中**从未通过过一次**
- `analytic_without_hull_pass` 全部为 `0`

### 1.1 因果链

```
模型输出原子重叠 (0.33 Å)
  → 推理期 distance restoration 补出"看起来合法"的几何 (17/32 → 22/32)
  → 但结构完全不在 UMA 弛豫流形上
  → 弛豫把配位打散 (19/19 力收敛，但严格几何仅 1/19)
  → 统一优化要求解落在 PES 极小点上
  → 求解器只能落到能量随机的孤立临界点
  → E_hull 0.4–1.4 平台，所有求解器、所有种子一致
```

配对 delta `E_hull` 为 **−18 到 −26 eV/atom**。**优化器把能量搬下来了二十多个 eV/atom，它不是瓶颈。** 它做不到的最后一段是合法性与拓扑问题，梯度里不存在这个方向。

### 1.2 流程问题

`pilot_summary.json` 自身记录：

```json
"pilot_gate_passed": false,
"main_256_authorized": false,
"stop_reason": "all four M0 configurations have zero exact analytic-constraint
                passes; minimum-distance and P-coordination failures remain"
```

按 `docs/8_27_progress.md` 第 4 节恢复顺序第 7、8 条，**G5 通过是冻结配置和启动主实验的前置条件。G5 未通过。** 9/1–9/3 的 11 组求解器横扫建立在一个未通过的生成门之上，它们一致收敛到同一平台有了完整解释，不是巧合。

## 2. E_hull 口径问题（2026-09-03 新增）

### 2.1 数学是对的

`src/nagen/inverse/uma_hull_campaign.py:560,581`：

```python
diagram = PhaseDiagram(entries)                  # 560 个全 UMA 弛豫竞争相
reference = diagram.get_hull_energy(composition) / composition.num_atoms
```

`get_hull_energy` 是标准凸包下包络求值，在组成单纯形上连续、分片线性、处处有定义。不存在"用查表代替凸包"的问题。

### 2.2 但连续对象在构建期被丢弃了

```python
for item in pool["unique_compositions"]:     # 只在有限组成池上采样
    compositions[composition.reduced_formula] = {"reference_energy_eV_atom": reference, ...}
```

缓存只保留 `{reduced_formula → 能量}` 字典；`UMAHullCache.get()` 是一次 dict 查表，未命中抛 `MissingUMAHullReferenceError`。后果：

| 编号 | 后果 | 说明 |
|---|---|---|
| a | **组成池外无定义** | 模型自由采样 `p(N,A)`，池外组成得到的是异常而非数值 |
| b | **梯度信息被销毁** | 真实 `E_ref(x)` 分片线性、对组成有非零梯度；离散成表后分片常数，`∂E_ref/∂z_A ≡ 0` |
| c | **指标混淆两件事** | `E_UMA` 是结构质量（连续、可归因于模型），`E_ref` 是组成落格（离散、与模型无关）；平台无法归因 |

> **更正（9/3）**：本文早先据 (b) 断言"此前记录的『组成被冻结』病理根源在此"。**该断言错误。** 8/26 pilot 走 `paired_shooting_campaign.py`，其源由 `_campaign.fixed_composition_source` 构造（A = `one_hot × 2.0`），且 `integrate_flow(..., freeze_atom=True)`，另有 `fixed_A_preserved` 校验——**组成在该 campaign 中是固定输入，不是决策变量，"冻结"是设计意图而非病理。** (b) 对 pilot 不适用。
>
> (b) 的适用范围是 `docs/HybridOptimization.md` 9/3 定义的统一优化问题：那里 `z_A` **是**决策变量。届时查表的零梯度会成为真问题，`DifferentiableUMAHull`（2.5）正是为此准备。(a) 与 (c) 对两种设定都成立。

### 2.3 provenance bug〔2026-09-03 已修复〕

`uma_hull_campaign.py:626` 原先将缓存的 `definition` 字段写为：

```
"definition": "E_hull=E_UMA+E_MP2020_correction-E_MP_hull"
```

而 `get()` 实际返回的 `reference_energy_eV_atom` 是全 UMA 口径。**文件自我描述的是被 8/27 明令禁止的跨能标定义，实际取用路径是正确的。** 这是已观察到的口径分歧的确切来源。

已改为 `E_hull=E_UMA_candidate-E_UMA_hull`，并新增 `official_reference_field` 与 `secondary_cross_scale_definition` 两个字段，把"审计用"和"打分用"在文件层面分开。**注意：该改动只影响此后新构建的缓存，现存缓存文件内的字符串不变**（缓存重建需要重跑弛豫，8/27 第 4 节禁止覆盖已完成记录）。

`UMAHullCache.get_mp2020_reference()` 已改为必须显式传 `acknowledge_cross_scale=True`，否则抛 `ValueError`。**这会让任何仍走跨能标口径的 9/1–9/2 代码路径立即硬失败**——这是预期效果：把静默口径漂移变成可见异常。

### 2.4 修正设计

`E_ref` 在单纯形上是分片线性凸函数（下凸包是凸函数，凸函数等于其全部支撑超平面的逐点上确界），可写为 facet 线性函数取最大：

```
E_ref(x) = max_f ( a_f · x + b_f )
E_hull(C) = E_UMA(C) − max_f ( a_f · x(C) + b_f )
```

收益：

- 全单纯形有定义，消除 missing-key
- 连续、几乎处处可微
- 配合软组成 `x(z_A) = mean(softmax(logits))`，`E_hull` 对 `z_A` 端到端可微，直接解掉组成冻结
- 运行期同时记录 `E_UMA / E_ref / decomposition` 三项（`decomposition` 缓存中已有），使平台可归因

### 2.5 关键发现：对的数学在错的标尺上

`src/nagen/inverse/differentiable_mp_hull.py` **已经实现了 2.4 节的全部数学**——facet 平面提取、`max_f(a_f·x+b_f)`、甚至带 log-sum-exp 平滑温度。

但它建在**被禁的 MP2020 标尺**上：`from_phase_cache` 要求 `payload["compatibility"] == "MaterialsProject2020Compatibility"`，且 `raw_e_hull` 额外叠加 `candidate_correction`（Fe = −2.256、O = −0.687 eV/atom·fraction），即 `E_UMA + MP2020_correction − E_MP_hull`。

于是真实状况是：

| 模块 | 标尺 | 数学 |
|---|---|---|
| `uma_hull.py` | ✅ 全 UMA（正式） | ❌ 离散查表，`∂E_ref/∂z_A ≡ 0` |
| `differentiable_mp_hull.py` | ❌ MP2020 跨能标（禁止） | ✅ 连续分片线性可微 |

**对的数学在错的标尺上，对的标尺用了错的数学；两者的交集从未构建过。** 这解释了为什么"离散查表"这个问题一直没被修——修它的代码其实存在，只是长在被禁的那一支上。

而 `uma_hull_campaign.py:560/574` 同时构建 `diagram`（全 UMA）与 `mp_diagram`（MP2020），每个成分同时缓存 `reference_energy_eV_atom` 与 `mp_hull_reference_eV_atom`，`definition` 字段又写的是后者——9/1–9/2 漂到 `calibrated_mp2020` 的完整路径至此闭合。

**已补上交集**：`src/nagen/inverse/differentiable_uma_hull.py`，`DifferentiableUMAHull.from_hull_cache()` 从缓存的 `support_entries` 重建同一张全 UMA `PhaseDiagram`，导出 facet 平面，**不含任何 compatibility correction**（同能标下按定义不需要）。标尺与支撑集都取自缓存本身，不存在第二份 provenance 可漂移。

## 2.6 训练/推理源分布不一致（H1f，2026-09-03 新增，当前最强线索）

训练时 X 通道的源（`src/nagen/inverse/model.py:216-221`）：

```python
source_frac = torch.remainder(
    target_frac + coordinate_noise_sigma * torch.randn_like(target_frac), 1.0
)   # coordinate_noise_sigma = 0.50
```

推理时 X 通道的源（`src/nagen/inverse/sample.py:57`，`random_source`）：

```python
torch.rand(batch_size, n_atoms, 3, dtype=dtype, device=device)   # i.i.d. 均匀
```

**两者不是同一个分布。** 训练源是**真实晶体的加噪副本**——它保留了位点对应关系，其最近原子间距分布继承自真实结构；推理源是**无结构的 i.i.d. 均匀点云**，在一个几十原子的原胞里必然出现大量亚埃接触。

`coordinate_noise_sigma` 全仓库只出现在 `model.py:204`（默认值）和 `model.py:220`（使用处），**在任何推理路径中都不存在**。

代码注释写的理由是"wrapped N(0, 0.5²) 的环面边缘分布近似均匀"。**边缘分布近似均匀不等于联合分布近似均匀**：`wrap(x + 0.5ε)` 的每个坐标分量边缘确实接近均匀，但各原子之间保留了目标结构诱导的强相关，而 `torch.rand` 各原子独立。rectified flow 学到的速度场只承诺搬运**训练源的边缘分布**；换一个源分布，输运保证直接失效。

这可能直接解释 `0.33 Å`：模型从未见过带亚埃接触的输入，因此把它们原样放行。若成立，则 P4 的重训与扩数据是**纯浪费算力**——问题在推理入口的三行，不在容量。

`tools/p1_5_source_mismatch.py` 用纯 CPU、不加载模型来量化：对同一批打包数据集分别计算 (c) target、(b) `wrap(target+0.5ε)`、(a) `Uniform` 三种点云的最小周期原子间距分布，最小镜像约定与 `model.CrystalVectorField._radial_features` 一致。**判据**：若 (a) 复现观测到的 `0.33 Å` 中位数而 (b)(c) 远高于它，H1f 坐实。

### 2.6.1 三个通道逐一对照——只有 X 错了

`sample.random_source`（无条件采样路径）：

| 通道 | 训练源（`model.py`） | 推理源（`sample.py`） | |
|---|---|---|---|
| A | `torch.randn_like(target_atom)` :216 | `torch.randn(B,N,V)` :56 | ✅ 一致 |
| **X** | **`wrap(target_frac + 0.5·randn)` :219** | **`torch.rand(B,N,3)` :57** | ❌ **不一致** |
| L | `torch.randn_like(target_lattice)` :222 | `torch.randn(B,6)` :58 | ✅ 一致 |

A 和 L 两个通道都被正确地写成了同分布，**唯独 X 没有**。这不像是深思熟虑的设计选择，更像是在给 X 加"位点对应耦合"时忘了同步推理入口。三选二正确这个模式本身就是证据。

**8/26 pilot 走的不是这条路径**，而是 `_campaign.fixed_composition_source`（组成条件生成）。**但 X 通道的错配在那条路径上同样存在**——见 2.9 的逐通道核对。两条推理路径共享同一个缺陷。

## 2.9 `geometry_only` 与 `freeze_atom`：不是 bug，是组成条件生成（H1h 已排除）

`_campaign.fixed_composition_source` 构造的源：

```python
atom = F.one_hot(indices - 1, num_classes=vocab).to(dtype) * 2.0    # 目标组成
frac = torch.rand(1, N, 3, generator=generator)                     # i.i.d. 均匀
lattice = torch.randn(1, 6, generator=generator)
```

配合 `integrate_flow(..., freeze_atom=True)` 与 `_exact` 里的 `fixed_A_preserved` 校验。

这与 `flow_matching_loss(geometry_only=True)` 的语义**完全自洽**：训练时 `source_atom = target_atom = one_hot × 2.0`（`type_scale` 同为 `2.0`）、`atom_t` 全程不插值、`v_A` 监督恒为 0；推理时 A 也取 `one_hot × 2.0` 并冻结。**A 通道没有错配，`v_A ≡ 0` 是被正确利用的设计，不是缺陷。** H1h 排除。

**但同一段代码确认了 H1f**：X 源是 `torch.rand`，而训练 X 源是 `wrap(target + 0.5·randn)`。三个通道逐一核对的最终结论：

| 通道 | 训练源 | pilot 推理源 | |
|---|---|---|---|
| A | `target_atom`（`geometry_only`） | `one_hot × 2.0` + 冻结 | ✅ 一致 |
| **X** | **`wrap(target + 0.5·randn)`** | **`torch.rand`** | ❌ **错配** |
| L | `randn_like` | `randn` | ✅ 一致 |

**唯一的源分布错配是 X，且它就在产出 `0.33 Å` 的那条确切代码路径上。**

## 2.10 pilot 实际跑在被禁口径上（9/3 确认并已修复）

`multiobjective_shooting.CrystalDesignProblem` 原先的 `E_hull`：

```python
hull = uma + self.mp2020_correction_eV_atom - (
    self.mp2020_hull_reference_eV_atom          # campaign 无条件传非 None
    if self.mp2020_hull_reference_eV_atom is not None
    else self.uma_hull_reference_eV_atom        # 正确口径，死代码
)
```

`paired_shooting_campaign.py:205` 无条件 `mp_ref, mp_corr = hull_cache.get_mp2020_reference(...)` 并传入，**因此回落到全 UMA 口径的分支从未执行过**。

调用点全量核查（4 个 campaign）：

| 文件 | 传 MP2020 参数 | 口径 |
|---|---|---|
| `paired_shooting_campaign.py` | ✅ 传 | ❌ 跨能标（被禁） |
| `pilot_campaign.py` | ✖ 不传 | ✅ 全 UMA |
| `geometry_baseline.py` | ✖ 不传 | ✅ 全 UMA |
| `shooting_spike.py` | ✖ 不传 | ✅ 全 UMA |

**即：口径分歧不始于 9/1，8/26 pilot 就在被禁口径上，且恰好是产出 `pilot_summary.json`、决定 G5 门的那一个。**

失效范围（务必精确，不要过度作废）：

| 量 | 是否受影响 |
|---|---|
| `median_minimum_distance_A`、各几何通过率、`analytic_without_hull_pass` | ✅ **不受影响**（纯几何） |
| 配对 delta `E_hull`（−18~−26 eV/atom） | ✅ **不受影响**（固定组成下口径偏移是常数，作差抵消） |
| 同一 sample 内的轨迹排序 | ✅ 不受影响（同上） |
| **绝对 `E_hull` 值、与 `0.15` 阈值的比较、`hull_threshold` 检查、`analytic_with_hull` 通过率** | ❌ **口径错误，作废** |
| G5 门结论 `pilot_gate_passed: false` | ✅ 不受影响（`analytic_without_hull_pass` 已全为 0，与 hull 无关） |

**已修复**：`CrystalDesignProblem` 删除跨能标项，`E_hull = uma − uma_hull_reference_eV_atom`；两个 MP2020 参数保留在签名中但传入即抛 `ValueError`，使任何残留调用者硬失败而非静默改变能标；`paired_shooting_campaign.py` 的传参已移除。

> ⚠️ **这是对实验代码的行为变更，需要你确认后才能重跑。** 另三个 campaign 行为不变。`paired_shooting_campaign` 重跑将产生与既有记录**不同口径**的 `E_hull`，二者不可混入同一份汇总；按 8/27 第 4 节，既有已完成记录不得删除或覆盖，应写入新的 campaign 目录。



## 2.7 最小镜像约定不一致（H1g，2026-09-03 新增）

**模型**算原子间距（`model.py:127-130` `_radial_features`，以及 `flow_matching_loss` 内的 `pair_distances`）：

```python
delta = delta - torch.floor(delta + 0.5)      # 只取 [-0.5,0.5)^3 这一个镜像
```

**约束检查器**算同一个量（`constraints.py:18-31` `pbc_distance_matrix`）：

```python
delta -= np.floor(delta + 0.5)
for i in (-1., 0., 1.):                        # 再搜 27 个镜像
    for j in (-1., 0., 1.):
        for k in (-1., 0., 1.):
            best_squared = np.minimum(best_squared, ...)
```

对斜切原胞，真正最近的周期镜像可以落在 `[-0.5,0.5)^3` 之外。朴素最小镜像是在更小的候选集上取 min，故恒有

```
d_minimum_image  ≥  d_exact
```

**方向正好指向病理**：模型系统性地**高估**近距离——它以为原子比实际更远，于是学会容忍那些检查器随后判定为违规的接触。晶体原胞常常强烈斜切，这个间隙不是理论上的小量。

注意这把偏松的尺子同时污染两处：`_radial_features` 决定模型**看到**什么，`flow_matching_loss` 里的 `pair_distance_loss` 决定模型**被监督成**什么。两处用的都是同一个朴素约定。

H1g 与 H1f 不互斥、可叠加：H1f 说模型在推理时拿到了没见过的输入分布，H1g 说模型即使在训练时也在用一把偏松的尺子量几何。

已并入 `tools/p1_5_source_mismatch.py`：对打包数据集同时用两种约定算最小距离，报告 `gap = d_min_image − d_exact` 的中位数、最大值、非零比例、超 `0.1 Å` 比例。

## 2.8 `quality_dataset.py` 与 `constraints.py` 名字对不上（H1c 未排除）

`quality_dataset.py:28-38` 的 `GEOMETRY_CHECKS` 有 9 项，`evaluate_feasibility` 返回的 `checks` 里**只有 4 项对得上**：

| GEOMETRY_CHECKS | 当前 evaluate_feasibility |
|---|---|
| `fractional_coordinate_domain` | ✅ 同名 |
| `positive_lattice_determinant` | ✅ 同名 |
| `minimum_pbc_distances` | ✅ 同名 |
| `p_coordination_eq_4` | ✅ 同名 |
| `element_domain` | ❌ 现为 `element_support_exact` |
| `positive_atom_count` | ❌ 现为 `atom_count_range` |
| `tm_coordination_in_4_5_6` | ❌ 现为 `fe_coordination_in_4_5_6` |
| `p_all_bonded_neighbors_oxygen` | ❌ **已整个消失** |
| `tm_all_bonded_neighbors_oxygen` | ❌ **已整个消失** |

而 `_evaluate` 是直接下标取的：`feasibility["checks"][name]`。**所以 `quality_dataset.py` 现在跑不起来，第一条结构就 KeyError。**

推论：在用的打包数据集是**另一个代码版本**建的（在 TM 泛化 → Fe 专用那次重构之前），且两个"成键邻居必须是氧"的检查在新评估器里被删掉了。建集时实际施加的几何定义与今天 pilot 评估候选用的定义**不是同一套，且无法从当前代码确认差异**（`git` 在本机不可用，见第 7 节）。

**因此撤回此前一条结论**：本文早先据 `GEOMETRY_CHECKS` 含 `minimum_pbc_distances` / `p_coordination_eq_4` 判定"训练集按构造是干净的，H1c 在数据集层面排除"。该判据依赖两边名字同义，其前提现已证伪。**H1c 恢复为未排除**，必须由 P1.2 用实际打包数据回答，不能靠读代码。

## 3. 假设树

| 编号 | 假设 | 优先级 |
|---|---|---|
| **H1** | **生成模型有实现级 bug（非欠训）** | **最高** |
| H1a | 分数坐标未周期 wrap / 损失未定义在环面上 | |
| H1b | 晶格尺度与坐标尺度耦合错误（frac/cart 混用） | |
| H1c | 数据预处理阶段即已损坏（模型无辜） | **未排除，见 2.8** |
| H1d | ODE 积分空间或步数问题（当前 24 步） | |
| H1e | 训练/推理路径不一致 | |
| **H1f** | **训练/推理源分布不一致（X 通道）** | **最高，见 2.6** |
| **H1g** | **最小镜像约定不一致：模型朴素、检查器搜 27 镜像** | **最高，见 2.7** |
| **H1h** | ~~`geometry_only` ⇒ 组成冻结~~ | **已排除，见 2.9** |
| **H5** | **口径污染：pilot 实跑在被禁跨能标口径上** | **已确认并修复，见 2.10** |
| H2 | 欠训 / 容量不足（2,357 结构，hidden 192 / 5 层 ≈ 2–3M 参数；17,774 未用） | 中 |
| H3 | 训练流形 ≠ 约束流形（训练数据为 MP DFT 弛豫几何，要求输出落在 UMA 弛豫流形） | 中 |
| H4 | 可行集近乎空（novelty ∧ E_hull≤0.15 ∧ 全部硬约束；截断脆性） | 中 |
| H5 | 度量口径污染（见第 2 节） | 已定位 |
| H6 | Flow 链路有损（ODE 不可逆或重建误差大） | 待测 |

`0.33 Å` 是训练数据中最显著的一阶特征，训练正常的晶体模型会最先学会它。中位数 `0.33 Å` 的形态更像 bug 而非欠训，故 H1 优先级最高。

## 4. 诊断级联

原则：每步可证伪，失败即止损。P0 / P1 / P3 互相独立可并行；P2 依赖 P1；P4 依赖全部。

### P0 — 度量可信性〔约 1 小时〕

必须最先执行：它是量所有其他结论的尺子。

- **P0.1** grep `get_mp2020_reference` 与 `UMAHullCache` 全部调用点，结合 9/1–9/3 各 run 的 manifest/config，确认 11 组实验实际走的路径〔**未做，Bash/Agent 被安全分类器阻塞**〕
- **P0.2** **已知稳定相自检**：`mp-19226` (NaFePO₄, MP E_hull=0.0)、`mp-556984` (Na₃Fe₂(PO₄)₃, 0.0026) 等，走与候选**完全相同**的 pipeline，在正式全 UMA 口径下打分〔待跑〕
- **P0.3** 修正 `definition` 字段；封死 `get_mp2020_reference()`〔**已完成，见 2.3**〕
- **P0.4** 按 2.4 节导出 facet 系数，实现连续可微 `E_ref`，与 `PhaseDiagram.get_hull_energy` 交叉校验〔**代码已写完，见 2.5；验收脚本 `tools/p0_4_uma_hull_facets.py` 待跑**〕

`tools/p0_4_uma_hull_facets.py` 是 P0.4 的验收测试，纯 CPU、不需模型和 GPU，回答三问：

| 检验 | 判据 |
|---|---|
| **精确性** | `max_f(a_f·x+b_f)` 是否对每个池内成分复现缓存的 `reference_energy_eV_atom`，容差 `1e-6 eV/atom`。超出即说明平面提取与 `get_hull_energy` 语义不符，**替换不安全** |
| **梯度** | `∂E_ref/∂fractions` 是否真的非零——这正是要修的缺陷，故断言而非假定 |
| **覆盖** | 在单纯形上随机采 2000 点，统计查表的 missing 率，量化此前被静默拒绝的搜索空间比例 |

`safe_to_swap = 精确性 ∧ 梯度`，即连续版是查表的严格超集时才允许替换。

**止损门**：已知稳定相不落在 ≈0.0 → 一切暂停先修口径。现有 `uma_mp_zero_point.json` 中 `passed=false` 那条是跨能标诊断，**不能替代此自检**。

### P1 — 模型是 bug 还是欠训〔约 1–2 小时，不训练〕

信息量最高、成本最低。**从 P1.5 开始。**

- **P1.5**（先做，`tools/p1_5_source_mismatch.py`）**训练/推理源分布对照**：见 2.6
  → 若均匀源复现 `0.33 Å` 而训练源远高于它，**H1f 坐实，bug 在推理入口，与容量无关**
- **P1.2** **训练数据入模检查**：从 dataloader 取 batch，按模型实际看到的表示反解回 `Structure`，算最近原子间距直方图
  → 若进模型时已是 `0.33 Å`，**H1c 坐实，bug 在预处理，模型无辜**
- **P1.1** **一阶统计对照**：200 无条件样本 vs 2,357 训练结构，叠加直方图——最近原子间距、每原子体积、配位数分布、晶格参数、密度
  → 训练集应集中 `1.4–2.5 Å`；样本集中 `0.33 Å` 则为 bug 而非容量
- **P1.4** **周期性单元测试**：构造仅差一个晶格矢量平移的等价结构，过模型/损失，输出必须等价；不等价则 **H1a 坐实**
- **P1.3** **加噪-重建 sanity**：训练结构加噪至 `t≈1` 再反积分回 `t=0`，测重建误差

**止损门**：P1.1/P1.2 指向 bug → **停止一切重训与扩数据计划**，先修 bug。

### P2 — 链路可逆性与可行集非空证明〔约半天，依赖 P1 通过〕

对 maricite NaFePO₄、Na₃Fe₂(PO₄)₃、Na₄Fe₃(PO₄)₂P₂O₇ 反向积分 ODE 求 `z*`：

| 观察 | 结论 |
|---|---|
| `Φ(z*)` 重建不出原结构 | H6 坐实，ODE 链路有损，下游全部无效 |
| `z*` 可行且 `‖z*‖` 在先验典型分位内 | 可行集非空且在分布内 → 纯全局搜索问题，并获得 warm start |
| `z*` 可行但 `‖z*‖` 在尾部 | 先验不覆盖 → 从 `N(0,I)` 出发永远不可达 |

P1 未过时不做：模型若坏，反演必然失败，此步无区分力。

### P3 — 规范自检：可行集是否被定义成空集〔约 1 小时，可与 P1 并行〕

对 560 个已弛豫竞争相中的 Na–Fe–P–O 子集，逐条过 `HybridOptimization.md` 全部硬约束：

- 报告每条约束的**单独通过率**与**全部同时通过率**
- 对 Fe–O `2.50 Å`、P–O `2.00 Å` 截断做 `±0.1 Å` 扫描，量化脆性
- 顺带回答 1/19 是结构问题还是截断过脆

**止损门**：已知好结构的全约束通过率极低 → 规范需重谈，而非继续投算力。

### P4 — 仅当 P0–P3 全过，才谈模型改造

按性价比排序：训练数据全部 UMA 弛豫（同时解决 H3 与 1/19）→ 使用 17,774 而非 2,357 → 扩模型 → 组成条件化。

## 5. 时序

```
Day 1  ├─ P0 (口径)      ─┐
       ├─ P1 (bug 判定)  ─┼─→ 独立，同日并行
       └─ P3 (规范自检)  ─┘
Day 1 晚  汇总 → 决定分支
Day 2  P2 (仅当 P1 过)  或  修 bug (若 P1 未过)
Day 3+ P4 (仅当 P0–P3 全过)
```

## 6. 停止规则

- 不再增加求解器、步数、乘子、种子；11 组已一致收敛，再跑不产生新信息
- P1 出结论前不重训、不扩数据
- P0 出结论前不引用任何 `E_hull` 数字做决策
- 不启动 256 样本主实验（`main_256_authorized: false` 仍然成立）

## 7. 待核实事项

- **9/1–9/3 实验目录未在仓库中找到**。`experiments/` 最新为 `exp_2026-08-27*`，检索未发现 9 月目录，应在 `/mnt/data2` 耐久存储中。因此那 11 组实验具体调用 `get()` 还是 `get_mp2020_reference()` **尚未确认**（P0.1）
- 本文第 1、2 节结论全部来自只读文件核对，**未实际运行代码验证**
- P1.5 与 P0.4 的脚本已写完但**都未运行**：`Bash` 与 `Agent` 工具被安全分类器持续阻塞（`og-5-6-sol-standard is temporarily unavailable`），只读的 `Read` 可用。打包数据集 `.pt` 与 `uma_hull_cache.json` 的实际路径因此**尚未定位**，两个脚本都留了自动 glob 与 `--dataset` / `--cache` 显式参数
- `DifferentiableUMAHull` 的 facet 提取沿用 `differentiable_mp_hull.py` 的写法，依赖 `PhaseDiagram.qhull_data` 的列语义（前 n−1 列为 `elements[1:]` 的原子分数，末列为每原子能量）。**该语义未经运行验证**——`tools/p0_4_uma_hull_facets.py` 的精确性检验就是为此设计的：若 pymatgen 版本的 `qhull_data` 存的是形成能而非绝对能量，误差会以常数或线性偏移的形式立刻暴露

## 8. 主要依据

- `experiments/exp_2026-08-26/reports/pilot_summary.json`：G5 门结果与 B0/B1/B2/M0 逐项统计
- `src/nagen/inverse/uma_hull.py`：缓存读取与两条口径路径
- `src/nagen/inverse/uma_hull_campaign.py:560,574,626`：两张凸包、组成池采样、`definition` 字段
- `src/nagen/inverse/differentiable_mp_hull.py`：facet 数学的既有实现（错标尺）
- `src/nagen/inverse/model.py:204,216-221` 与 `sample.py:57`：训练/推理源分布不一致
- `src/nagen/inverse/quality_dataset.py:28-38` 与 `constraints.py:143-199`：`GEOMETRY_CHECKS` 与 `evaluate_feasibility` 的检查名 9 项中仅 4 项对得上（见 2.8）
- `src/nagen/inverse/constraints.py:18-31` vs `model.py:127-130`：两套镜像约定（见 2.7）
- `docs/8_27_progress.md`：正式口径定义、恢复顺序、G5 前置条件
- `docs/HybridOptimization.md`：统一优化问题表与全部硬约束

## 9. 本轮执行记录（2026-09-03）

| 项 | 动作 | 状态 |
|---|---|---|
| P0.3 | `uma_hull_campaign.py` `definition` 改为全 UMA 口径，拆出 `official_reference_field` / `secondary_cross_scale_definition` | ✅ 已改 |
| P0.3 | `UMAHullCache.get_mp2020_reference()` 增加 `acknowledge_cross_scale` 闸门 | ✅ 已改 |
| P0.4 | 新增 `src/nagen/inverse/differentiable_uma_hull.py` | ✅ 已写 |
| P0.4 | 新增 `tools/p0_4_uma_hull_facets.py` 验收脚本 | ⏸ 待跑 |
| P1.5 | 新增 `tools/p1_5_source_mismatch.py` | ⏸ 待跑 |
| H1g | 镜像约定不一致检验并入 P1.5 脚本 | ⏸ 待跑 |
| H1c | 撤回"训练集按构造干净"的结论（2.8） | ✅ 已撤回 |
| H1b | `lattice.py` 编解码往返自洽（Gram/N^(2/3) ↔ chol×N^(1/3)），该层无问题 | ✅ 已排除 |
| H1h | `geometry_only` 假设（2.9），需 checkpoint 或 grep 判定 | ✅ 已排除，非 bug |
| **P0.1** | **grep 完成：`get_mp2020_reference` 唯一调用点 = `paired_shooting_campaign.py:205`** | ✅ **已完成** |
| **H5** | **确认 pilot 跑在被禁口径上（2.10）** | ✅ **已确认** |
| **H5 修复** | `CrystalDesignProblem` 去掉跨能标项；MP2020 参数传入即抛错；campaign 传参移除 | ⚠️ **行为变更，待你确认** |
| H1f | 逐通道核对确认 X 是唯一错配通道，且在 pilot 路径上成立（2.9） | ✅ 已确认（定量待 P1.5） |

Bash 恢复后的执行顺序：

```
python tools/p0_4_uma_hull_facets.py            # 秒级，先确认尺子
python tools/p1_5_source_mismatch.py            # 秒级，判 H1f
grep -rn "get_mp2020_reference\|UMAHullCache" src tools experiments   # P0.1
```

**注意**：`get_mp2020_reference` 的闸门会让仍走跨能标口径的旧代码硬失败。这是预期效果，但 P0.1 的 grep 尚未做，**因此还不知道会打到哪些调用点**——跑 grep 时若发现正在运行的作业依赖它，先评估再决定是否临时放行。

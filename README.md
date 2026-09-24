<p align="center">
  <img src="asset/logo.png" alt="ShootingCSP logo" width="360">
</p>

<div align="center">
  <h1>ShootingCSP | 高精度可控晶体结构生成</h1>
</div>

[中文](README.md) | [English](README.en.md)

[![CI](https://github.com/aoyang17/ShootingCSP/actions/workflows/ci.yml/badge.svg)](https://github.com/aoyang17/ShootingCSP/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

ShootingCSP 是面向高精度、可控晶体结构生成的 Python 库。它将目标晶体的几何/性质约束表述为优化问题，并将这些约束逐步传播到生成式推理过程中，从而生成同时符合数据流形（目标晶体的先验知识）与高维约束的晶体。其核心算法 ShootingFlow 将 flow-matching 推理与优化表述为由最优控制理论中的打靶法求解的边值问题。

我们以 Na-Fe-P-O 磷酸盐晶体生成任务为例。项目包含三个模块：
- 晶体表示，用于参数化晶体编码；
- 生成架构，用于学习数学流形；
- 代理模型，用于提供物理估计。下面依次介绍这三个部分。

因为ShootingFlow是基于Optimal control theory的可控Flow matching生成，因此每针对一个新的晶体设计问题，我们会将其集成一个优化反问题进行描述。

## 安装

从仓库安装开发版本：

```bash
python -m pip install -e ".[shooting]"
```

基础 Python API：

```python
from shootingcsp.generation import optimize_source_with_surrogate
from shootingcsp.surrogate import SurrogateSpec, load_surrogate
from shootingcsp.training.train_flow import main as train_flow
```

重新训练或微调 Flow：

```bash
python -m shootingcsp.training.train_flow --help
```

数据集适配器和独立 surrogate model 接口见
[TrainingAndSurrogateInterfaces.md](docs/TrainingAndSurrogateInterfaces.md)。

当前版本为 `0.1.0`，采用 Apache License 2.0。引用信息见 `CITATION.cff`。

## 一、晶体表示

本项目采用 $(N, \mathbf{A}, \mathbf{X}, \mathbf{L})$ 晶体表示策略，其中：

- $N \in \mathbb{Z}_{>0}$ 是周期晶胞中的原子总数；
- $\mathbf{A} = (a_i)_{i=1}^{N}$ 指定每个原子位点对应的元素；
- $\mathbf{X} = (x_i)_{i=1}^{N}$，且 $x_i \in [0,1)^3$，指定晶胞内的原子位置，同时也是原始弛豫任务更新的原子位置变量；
- $\mathbf{L} \in \mathbb{R}^{3\times3}$ 指定晶胞尺寸和形状，同时也是原始弛豫任务通过晶胞过滤器更新的变量。

这四个部分共同构成生成优化问题的设计变量。

## 二、生成架构

1. 先固定离散组成：N = 52，A = Na6Fe6P8O32；仅优化 Flow 源变量 z_X 和 z_L。
2. 使用冻结的几何 Flow，执行 12 步 source-space Adam 优化，并在每一步通过完整的 48 步 midpoint ODE 反向传播。
3. 优化软目标：UMA 能量，以及可微的距离、P/Fe 配位、体积和源先验惩罚。
4. 在不计算梯度的情况下重新积分优化后的源变量，得到最终生成结构。
5. 执行灾难性预检查：非有限晶胞、无效密度/条件数以及严重原子重叠。
6. 对保留下来的结构进行固定晶胞 UMA/FIRE 弛豫，然后执行精确 hard gates：组成、电荷、PBC 距离、CN(P)=4、可配置的 CN(Fe) ⊆ {4,5,6}、多面体中心、Fe 共享、氧覆盖和力收敛。
7. 要求有限参考 E_hull^UMA ≤ 0.15 eV/atom；在固定组成下，最小化 UMA 能量等价于最小化当前有效凸包间隙。
8. 只有在全部 hard gates 通过后，才去除 Na 并执行骨架新颖性和多样性检查：不与已知 Fe–P–O 结构/拓扑匹配、具有足够的描述符距离、不存在跨候选重复，并通过 CIF 往返检查。
9. 以流式输出方式增量接受候选结构；新颖性和凸包稳定性是最终门控/选择准则，不是 source-space 生成 loss 中的优化项。


### ShootingFlow · 优化问题表述 | 以 Na–Fe–P–O 生成为例

| 类型 | 项目 | 要求 / 定义 | 当前结果 / 备注 |
|---|---|---|---|
| ![目标](https://img.shields.io/badge/-Objective-2ea44f?style=flat-square) | 稳定性：最小化 $E_{\mathrm{hull}}$ | $\le 150\ \text{meV/atom}$ | ✓ 结果范围：122–168 meV/atom；50% 符合 |
| ![目标](https://img.shields.io/badge/-Objective-2ea44f?style=flat-square) | 新颖性：最大化 $f_{\mathrm{nov}}$ | 去除碱金属后的 Fe–P–O 骨架 | ✓ 与已知结构无匹配 |
| ![设计变量](https://img.shields.io/badge/-Design_Variable-8250df?style=flat-square) | 原子数 | $N\in\mathbb Z_{>0}$ | 周期晶胞中的原子总数。 |
| ![设计变量](https://img.shields.io/badge/-Design_Variable-8250df?style=flat-square) | 元素序列 | $\mathbf A=(a_i)_{i=1}^{N}$ | 指定每个原子位点对应的元素。 |
| ![设计变量](https://img.shields.io/badge/-Design_Variable-8250df?style=flat-square) | 分数坐标 | $\mathbf X=(\mathbf x_i)_{i=1}^{N},\ \mathbf x_i\in[0,1)^3$ | 指定晶胞内的原子位置；同时也是原始弛豫任务更新的原子位置变量。 |
| ![设计变量](https://img.shields.io/badge/-Design_Variable-8250df?style=flat-square) | 晶格矩阵 | $\mathbf L\in\mathbb R^{3\times3}$ | 指定晶胞尺寸和形状；同时也是原始弛豫任务通过晶胞过滤器更新的变量。 |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | 元素组成 | 仅包含 Na、Fe、P、O，且四种元素均出现 | ✓ Na<sub>6</sub>Fe<sub>6</sub>P<sub>8</sub>O<sub>32</sub> |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Fe 平均形式价态 | $\bar z_{\mathrm{Fe}} \ge 2$ | ✓ $\bar z_{\mathrm{Fe}} = 3.0$ |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | 理论比容量 | $C_{\mathrm{th}} \ge 130 mAh g^{-1}$ | ✓ 130.4 mAh $g^{-1}$ |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | 体积 | 10.5–20.5 $Å^3$/atom | ✓ 12.91–16.41 $Å^3$/atom |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | 原胞原子数 | $23 \le N_{\mathrm{prim}} \le 184$ | ✓ 当前晶胞 $N = 52$ |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | PBC 最小距离 | P–O $\ge 1.40$；Fe–O $\ge 1.55$；Na–O/O–O $\ge 2.00$；P–P $\ge 2.60$；Na–Na $\ge 2.20$ Å | ✓ 无 PBC 重叠；P–O 1.474–1.652 Å，Fe–O 1.791–2.428 Å |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | P–O 配位 | $r_{\mathrm{P-O}} \le 2.00$ Å；CN(P) = 4 | ✓ 全部为 PO<sub>4</sub>；8/8 个 P 通过 |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Fe–O 配位 | $r_{\mathrm{Fe-O}} \le 2.50$ Å；CN(Fe) ∈ {4,5,6} | ✓ 每胞 6 个 Fe；均为 FeO<sub>4</sub>/FeO<sub>5</sub>/FeO<sub>6</sub> |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | P/Fe 多面体中心 | P 和 Fe 严格位于各自的 O 多面体内部 | ✓ 14/14 个 P 和 Fe 中心通过 |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Fe 多面体相邻关系 | 任意 Fe–Fe 共用 O 数 $\le 2$ | ✓ 任意两个 Fe–O 配位多面体均不共用同一个氧三角面 |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | 氧覆盖 | 全部 O 均被骨架多面体覆盖 | ✓ 32/32 个 O 被覆盖 |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | UMA $E_{\mathrm{hull}}$ | $E_{\mathrm{hull}}^{\mathrm{UMA}} \le 150$ meV/atom | ✓ 100.00–146.21 meV/atom |
| ![约束](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | 原子力收敛 | $F_{\max} \le 0.03$ eV $Å^{-1}$ | ✓ 0.01874–0.02996 eV $Å^{-1}$ |

## 三、代理模型

本研究采用 [UMA](https://arxiv.org/abs/2506.23971) 作为力场代理模型，提供凸包上方能量 $E_{\mathrm{hull}}$ 和力 $F$ 对原子坐标 $X$ 与晶格参数 $L$ 的偏导数：

$$
\frac{\partial E_{\mathrm{hull}}}{\partial X},\quad
\frac{\partial E_{\mathrm{hull}}}{\partial L},\quad
\frac{\partial F}{\partial X},\quad
\frac{\partial F}{\partial L}.
$$

## 程序入口

- `tools/run_uma_campaign.py`：可恢复的多 worker 生成、弛豫、门控、凸包代理和增量选择。
- `tools/generate_uma_guided_framework_candidates.py`：单批生成和 UMA 引导的源空间优化。
- `tools/select_framework_candidates.py`：离线严格筛选和排序。
- `tools/audit_uma_campaign.py`：对最终 CIF 进行独立验证。
- `tools/audit_crossmetal_novelty.py`：2026 年 9 月 11 日执行的跨金属新颖性审计。

`runs/` 是指向不可变实验资产的符号链接，其中包含
flow checkpoint、conditioning profile、参考标签、campaign 记录和已选结构。

## 运行弛豫 (当前生成优化架构仍然与弛豫独立进行)

```bash
tools/launch_uma_campaign.sh
```

使用 `tools/launch_uma_campaign.sh --audit` 可重新验证 campaign 导出的 CIF。

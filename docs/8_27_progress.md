# NaGen 2026-08-27 实验进度与平台迁移交接

> 记录时间：2026-08-27（UTC）
> 当前正式实验：`exp_2026-08-26_02`
> 项目代码：`/home/aobo/NaGen`
> 耐久产物：`/mnt/data2/aobo/NaGen`
> 本文是阶段快照；恢复实验时应以各实验目录中的 manifest、scheduler state 和逐样本记录为实时事实。

## 1. 当前结论

本轮算法路线已经确定：先训练并冻结无条件 Flow Matching 模型，在推理阶段固定 `N,A`，由 ShootingFlow 优化连续初始噪声 `z_X,z_L`，再从优化后的初始噪声完整重积分生成结构。UMA 单点能量用于生成期可微引导；UMA + FIRE/BFGS 完整结构弛豫只在生成完成后执行。

项目正式 `E_hull` 口径已经改为：

```text
E_hull(C) = E_UMA(C) - E_ref^UMA(comp(C))
```

其中 Materials Project 只提供竞争相结构和组成；候选与全部竞争相必须使用同一 UMA checkpoint、`omat` 任务头和同一弛豫协议计算能量。最终强约束保持：

```text
E_hull <= 0.150 eV/atom
```

旧定义 `E_UMA-E_MP` 是跨能标诊断量，已证明不能承担绝对凸包门控，不得在恢复实验或迁移平台时重新接回正式流程，也不得拟合或扣除常数偏移。

## 2. 已完成工作

### 2.1 数据、模型和基础链路

- Na–Fe–P–O 数据规范、组成筛选、解析约束和数据审计已经完成。
- 主质量数据集为 2,357 个结构；按化学式分组后无 split 泄漏。
- 两阶段 Flow Matching 训练已经完成；历史记录中的 joint best validation loss 为 `3.46420`，geometry best 为 `1.19556`。
- UMA 能量、力、应力调用及坐标/晶格 VJP 有限差分检查已经通过。
- 24 步 ODE 已作为当前积分设置；历史 24/48 步最大相对差为 `0.2604%`。
- 固定 `A`、冻结 Flow 权重、源噪声更新和最终重积分检查均已通过。

### 2.2 软约束和距离恢复实验

32 样本 Fe 配位 trade-off 实验已经完成：生成期严格几何通过 `17/32`；按允许 Fe 先软化的预筛规则有 19 个结构进入 UMA 弛豫，`19/19` 达到最终力收敛，但弛豫后严格几何通过只有 `1/19`。这说明 Fe 配位可作为生成期 trade-off，但最终解析硬约束不能放宽，也不能依赖弛豫自动修复明显原子碰撞。

后续距离恢复配对实验已经通过，冻结生成期配置为：

| 参数 | 冻结值 |
|---|---:|
| restoration steps | 200 |
| learning rate | 0.005 |
| 最小距离安全余量 | 0.10 Å |
| distance weight | 40 |
| P coordination weight | 12 |
| Fe coordination weight | 2 |
| Fe coordination prior | 0.084 / 0.1025 / 0.8135 |
| ODE steps | 24 |

相同 32 个源噪声下，严格几何通过率由 `17/32` 提高到 `22/32`，新增 5 个通过样本且原通过样本没有回退。该改善来自原正式 `flow_geometry.best.pt` 上的推理期 restoration 和距离余量调整，不是重新训练 checkpoint 的收益。后续 pilot 必须继续使用该正式 checkpoint 和上述冻结距离配置。

`exp_2026-08-27/checkpoints/flow_geometry_repair.best.pt` 的无条件复核为严格几何 `0/32`，未被采用；不得因为时间更新而把它误认为后续正式 checkpoint。

### 2.3 `E_hull` 口径修正

旧的 12 结构 UMA–MP 校准得到 `median |E_UMA-E_MP| = 0.683 eV/atom`，明显超过 `0.150 eV/atom`。据此完成了以下修正：

- 正式口径改为同一 UMA 能标的凸包距离；
- MP 仅负责提供竞争相结构；
- 全部 560 个 MP2020 兼容竞争相都纳入 UMA 支持域，而不只使用 MP 能标下的稳定相；
- 正式缓存缺失、支持域不完整、模型哈希或任务头不匹配时 fail closed；
- 推理阶段禁止访问 MP API 或现场重算竞争相。

### 2.4 已实现的新代码

- `src/nagen/inverse/uma_hull.py`：正式同能标 UMA 凸包缓存读取和完整性校验。
- `src/nagen/inverse/uma_hull_campaign.py`：竞争相结构下载、分片 UMA 弛豫和凸包缓存构建。
- `src/nagen/inverse/paired_shooting_campaign.py`：可恢复的 B0/B1/B2/M0 配对 pilot/主实验；主实验强制使用 pilot 冻结的目标尺度。
- `tools/dispatch_uma_hull_relax.py`：只在 GPU 没有计算进程且连续两次确认空闲后启动 UMA 弛豫分片。
- `src/nagen/inverse/pilot_relax.py` 和竞争相弛豫记录已补充最终能量和应力。
- B0/B1/B2/M0 逐样本记录现已保存可直接交给生成后 UMA 弛豫的物理结构。

当前测试结果：`32` 项全部通过。

## 3. 当前正式实验状态

正式实验目录：

```text
/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26_02
```

仓库入口是指向该目录的符号链接：

```text
/home/aobo/NaGen/experiments/exp_2026-08-26_02
```

### 3.1 MP 竞争相下载

已完成 `560/560` 个竞争相结构下载和离线缓存：

```text
mp_cache/mp_competing_structures.json
```

校验结果：缺失 0、组成不一致 0、结构解析失败 0、敏感信息扫描命中 0。MP API key 只在下载进程内存中使用，没有写入代码、缓存、manifest、日志或本文档。当前阶段以后不需要再次访问 MP API。

### 3.2 UMA 弛豫 smoke

真实 UMA smoke 已完成。`mp-1005-GGA`（FeP，8 atoms）结果：

- 初始能量：`-7.4462693742 eV/atom`；
- 最终能量：`-7.4465773568 eV/atom`；
- 最终 `fmax`：`0.0035656 eV/Å`；
- BFGS 达到最终收敛阈值。

正式协议保持与参考脚本一致：FIRE 使用 `ExpCellFilter` 同时优化晶胞和位置，随后 BFGS 只精修原子位置。正式阈值为 FIRE `0.05 eV/Å, 1500 steps`，BFGS `0.01 eV/Å, 1500 steps`。

### 3.3 全部 560 个竞争相 UMA 弛豫

截至本快照，任务尚未真正开始：scheduler 为 `waiting_for_free_gpu`，32 个分片全部 pending，0 running，0 complete。原因是本机 8 张 H20 当时都有其他用户的计算进程；调度器按规则没有抢占或叠加负载。

实时状态文件：

```text
/mnt/data2/aobo/NaGen/experiments/exp_2026-08-26_02/guidance/uma_hull_relax/scheduler/state.json
```

当前终端会话中曾启动空闲 GPU 调度器，但终端会话和本机进程不能作为跨两天或跨平台的可靠状态来源。恢复时必须先读取 `scheduler/state.json`、`manifest_shard_*.json` 和 `records/*.json`，再决定是否重启调度器。所有正式弛豫命令带 `--resume`，不得删除或覆盖已完成记录。

### 3.4 尚未完成、不得提前宣称的结果

- 正式 UMA 凸包缓存尚未构建；
- 新口径下的 32 样本 B0/B1/B2/M0 pilot 尚未运行；
- G5 尚未按新 `E_hull` 口径重新判定；
- 256 样本主实验尚未启动；
- 主实验生成后的 UMA 弛豫、最终 `E_hull`、精确新颖性和去重尚未执行；
- 当前正式最终候选数仍为 0，不代表新路线失败，只表示实验尚未走到最终门控。

## 4. 两天后恢复顺序

恢复工作时严格按以下顺序执行，不要直接跳到 256 样本：

1. 检查代码、`docs/PLAN.md`、`docs/OPTIMIZATION.md` 和本文件，确认正式 `E_hull` 口径没有退回旧定义。
2. 检查 `scheduler/state.json`、32 个 shard manifest、记录数和失败记录；同时检查 GPU 是否被其他任务占用。
3. 若竞争相弛豫未完成，使用相同模型、任务头、阈值和 32 分片设置执行 `--resume`。不得改变半途协议。
4. 只有 560 个条目全部存在且最终 BFGS 收敛后，才运行 `uma_hull_campaign build` 构建正式 `uma_hull_cache.json`。
5. 用 `UMAHullCache` 重新校验：状态 complete、支持数 560/560、UMA 哈希匹配、任务头为 `omat`、每个组成有参考能。
6. 使用已验证的 `exp_2026-08-26/checkpoints/flow_geometry.best.pt` 和冻结距离参数运行 32 样本配对 pilot：B0 无引导、B1 约束、B2 约束+新颖性、M0 完整 MGDA。
7. 汇总配对差值、严格几何通过率、`E_hull`、新颖性、源噪声漂移、梯度范数/夹角和 MGDA 权重。只有 G5 通过后才冻结目标尺度和配置。
8. G5 通过后启动同一 `N,A,z_0` 配对的 256 样本主实验；不得通过挑 seed 只报告最佳样本。
9. 对 M0 结构先做精确预筛，再执行 UMA + FIRE/BFGS 完整弛豫。
10. 弛豫后重新计算全部解析硬约束和正式 `E_hull`，执行训练集匹配、批内去重、新颖性及 Pareto 分析，最后生成 G6–G8 报告。

如果 560 个竞争相中存在未收敛条目，正式凸包缓存必须保持失败关闭。先诊断失败类型，再使用完全相同的协议恢复；若确需修改阈值或优化器协议，必须将全部支持相作为一个新版本整体重算，不能混合两套协议。

## 5. 更换超算平台迁移清单

### 5.1 必须迁移的内容

代码和文档：

```text
/home/aobo/NaGen
```

耐久数据、模型、缓存和实验记录：

```text
/mnt/data2/aobo/NaGen
```

最低限度必须包含：

- `models/uma/uma-m-1p1.pt`；
- `experiments/exp_2026-08-26` 中的数据、组成池、MP phase cache、新颖性索引和既有训练 checkpoint；
- `experiments/exp_2026-08-26/checkpoints/flow_geometry.best.pt`；
- `experiments/exp_2026-08-26_02` 整个正式实验目录，包括 MP 结构缓存和所有已完成弛豫记录；
- 旧实验报告，作为历史结论和参数来源。

不要把当前 Python 环境目录当成可直接跨平台运行的二进制环境。新平台应按 `pyproject.toml` 和 preflight 记录重新创建隔离环境，再运行完整测试和真实 UMA smoke。

### 5.2 新平台必须重新验证

- Python：已验证版本为 `3.13.11`；
- PyTorch：`2.8.0`；
- fairchem-core：`2.21.0`；
- mp-api：`0.46.5`；
- NumPy：当前为 `2.4.6`，应保持 `<2.5`；
- ASE：`3.29.0`；
- pymatgen：`2026.5.4`；
- CUDA、GPU 架构、显存、驱动和 PyTorch CUDA 可见性；
- UMA 标准 calculator 的能量/力/应力和 VJP 有限差分；
- 单个竞争相的 FIRE/BFGS smoke；
- 32 项单元测试；
- 新存储根目录的可写性、容量和配额。

当前机器是 8 张 H20；新平台若不是 H20，不能只因为模型能加载就视为环境等价。需要重新测量单结构显存、单步耗时和可安全并行的 shard 数量。32 是逻辑分片数，不代表必须同时使用 32 张卡。

### 5.3 路径和凭据

代码与 manifest 中有当前平台的绝对路径。迁移后应建立新的耐久存储根目录，并更新新实验配置；旧 manifest 作为不可变历史记录保留，不在原地篡改。项目内符号链接只允许指向新平台的个人可写耐久存储。

MP API key 不属于迁移产物：

- 不复制含 key 的参考脚本；
- 不把 key 写入 shell 脚本、Slurm 文件、环境导出、日志、Markdown 或 JSON；
- 如未来确需重新查询，只通过平台 secret/environment 注入，并在程序内存中读取；
- 当前 560 个结构已经完整缓存，继续本轮实验不需要 key。

原始共享数据当前位于 `/mnt/data2/shared` 且严格只读。新平台未必有相同挂载；若将来需要重建数据集，应由数据管理方提供只读副本或正式传输。当前续跑所需的打包数据和索引已在个人耐久实验目录中，不能通过修改旧共享目录解决路径问题。

### 5.4 迁移完整性校验

迁移前后至少核对以下 SHA-256：

| 产物 | SHA-256 |
|---|---|
| UMA `uma-m-1p1.pt` | `c30034edbf2e127f703f814cacb632661767da99b3e71c5b2ee5290510a52d68` |
| 正式 Flow geometry checkpoint | `28965e09029879a43361e21a20107b32f20fcc26fa1538abfe563473f36dfccd` |
| 560 结构 MP 缓存 | `218067d9af2fe338cd5582ebb88e42f0be0017b75b79f197acb45a216ee1a814` |
| MP phase entries | `f67509ac0a823f0e65ac6d879e25bccb2329963d9b791115e9b86c763fd81b69` |
| 组成池 | `08162062a653e083a0ee5559c6e8eb929d102746f311125bfe22b807f851a76f` |
| 新颖性索引 | `1028e099514c7aaf24b2c6f491e99576bfde937df2927ed2f14f99524135b3b3` |

对正在增长的 `uma_hull_relax` 目录不要只记录单一目录哈希。迁移前先停止提交新 shard，确认没有写入中的 `.tmp` 文件，再保存文件清单、每个 record/manifest 的哈希和总记录数。迁移后逐文件校验后再恢复。

## 6. 恢复时常用命令

从项目目录运行测试：

```bash
env MPLCONFIGDIR=/tmp/nagen-mpl \
  PYTHONPATH=/home/aobo/NaGen/src:/home/aobo/NaGen/.pylibs \
  /mnt/data2/aobo/envs/NaGen/bin/python \
  -m unittest discover -s tests -q
```

正式实验应使用不混入 `.pylibs` 的 Python 3.13 环境：

```bash
export PYTHONPATH=/home/aobo/NaGen/src
export MPLCONFIGDIR=/tmp/nagen-mpl
PY=/mnt/data2/aobo/NaGen/envs/nagen-shooting313/bin/python
```

恢复 UMA 竞争相弛豫时优先使用 `tools/dispatch_uma_hull_relax.py`；它会检查已完成 shard，并只在确认 GPU 无计算进程后启动。不要直接复制旧进程 PID，也不要在新平台沿用旧 GPU index 假设。

构建凸包缓存前，必须确认所有 shard 的 `selected == converged` 且 `failed == 0`。构建命令所需输入为：

```text
phase cache:
  /mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/cache/mp_phase_entries.json
composition pool:
  /mnt/data2/aobo/NaGen/experiments/exp_2026-08-26/compositions/generated_pool_merged.json
records root:
  /mnt/data2/aobo/NaGen/experiments/exp_2026-08-26_02/guidance/uma_hull_relax
output:
  /mnt/data2/aobo/NaGen/experiments/exp_2026-08-26_02/mp_cache/uma_hull_cache.json
```

新平台上应将这些路径映射到新的个人耐久存储根目录，并把映射写入新的实验 manifest。

## 7. 主要依据

- `docs/PLAN.md`：本轮完整执行顺序和 G0–G8 验收门；
- `docs/OPTIMIZATION.md`：优化问题、硬约束和指标唯一口径；
- `configs/exp_2026-08-26_02_hull.json`：正式 UMA 凸包缓存配置；
- `exp_2026-08-26_02/manifest/preflight.json`：环境和模型 preflight；
- `exp_2026-08-27_hull_distance_calibration/reports/LATEST_RESULTS.md`：距离恢复和旧跨能标诊断；
- `exp_2026-08-27_tradeoff32/reports/LATEST_RESULTS.md`：Fe 软约束及弛豫存活率实验。

恢复时若本文与实时 manifest 冲突，以不可变逐样本记录和最新 scheduler state 为事实来源；若本文与 `OPTIMIZATION.md` 的数学定义冲突，以 `OPTIMIZATION.md` 为准，并先修正文档/代码一致性后再继续实验。

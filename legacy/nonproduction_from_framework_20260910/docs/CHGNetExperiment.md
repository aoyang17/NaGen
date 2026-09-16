# CHGNet 四层实验入口

> 最新进展：用户已授权从头训练固定N/A生成器，训练及新候选对照已提交。
> 当前协议和作业链见 [ConditionalGenerationExperiment.md](ConditionalGenerationExperiment.md)。
> 下文“等待生成器授权/权重”是先前阶段的状态，现已转为新模型训练。

> 当前任务范围已更正：保留全部原结构单点评估，停止训练集批量弛豫。
> 2333340 批量作业与 2333352 汇总作业已取消；已完成单点标签及历史弛豫文件保留。
> 下文第三轮批量弛豫/汇总/自动导出安排属于已取消的历史计划，不再继续。
> 后续优化、排序和 diversity 实验转向新生成候选；CHGNet 替代能量 surrogate，
> 但不替代当前尚未定位到 checkpoint 的 NaGen 生成器。

本次单点评估续算作业为 **2333359**（仅分片4–7），使用
`deployment/bscc/chgnet_score_only.slurm`，不含弛豫命令。
分片0–3的2107条已完成标签来自2333340，分片4–7补齐2106条。
`tools/merge_singlepoint.py` 按输入 SHA256、完整 ID 集、模型 hash 和成功计数校验后合并，
完全忽略旧弛豫目录，结果仍是原始几何的单点评估，不改写原训练结构。
旧自动取回进程已关闭，旧 Desktop 批量弛豫导出不再执行。

2333359 的4个单点评估分片已全部 COMPLETED（退出0:0），取回后与旧分片0–3合并。
完整4213条原结构单点评估及验证manifest位于 `outputs/full_singlepoint_verified/`。
合并校验覆盖全部输入ID、输入hash和统一CHGNet权重hash；不包含任何弛豫后的标签。
当前没有继续运行的数据集弛豫任务。新生成候选阶段等待可访问的NaGen生成器checkpoint，
或用户确认从当前训练集重新训练生成器；没有用随机扰动原结构替代真实生成实验。

本实验使用 CHGNet 0.4.2 包附带的 0.3.0 预训练模型（412,525 参数），无需 Hugging Face 登录。NaGen 生成状态仍是 N/A/X/L；CHGNet 只在评价时内部构建周期邻居图。忽略输入 JSON 中全部旧能量标签。

## 实现与边界

| 模块 | 功能 |
|---|---|
| `selection/energy.py` | CHGNet 能量、力、应力；NAXL 一阶 VJP；坐标/晶胞有限差分验证；模型内容 hash |
| `selection/score.py` | 全量重新标注，不因 hard gate 失败而省略预测；旧能量全部忽略，错误逐条记录 |
| `selection/hull.py` | 同一模型重算参考结构，线性规划计算固定参考集的凸包能量及分解比例 |
| `selection/pipeline.py` | conditioning、电中性、周期短距离、数据支持 CN、inside 氧凸包、键长、氧覆盖、最终最大原子力；minimax 排序 |
| `selection/diversity.py` | 平滑周期元素对径向描述符；距训练集及入选集合的距离 |
| `selection/experiment.py` | 独立 train/candidate/reference 输入 → 校准 → gates → 排序 → diversity → JSONL/JSON 输出 |
| `selection/generate.py` | 冻结 NaGen flow、固定原子通道，以 CHGNet 能量梯度优化连续 source；输出配对 baseline/optimized |

正式候选不允许与训练/reference ID 重合。结构族隔离仍须由上游 split 保证：不同 ID 不等于不同结构。可指定 `--condition` 检查统一设计规格；未指定时，以输入每个候选的组成作为固定设计条件，不声称验证外部设计要求。

默认使用自有 `native` motif 实现。它沿用 PolyAnionCathodeFilter 的距离 cutoff、多面体检测及氧覆盖思路，并明确增加氧凸包 inside 检查、数据校准 Fe CN/键长。没有冒用外部工具的运行结果。旧工具的 ATLAS/src 导入、默认关闭 inside 判断以及 TM 4–8 允许范围未直接带入。原工具的 legacy MP hull 修正也未带入。

能量比较仅在相同 family/约化组成内进行；此时参考凸包能量是常数，按能量排序与按未截断 hull 差排序一致。全部连续指标转为 percentile，按最差到次差的字典序排序。`--reference-threshold` 是可选质量预筛，只有提供参考结构才允许设置，未知参考不通过此项。

## E_hull 的诚实口径

此实现返回 `delta_e_reference_eV_atom`（有符号）和 `e_above_reference_hull_eV_atom`（截断为非负）。当前样本参考集不包含完整元素相/二元/三元/四元竞争相，结果必须称为 **有限参考集 hull 估计**，不可宣称真实热力学 E_hull。成分不在参考集凸组合范围内时返回 unknown。参考模型 hash 不一致时直接报错。

输入结构按用户说明已优化，因此默认执行单点评价，不覆盖或重新弛豫原结构。它们在 CHGNet 势能面上未必稳态，参考输出保留 `force_max_eV_A`。未来完整 hull 需补齐竞争相，并统一优化标准。当前原始 CHGNet 能量与 UMA、MP 修正后能量绝不混用。

## 本地运行

```bash
python -m venv .venv
.venv/bin/pip install -e '.[experiment]'

PYTHONPATH=src .venv/bin/python -m nagen.selection.experiment \
  --train /path/to/train_only.jsonl \
  --candidates /path/to/candidates.jsonl \
  --references /path/to/reference_structures.jsonl \
  --condition configs/nagen_chgnet_condition.json \
  --check-gradient --device cpu --out outputs/chgnet_run
```

本轮已安装环境：`/private/tmp/nagen-chgnet-env/bin/python`。原型输入准备脚本 `tools/prepare_chgnet_smoke.py` 只抽取文件前缀，明确写入 SMOKE ONLY 警告。它不产生科学有效的 train/test split。全量 packed 数据可使用 `--train-split train`。

独立重新标注命令（不做候选筛选）：

```bash
PYTHONPATH=src python -m nagen.selection.score \
  --input /path/to/structures.jsonl --device cuda --out outputs/chgnet_labels.jsonl
```

先用同一命令标注独立竞争相结构，再通过 `--reference-scores reference_labels.jsonl` 为目标结构追加有限参考 hull。参考中有失败记录或模型 hash 不一致时拒绝构建 hull。

默认 force threshold=0.05 eV/Å、min_structures=20、每个组成组 k=8、质量池=4k、描述符 RMS 去重距离=0.02。各值都是待在独立 validation 校准的实验参数。径向描述符可能发生不同结构同描述符的碰撞；已测试平移、旋转、原子重排、基矢变换及超胞不变性，但不可替代最终 StructureMatcher 复核。

## DFlow 入口

```bash
PYTHONPATH=src python -m nagen.selection.generate \
  --flow /path/to/nagen_checkpoint.pt \
  --condition configs/nagen_chgnet_condition.json \
  --gradient-structure /path/to/real_structure.jsonl \
  --count 64 --steps 20 --ode-steps 24 --seed 17 --device cuda \
  --out outputs/chgnet_seed17
```

分别对输出 baseline.jsonl 与 optimized.jsonl 用相同校准集、预算和四层设置评价。实际 pretrained NaGen checkpoint 尚未在本轮可访问环境定位，因此此入口已编写，但没有宣称完成训练后生成效果验证。旧无条件模型固定 atom channel 属于实验性条件采样；N/A 在 midpoint 子步也保持固定，family 目前只是 phosphate 策略而非新训练 embedding。源空间只优化 energy，不加入 motif、force、composition 或 novelty penalty。

## 已完成的实测

2026-09-08，本地 CPU，数据来自用户 parent_RDF_opt.jsonl 的前 200 条完整记录，其中 40 条目标体系。26 条用于流程校准/有限 hull 参考，14 条作为候选；相关母结构泄漏，故这些数字只用于接口验证。

- 19 项单元测试通过，包括凸包分解、参考不足、混合模型拒绝、结构描述符不变性及固定原子通道的 midpoint 积分。
- 真实 CHGNet 的 6 个坐标/晶胞方向有限差分检查通过。
- NAXL 梯度桥接的 5 步能量优化从 -7.4164834 降至 -7.4220810 eV/atom。该验证使用 identity geometry map，不是经过训练的 DFlow 生成效果。
- 默认 force=0.05：11/14 因默认 Fe2+/Fe3+ 价态规则不通过；其余 3 条的最大残余力约 0.09866、0.06019、0.10690 eV/Å，最终通过 0。
- 独立敏感性运行 force=0.10：2/14 通过全部 hard gates，但均为训练描述符近重复，最终选出 0。默认阈值没有因此放宽，不回填失败结构。
- 三条已评价候选相对本次有限参考凸包的差值约为 0.000588、0.002861、0.002160 eV/atom；这些小数值不能解释为已证明热力学稳定。

结果：`outputs/chgnet_smoke_cpu_v1/`、`outputs/chgnet_smoke_cpu_force01/`、`outputs/chgnet_bridge_cpu.json`。每次运行保留配置、输入 hash、模型 hash、逐候选拒绝原因、能量/力/应力、参考分解、模型调用次数与墙钟时间。输出目录已存在时拒绝覆盖。

独立标注入口也已执行：`outputs/chgnet_all14_labels.jsonl`，14/14 结构成功重新计算能量、力和应力，不因不满足电中性或 force gate 而跳过；CPU 评价阶段约 1.45 秒。筛选入口则先做便宜的结构/化学 gates，以节省模型调用。

## BSCC

目标目录：`/data02/home/scv7eyx/projects/NaGen`。
独立环境：`/data02/home/scv7eyx/.conda/envs/nagen-chgnet`。
安装脚本：`bash deployment/bscc/setup_chgnet.sh`。固定 PyTorch 2.4.1、CHGNet 0.4.2、pymatgen 2025.10.7、spglib 2.5.0 和 NumPy 1.x，避免旧系统 GCC 被最新依赖源码编译阻塞。venv 只读复用已有 Python 环境的基础依赖，新依赖写入自己的目录。
运行脚本：`deployment/bscc/chgnet_smoke.slurm`，1 GPU、10 CPU、20 分钟上限；内存按 N26 每卡默认约 38 GB 分配，不传 `--mem`（该集群提交策略拒绝显式内存参数）。环境安装与 GPU 作业状态以实际执行报告为准。

已完成 GPU 验证：Slurm 作业 **2333320**，节点 **g0019**，Tesla V100-SXM2-32GB，驱动 535.54.03；状态 COMPLETED，退出 0:0，分配时间 00:01:32。环境 `pip check` 无依赖冲突。桥接检查及 energy descent 通过；GPU/CPU 模型内容 hash 相同，逐项 hard-gate 判定完全一致，已评价候选的最大能量差为 4.7684e-7 eV/atom。

GPU 作业产生 19 次桥接验证调用及 42 次筛选/参考/梯度检查调用。GPU 结果与原始日志已取回本地 `outputs/chgnet_gpu_2333320/`，远端仍保留相同结果目录。实验框架已可执行；没有把 smoke 当成新材料发现或完整凸包稳定性验证。

```bash
cd /data02/home/scv7eyx/projects/NaGen
sbatch deployment/bscc/chgnet_smoke.slurm
```

后续正式实验：先全量目标子集统计/结构族 split，再固定校准参数，进行 17/29/43 三个配对种子的原始生成与能量 DFlow 比较；排序和 diversity 消融使用完全相同的通过 gates 候选池。统一报告通过率、各拒绝原因、质量分位数、去重率和计算开销。

完整数据下载目前受 GitHub LFS 吞吐限制：BSCC 留有 `data/parent_RDF_opt.jsonl.part`，不得把它当作完整数据。此轮只使用已完整解析并单独保存的 smoke 输入，未声称完成约 15,000 条结构的标注或训练。

CHGNet 官方来源：https://github.com/CederGroupHub/chgnet 。本模块使用官方提供的能量/力/应力接口，不需要外部付费 API。

## 第二轮：固定晶胞弛豫诊断（2026-09-08）

新增 `selection/relax_audit.py` 和 `deployment/bscc/chgnet_relax_audit.slurm`。
BSCC 作业 2333337 对原 14 个候选全部进行 CHGNet + ASE FIRE 固定晶胞弛豫，
每结构最多 200 步，目标最大力 0.03 eV/Å；原输入不覆盖，输出独立 before/after 标签和结构。
本实验只是势能面差异诊断，不自动将弛豫加入正式生成流程，也不重新计算或宣称完整 hull。

- 14/14 成功且达到目标；评价及弛豫阶段约 28.89 秒（不等于 Slurm 总分配时间）。
- 最大力 ≤0.05 eV/Å：弛豫前 0/14，弛豫后 14/14。
- 使用原有 profile 和不变的所有 hard gates：通过数从 0/14 变为 3/14。
- 其他 11 个样本在 Na+、P5+、O2- 假设下所需平均 Fe 价态为 +3.25 至 +4.25，
  不符合默认 Fe2+/Fe3+ 模型；不可仅据此宣称一般物理上不可能，也未擅自放宽价态范围。
- 结果位于 `outputs/relax_audit_2333337/`，不与未统一弛豫的参考能量作稳定性比较。

新增全量预处理 `tools/prepare_chgnet_dataset.py`：默认要求完整文件 529741128 字节，
逐行严格解析、保存 SHA256、丢弃旧能量；按母结构 ID 哈希分组拆为 train/val/test。
无法识别的 ID 和部分占位进入 quarantine，不随机混入测试集。
这只是母结构 ID 层面的防泄漏，还必须进行跨 split StructureMatcher 复核。
完整数据仍未下载完，尚未运行全量拆分或标定；现有相关测试共 21 项通过。
可访问项目目录及 GitHub 文件树未定位到训练后的 NaGen checkpoint，真实生成对照仍缺此输入。

## 第三轮：完整数据与全量 CHGNet 实验

完整数据已下载，并在本地和 BSCC 验证 Git LFS SHA256：
`d1d9755714e78715fb3a7f8a6d49b6a945615b114a281a563a7f2528331cd8f5`，
529741128 字节。全量严格 JSONL 解析得到 17774 条，目标 Na–Fe–P–O 子集为 4213 条。
本地完整原始文件为 `outputs/datasets/nagen-parent_RDF_opt-full.jsonl`；
BSCC 为 `data/nagen-parent_RDF_opt-full.jsonl`。旧 `.part` 文件不是完整数据，不用于本轮。

目标数据只有 17 个母结构组；`outputs/full_dataset_v3` 使用预先按组大小平衡的
2950/631/632 train/val/test（6/5/6 个母结构组）。未根据能量或测试表现挑选 split。
`v2` 是初始不均衡哈希拆分，仅保留作审计记录，不用于本轮实验。
不同母结构 ID 仍可能同构，`tools/check_split_leakage.py` 对同约化组成执行
StructureMatcher 复核：val 对 train，test 对 train+val，匹配的 held-out ID 从排序对照中排除。

`outputs/full_audit_v3` 的 profile 仅用 2950 条训练结构校准：

- 全部目标 P 位点 86238 个，均 CN=4。
- Fe 位点 CN=4/5/6 分别为 4040/3332/31978；并非只有六配位。
- 训练中的 Fe4/5/6 配位分别来自 1/3/6 个母结构组，不能把大量相关变体当独立证据。
- 无原子重叠；motif 通过 4208/4213；默认 Fe2+/Fe3+ 电中性通过 1269/4213。
- 所有非力 gates 同时通过 1266 条，作为统一固定晶胞弛豫输入。

BSCC 全量数组作业 **2333340** 已提交（8 分片、最多同时 4 GPU，每分片 2 小时上限）。
每个分片先对所有输入单点计算 CHGNet 能量、力、应力，再对非力 gates 通过者进行
FIRE 固定晶胞弛豫（200 步上限，目标最大力 0.03 eV/Å）。最终 force gate 保持 0.05 eV/Å。
单点标签覆盖全部 4213 条，弛豫不覆盖其他 2947 条；二者分母不得混淆。
输出路径 `outputs/full_chgnet_2333340/shard_*/`，保留逐结构失败而非补齐虚构标签。
任务实际完成状态以 Slurm 和各 shard manifest 为准，提交不代表完成。

`tools/summarize_full_chgnet.py` 要求全部分片和 split matching 完成、模型 hash 一致、
标注/弛豫 ID 完整，才允许汇总。在同一通过 hard gates 的 held-out 候选池内比较
energy vs minimax，每组 k=8、质量池32、径向距离去重阈值0.02，分别报告 diversity 前后。
这里 CHGNet 是能量 surrogate，不依赖旧本地 surrogate；其能量不能直接叫 E_hull。
本轮不将未统一弛豫的参考能量混入稳定性结论，也未宣称已运行 NaGen 新结构生成。

`tools/motif_cutoff_sensitivity.py` 额外对 Fe cutoff 2.3/2.5/2.7 Å 做训练校准、验证集诊断，
不检查 test 来选阈值，不修改主实验冻结的 2.5 Å cutoff。

已将第二轮 3 个通过 hard gates 的结构导出并校验 CIF 回读，保存到
`/Users/yangaobo/Desktop/NaGen_feasible_CIF_2333337/`，附 manifest 与限制说明。
这些是已有数据结构的弛豫结果，不保证 novelty 或真实热力学稳定。

跨 split 匹配已完成：14264 次比较、约489.81秒，排除11个验证集 ID（测试集未发现
与前序 split 的配置阈值匹配）。有效验证集620条，测试集632条。参数与匹配对象见
`outputs/full_split_matching_v3/summary.json` 和 `matches.jsonl`。
Fe cutoff 2.3/2.5/2.7 Å 的原始631条验证集 motif 通过数分别626/629/629；
这是诊断结果，包含上述匹配排除 ID，不能直接当独立验证性能。
当前24项测试通过，其中包括真实小批量标签驱动的汇总入口集成测试，
以及向量化周期邻居与显式周期像穷举的一致性测试。

汇总与 CIF 导出依赖作业 **2333352** 已提交，依赖 `afterok:2333340`，
若前置失败则取消，不把残缺结果标为完成。
本地 `tools/fetch_full_when_done.py` 已启动有界8小时等待：只有汇总作业 COMPLETED 后，
才取回 `outputs/full_chgnet_2333340/`，校验每个 CIF 的 SHA256，并复制到新目录
`/Users/yangaobo/Desktop/NaGen_full_CIF_2333340/`。状态写入
`outputs/full_chgnet_2333340/fetch_status.json`；该自动取回需要本机进程保持运行。
本地取回中断不影响已提交的 BSCC 作业；可以用同一工具重试（已有 Desktop 目录需先核对，工具不覆盖）。

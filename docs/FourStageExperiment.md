# NaGen 四层约束实验（独立实验分支）

更新：本文保留最初 UMA 方案及当时的小样本发现；当前已实现的 CHGNet 替代后端、可执行命令与实测状态见 [CHGNetExperiment.md](CHGNetExperiment.md)。用户随后已提供完整数据及外部 filter 的远端仓库位置。

实现：`src/nagen/selection/`。旧 ShootingFlow 路径保留作为 baseline；本实验不调用旧的加权约束目标。

## 设计

| 层 | 输入和行为 | 不满足时 |
|---|---|---|
| Conditioning | 固定元素计数，因而固定 N、Na 含量和化学计量；family 当前仅支持 phosphate | 拒绝与规格不匹配的输出，无 composition penalty |
| Hard feasibility | 明确价态模型的整数电中性；包含自身周期镜像的短距离检查；P=4；数据支持的 Fe CN；合理键长；中心在氧凸包严格内部；最终 max‖Fᵢ‖；外部 filter | 逐项记录，任一失败或未测量均不进入正式排序 |
| Robust ranking | 同 family、同约化组成内，对 E_UMA/atom 和 P/Fe 最差位点的偏心、键长畸变、角度畸变、配位稀有度分别取 percentile | 降序排列 percentile 向量，按字典序最小化；禁止极好能量补偿极差 motif |
| Diversity selection | 固定质量候选池，先按最低结构距离去重，再以最佳质量为锚点做 farthest-first | 与训练集和已选择集合的最小距离决定后续选择，不进入生成 loss |

原始 UMA 能量不是 hull energy。本阶段名称统一为 `E_UMA_eV_atom`；不同组成之间不直接比较其绝对值。可在同组成训练参考中建立宽松能量上限，仍需称为 energy-screen threshold；不得套用 0.15 eV/atom 的 hull 阈值。真正 ΔE_hull 需要一致 UMA 设置下的竞争相参考。

`optimization.optimize_source` 接受固定条件的生成器闭包及已有 UMA VJP energy callable，只对连续 source variables 优化 E_UMA。保存初始和每步终点，允许调用方保留通过 gates 的早期终点。这里没有 composition/force/motif/novelty 加权项。它是集成接口，尚未用远程 NaGen checkpoint 和 UMA 做端到端验证。无条件 checkpoint 冻结原子通道不是经过验证的条件模型；需实际检查固定 A 的 rollout，必要时训练条件几何 flow。family 标签也不能凭新增字段变成模型已学会的条件。

外部工具尚未找到，因此没有伪造 `PolyAnionCathodeFilter` 实现。`hard_gates` 接受适配器 `external_filter(crystal) -> {passed: bool, details: ...}`；未提供时明确不通过。接入时需要核对工具的 PBC、CN、键长、inside 判定，避免复用工具中写死的 Fe CN。当前自有几何实现用于独立审计。

## 校准与防泄漏

从完整训练 split 统计 family × element × CN；以独立结构数量记录支持度，同时记录位点数、inside 比例、键长分位数和排序后的角度谱。P 固定四配位；Fe 采用达到支持门槛的 CN，不人为指定为六。当前最小支持数 20、P/Fe cutoff 2.0/2.5 Å、键长 0.1%–99.9% 分位加 0.1 Å margin 都是待验证的实验配置，不是已确立物理常数。低支持类别记录为 unsupported，不应直接宣称其物理不可能；在扩大数据或专家核验前不会入选。

角度质量目前是 P 相对理想四面体、Fe 相对该 CN 的训练角度谱。若同 CN 存在多种多面体模式，应改为多个数据支持模板的最近距离，避免平均模板压制少数有效环境。spatial freedom 暂不加入，待明确与 Na 迁移通道的可测关系。配位稀有度是描述性代理，不能替代局部能量或键价评价。

训练集用于校准；validation 用于 cutoff/支持门槛/force threshold 敏感性分析；test 在配置冻结后仅评估一次。重复晶体、超胞、同源扰动必须按结构族分组拆分，禁止随机按行泄漏。packed `.pt` 必须指定 `--split train`；JSONL 可带 split 标签，或由调用方提供已经隔离的训练文件。生成结构不能反过来扩大 hard gate 支持范围。

默认电中性模型为 Na+、P5+、O2-，Fe 支持显式配置的整数价态（默认 2/3）。不支持的氧化态、氧空穴和部分占位需要单独的材料模型；不得为了接受全部数据自动放宽到 Fe4.5+。

## 当前实测（2026-09-08）

本地发现 `/Users/yangaobo/Downloads/NaSICON_materials_dataset_clean.txt` 只有 9 条数据；相邻旧项目有 32 个生成结果。这不是用户所说的约 15,000 条训练集。

| 检查 | 9 条原始结构 | 32 条旧生成结构 |
|---|---:|---:|
| 周期短距离检查通过 | 9 | 0 |
| 自有 motif 检查通过 | 9 | 0 |
| 默认 Fe2+/Fe3+ 电中性通过 | 0 | 0 |
| 完整正式验收通过 | 0 | 0 |

此 smoke 为验证代码显式使用 `--min-structures 1`，且 9 条原始结构是校准集自身的回代审计。不能据此声称 gate 的泛化有效性。原始数据要求的 Fe 平均价态为 3.75–4.5；发现 36 个 FeO6 位点、54 个 PO4 位点，全部中心在局部氧凸包内部。UMA 力/能量与外部 filter 均未测量，因此没有正式入选候选，也没有“新架构已提升性能”的结论。

复现（本机使用 `miniconda3/envs/torchFM/bin/python`）：

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_selection.py -v
PYTHONPATH=src python -m nagen.selection.audit \
  --train /path/to/naxl_packed_v1.pt --split train \
  --out outputs/four_stage/train_profile.json
PYTHONPATH=src python -m nagen.selection.audit \
  --train /path/to/train_only.jsonl --candidates /path/to/candidates.jsonl \
  --out outputs/four_stage/candidate_audit.json
```

本机审计 JSON 存于 `outputs/four_stage/local_training_audit.json` 和 `local_generated_audit.json`，包含来源 SHA256、profile、逐结构/逐位点结果。outputs 按仓库规则不入 Git。

## 下一阶段实验及资源

1. CPU 审计完整训练集、split、价态与新 filter；Fe cutoff 2.3/2.5/2.7 Å 敏感性检查。冻结 profile，并在独立 validation 上统计拒绝原因及少数 family 的误拒率。
2. 先跑 16 个结构的 UMA/flow smoke，检查最终能量与力单位、原子数归一化、checkpoint hash，并运行已有 UMA VJP 有限差分检查。预计先申请单张 24–48 GB GPU；实际显存与吞吐以 smoke 为准。CPU 审计无需 GPU。
3. 配对种子 17/29/43，各 64 个固定条件起点；比较原始生成、energy-only DFlow、旧 weighted baseline。预算同时报告 flow evaluations、UMA calls、GPU seconds；优化组使用相同初始源和条件。
4. 对同一个通过 gates 的池比较 energy-only ranking、minimax ranking；再比较 quality-only top-k 与固定质量池内 diversity selection。k=8/16/32，质量池初始取至多 4k。不足 k 就报告短缺，不回填失败结构。
5. 距离工具须先验证原子置换、整体旋转/平移、周期换胞及超胞等价不变性。提供候选间距离矩阵、到完整 TRAIN 集的最近距离及 descriptor 版本；当前 selection 接口接收这些已验证距离，尚未假设现有 descriptor 能满足全部要求。
6. 报告每层通过率、最终 force residual、同组成能量分位数、最差局部质量、unique/family coverage、与 train/selected 的最近距离、配对种子差异。零通过也作为诊断结果，优先定位生成质量、价态口径或阈值问题。

启动全量实验所缺：完整训练集和 split 的位置、PolyAnionCathodeFilter 源码/调用说明、可访问的 NaGen 与 UMA checkpoints，以及 GPU 的 SSH/Slurm 信息。上述资料到位前，端到端性能结论仍未验证。

参考：BoltzGen 的 [filter.py](https://github.com/HannesStark/boltzgen/blob/main/src/boltzgen/task/filter/filter.py) 将 hard thresholds、最差 rank 和最终 diversity 分开。本实现采用无指标权重的 lexicographic minimax，并用固定质量池约束 diversity。

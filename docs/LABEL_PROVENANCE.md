# 标签来源表（Label Provenance）— parent_RDF_opt.jsonl

日期：2026-08-18（Phase 0）
依据：全量字段统计（`data/audit/label_provenance.json`）+ 上游管线源码
（`/mnt/data2/LiuSW/CodingProjects/ATLAS/work_space/NaCathode/struct_opt_by_mlff_gpu-dynamic.py` 与
`run_mlff_struct_opt_gpu-dynamic.sh`，只读引用）

## 1. 逐字段来源与口径

| 字段 | 来源管线 | 口径 / 含义 | 可靠性 |
|---|---|---|
| `material_id` | 上游构造 | 稳定标识 | 高 |
| `formula` | 上游构造 | 化学式（reduced formula，与结构组成 100% 一致） | 高 |
| `structure` | MLFF 松弛后写回 | 成功记录：两阶段松弛后的几何（旧版代码写回 structure）；错误记录：本次松弛输入 | 高（成功）/中（错误） |
| `optimized_energy_per_atom` | UMA MLFF | `atoms.get_potential_energy() / n_atoms`，两阶段松弛（FIRE ExpCellFilter fmax=0.10 eV/A、最多 1500 步；BFGS fmax=0.01、最多 1500 步）后 | 近似：MLFF 非 DFT；错误记录为中断值 |
| `energy_above_hull` | — | 全部 null，无来源 | 缺失 |
| `energy_per_atom` / `formation_energy_per_atom` | — | 全部 null，无来源 | 缺失 |
| `num_NaO3..12` / `num_TMO3..12` / `num_PO3..12` | — | 键存在但 100% null；应为配位数统计，未写入 | 缺失（需自算） |
| `optimized_structure` | 错误分支快照 | 仅错误记录存在；崩溃时刻的几何（轨迹最后写入帧） | 收敛性不可验证 |
| `error_msg` / `traceback` | 异常捕获 | `[Errno 5] Input/output error`（ASE trajectory/ulm.py 写盘失败，6,304 条） | 事实记录 |

## 2. MLFF 细节（从管线脚本还原）

- 模型：UMA `uma-m-1p1.pt`，经 `fairchem-core` 的 `FAIRChemCalculator`（`task_name="omat"`）加载
- 松弛：ASE `FIRE`（`ExpCellFilter` 晶胞+原子联合，`fmax_coarse=0.10`，最多 1500 步）→ `BFGS`（仅原子，`fmax_fine=0.01`，最多 1500 步）
- 能量：松弛终态 `atoms.get_potential_energy() / len(atoms)`；单位未在文件中注明，数值范围 −7.84 ~ −6.06 eV/atom 与氧化物 MLFF 能量尺度一致
- 无 DFT 泛函/赝势/U 值/磁性/参考态元数据 → 不是 DFT 精度，也不能直接当形成能或凸包能量

## 3. 错误记录语义（35.5%）

- 6,304 条（35.5%）携带 `error_msg`、`traceback`、`optimized_structure`
- 崩溃点：ASE Trajectory 写盘（`ulm.py` 中 `fd.seek`），发生在 FIRE 松弛进行中 → 结构未完成两阶段收敛
- `optimized_structure` 相对 `structure`：平均 RMSD 0.39 A、体积 −1.2%（抽样 400 条）→ 是"部分松弛"状态
- 能量值分布与成功记录几乎相同（mean −6.942 vs −6.946），但**不代表已收敛**
- 与原始 parent 文件（`/mnt/data2/LiuSW/projects/NaCathode/PolyCSP/parent_RDF_raw.jsonl`，23,978 条）比对：成功记录 `structure` 不等于 parent（确认是松弛后结构）；错误记录的 `structure` 亦不等于 parent（推测输入为前一轮已松弛结构，Re-opt 流程）

## 4. 使用建议

1. **禁止**将 `optimized_energy_per_atom` 用于：凸包稳定性、形成能、任何 DFT 级结论
2. 可暂用于：相对排序/代理模型初筛（需与 DFT 标定）、结构松弛前后对比、训练能量代理的 baseline（需显式标注 MLFF 口径）
3. 35.5% 错误记录：训练/评测前应打 `has_error` 标记；生成任务建议先剔除，或用 `optimized_structure` 做"可松弛性"二分类标签（错误=失败样本）
4. 配位字段全部缺失 → 统一使用本审计的 `q_label` / `po4_frac` / `tm_coord_hist`（`data/audit/records.csv`）
5. `energy_above_hull` 是首要标签缺口，需外部 DFT/热力学数据补齐（见 `DFT_CONVENTIONS.md`）

## 5. 待确认项（需上游确认）

- 成功记录 `structure` 是否保证两阶段收敛（fmax<=0.01）——文件无残差力证据
- 错误记录的确切输入（哪个 Re-opt 版本）
- 能量单位与参考态
- `parent_RDF_raw.jsonl`（23,978 条）与本文件（17,774 条）的筛选关系

# Phase 0 完成报告 — 数据定义与审计

日期：2026-08-18
范围：全量数据审计、canonical 数据集 v1、结构往返验证、去重、group split v1、
标签来源表、硬约束规范、DFT 口径约定

## 1. 执行内容

| 项目 | 产出 | 状态 |
|---|---|---|
| 全量流式审计（缺失率/元素/原子数/晶格/空间群/配位/能量/无序/氧化态） | `data/audit/audit_summary.json`、`records.csv`（17,774×63） | ✅ |
| primitive/Niggli 约简统计 | 同上 + `provenance.json` | ✅ |
| canonical dataset v1（raw_id ↔ canonical_id + 原结构 + primitive 结构） | `data/audit/canonical_v1.jsonl`（724 MB） | ✅ |
| JSON → Structure → CIF → Structure 无损往返 | `data/audit/roundtrip_report.json`（150/150 + 150/150） | ✅ |
| 重复结构清单 | `data/audit/duplicates.json`（0 精确重复） | ✅ |
| group split v1（80/10/10，0 化学式泄漏） | `data/splits/split_v1.json`、`split_by_idx.csv` | ✅ |
| 标签来源表（字段口径 + 上游管线还原） | `data/audit/label_provenance.json`、`docs/LABEL_PROVENANCE.md` | ✅ |
| 硬约束规范 v1 | `docs/HARD_CONSTRAINTS_SPEC.md` | ✅ |
| DFT 口径约定 | `docs/DFT_CONVENTIONS.md` | ✅（口径未冻结，禁止先行 DFT） |
| 数据卡 v1 | `docs/DATA_CARD_v1.md` | ✅ |

## 2. 核心发现

1. **数据是"大晶胞 + 低对称"**：primitive 约简后中位 91 原子，99.9% >20 原子，91.5% 为 P1。
   symprec 1e-3~1e-1 均不减小 → 真实的 Na 空位有序超胞。**直接按 MatterGen MP-20 流程微调不可行**，
   需改生成策略（母体小晶胞 + 空位枚举，或大晶胞预算）。
2. **能量标签 = MLFF 且 35.5% 中断**：`optimized_energy_per_atom` 是 UMA（fairchem omat）MLFF 能量，
   6,304 条（35.5%）在松弛中因轨迹写盘 I/O 错误中断；收敛性均不可从文件验证。
   `energy_above_hull` 100% 缺失 → **hull 是首要标签缺口**。
3. **配位字段全空**：30 个 `num_*` 字段 100% null；审计已自算等价描述符
   （Q 标签、PO4 完整性、TM 配位直方图）。
4. **化学体系 ≠ 纯 PO4**：P2O7 焦磷酸占 45.7%，含 F（14.0%）、N（5.5%）、C（0.5%）体系；
   生成目标应覆盖多骨架类型。
5. **无重复、无解析失败、无无序、无 formula 不一致**：17,774 条全部唯一。
6. **199 条"异常近邻"经复核为短 V=O/Ti=O 键（化学合理）**，不是几何错误 → 硬过滤用键型下限而非共价半径规则。

## 3. 验收门（Gate）验证

### Gate 1：缺失值和泄漏可解释

- 缺失解释：`energy_above_hull`/`energy_per_atom`/`formation_energy_per_atom` 为管线未提供；
  `num_*` 字段为管线占位未写入；35.5% 记录带 I/O 错误标记。全部有据可查（见标签来源表）。
- 泄漏控制：split v1 按 `reduced_formula` 整组分配，**化学式泄漏 = 0**；
  随机逐行切分未使用。✅

### Gate 2：抽样结构 JSON → Structure → CIF 无损往返

- 原始 JSONL 抽样 150/150 通过（n_atom 覆盖 23–184）；
- canonical primitive 结构 150/150 通过（n_atom 覆盖 30–183）；
- 判定标准：Niggli 规范晶格等价 + 原子数/元素多重集一致 + 分数坐标最小镜像匹配（≤1e-3）。✅

**Phase 0 验收门全部通过。**

## 4. 存储与环境说明

- Phase 0 产物已迁移到 `/mnt/data2/aobo/NaGen/project_data`（约 802 MB），项目内的
  `data` 是兼容现有命令的软链接。
- Python 补充依赖已迁移到 `/mnt/data2/aobo/NaGen/dependencies/python313`，项目内的
  `.pylibs` 是兼容现有运行方式的软链接（pymatgen 2026.5.4 / spglib 2.7.0 /
  numpy 2.5.2 / ase 3.29.0）。
- 输入数据保持严格只读：审计仅读取 `/mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl`；
  上游脚本位于 `/mnt/data2/LiuSW/...`，仅读取用于标签来源推断。

## 5. 交接到 Phase 1 的待办

1. **冻结化学体系**：F 去留（建议保留为条件）、C/N 剔除（建议）、脱钠相（Na=0）处理
2. **MatterGen 适配策略**：因 99.9% 超 20 原子，需决策小晶胞母体+空位枚举 vs 大原子数微调
3. **环境固化**：将现有 target 依赖改为正式 Conda env，并加入 MatterGen
4. **hull 标签**：DFT 口径冻结（PBE+U/PAW/参考态）→ 小样标定 MLFF↔DFT → 564 公式凸包表
5. **配位描述符落地**：将审计的自算配位逻辑固化为 `src/nagen/features/` 模块，供生成/评测复用
6. **canonical schema 定稿**：`raw_id → canonical_id → split → features → calculation_id` 谱系列表
   （Phase 0 已产出 raw_id↔canonical_id↔split 与逐条特征，calculation 谱系待 Dflow 阶段）

## 6. 关键文件索引

- 文档：`docs/DATA_CARD_v1.md`、`docs/LABEL_PROVENANCE.md`、`docs/HARD_CONSTRAINTS_SPEC.md`、
  `docs/DFT_CONVENTIONS.md`
- 代码：`src/nagen/data/audit.py`、`roundtrip.py`、`duplicates.py`、`split.py`、`label_provenance.py`
- 数据：`data/audit/*`、`data/splits/*`

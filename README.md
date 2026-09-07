# NaGen

面向新型磷酸盐钠离子正极的高精度、可控晶体结构生成项目。

## 当前结论

当前默认实验主线已收敛为“冻结 Flow + 新训练的周期 E_hull surrogate + ShootingFlow
source-space 优化”。主线资产、优化变量和旧实验边界见
[docs/CODE_LAYOUT.md](docs/CODE_LAYOUT.md) 及
[configs/surrogate_shootingflow_v1.json](configs/surrogate_shootingflow_v1.json)。

NaGen 不应被定义成单一“生成模型”，而应是一个可追溯的闭环材料发现系统：

`只读原始数据 → 数据审计/标准化 → 表示与标签 → 生成 → 快速物理筛选 → MLFF 松弛 → DFT/Dflow 验证 → 主动学习回流`

首选技术路线是 **MatterGen 迁移学习基线 + 磷酸盐领域适配 + Dflow 多保真优化闭环**。现有数据量（17,774 条）适合领域微调和条件适配器，不适合直接替代大规模通用预训练、从零训练一个通用 MatterGen。自研模型应作为并行研究线，在统一数据与评测协议成熟后再推进。

完整历史分析见 [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md)，多目标问题定义见
[docs/OPTIMIZATION.md](docs/OPTIMIZATION.md)。

## Phase 0 状态（2026-08-18 完成）

全量数据审计、canonical dataset v1、group split v1、标签来源表、硬约束规范 v1 与
DFT 口径约定已完成，验收门通过（缺失值/泄漏可解释；抽样 150/150 + 150/150 无损往返）。

核心结论：

- 17,774 条无解析失败、无精确重复；564 个 reduced formula
- primitive 约简后中位 91 原子、99.9% >20 原子、91.5% P1 → 大晶胞 Na 空位有序超胞，
  MatterGen 需改适配策略
- `optimized_energy_per_atom` 为 UMA MLFF 能量（非 DFT），其中 6,304 条（35.5%）松弛中断；
  `energy_above_hull` 100% 缺失 → hull 是首要标签缺口
- 配位字段（num_*）100% null，已自算替代描述符；化学体系以 P2O7 焦磷酸为主（45.7%），非纯 PO4
- split v1：train/val/test = 80/10/10，化学式泄漏 = 0

详情见 [docs/PHASE0_REPORT.md](docs/PHASE0_REPORT.md)、[docs/DATA_CARD_v1.md](docs/DATA_CARD_v1.md)、
[docs/LABEL_PROVENANCE.md](docs/LABEL_PROVENANCE.md)。

## 数据边界

- 原始数据：`/mnt/data2/shared/NaCathode/parent_RDF_opt.jsonl`
- 原始目录严格只读，不做原位转换或缓存。
- 所有转换数据、缓存、日志、模型和计算结果均写入 `/mnt/data2/aobo/NaGen`，
  项目内只保留便于使用的软链接。
- `runs/`、模型权重、轨迹、日志、本地依赖和软链接目标不进入 Git；迁移这些耐久产物时应单独传输并校验哈希。

## 实验记录

生成实验统一记录在 `experiments/exp_YYYY-MM-DD/`；同一天的后续实验依次增加
`_02`、`_03`。目录规范和首版四候选结果见 [experiments/README.md](experiments/README.md)。

## 当前目录

```text
NaGen/
├── docs/                    # 设计、约束、数据卡与阶段报告
├── experiments/             # exp_YYYY-MM-DD 结果软链接与命名规范
├── src/nagen/
│   ├── analysis/            # 结构聚类等分析工具
│   ├── data/                # 审计、校验、去重、划分
│   ├── decomposition/       # 家族聚类、母体与 Na 亚晶格分解
│   ├── generation/          # flow、代理模型与候选生成
│   └── inverse/             # 约束逆向生成、松弛与筛选
├── tests/                   # 单元测试
├── tools/                   # 面向产物的辅助工具
├── data -> /mnt/data2/aobo/NaGen/project_data
├── .pylibs -> /mnt/data2/aobo/NaGen/dependencies/python313
└── pyproject.toml           # 包元数据、依赖与命令入口
```

## 安装与验证

基础环境可从仓库元数据安装：

```bash
python -m pip install -e .
```

需要 UMA 松弛或 ShootingFlow 时，安装对应可选依赖：

```bash
python -m pip install -e '.[relax,shooting]'
```

当前工作站的完整测试命令为：

```bash
env MPLCONFIGDIR=/tmp/nagen-mpl \
  PYTHONPATH=src:.pylibs \
  /mnt/data2/aobo/envs/NaGen/bin/python \
  -m unittest discover -s tests -v
```

代码仓库只保存可复现的源码、测试、配置和文档。平台迁移及实验恢复细节见
[docs/8_27_progress.md](docs/8_27_progress.md)。

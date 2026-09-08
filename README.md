# NaGen

NaGen 是面向 Na–Fe–P–O 磷酸盐正极的统一 Dflow / ShootingFlow 晶体生成与约束优化项目。

最新四层约束实验：

- [固定 N/A 生成器与新候选实验](docs/ConditionalGenerationExperiment.md)：从原始几何训练条件 flow，CHGNet 能量引导与同查询预算上限随机对照；不再弛豫训练集。
- [CHGNet 可运行实验](docs/CHGNetExperiment.md)：无需 HF 授权的能量/力后端、NAXL 梯度桥接、有限参考集 hull、四层筛选及 BSCC 作业脚本。
- [docs/FourStageExperiment.md](docs/FourStageExperiment.md)：Conditioning → Hard Gates → Robust Ranking → Diversity Selection；实现位于 `src/nagen/selection/`，包含本地小样本审计及待完成的全量实验。

已有 weighted baseline 算法规范：

- [docs/Algorithm.md](docs/Algorithm.md)：推理流程与模块路径；
- [docs/HybridOptimization.md](docs/HybridOptimization.md)：统一目标和全部约束。

## 推理主线

冻结无条件 Flow Matching，从 Na–Fe–P–O 噪声生成第一波样本；固定离散 (N,A)，用 E_hull surrogate、几何约束、UMA 力/应力约束和新颖性目标，对初始噪声 (z_X,z_L) 做 source-space 梯度优化；再从优化后的噪声完整积分 Flow，得到最终结构并执行精确验收。

## 已训练模型

- Flow Matching：`/mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt`
- E_hull surrogate：`/mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt`

## 运行

```bash
PYTHONPATH=src:.pylibs \
python tools/run_surrogate_shootingflow.py \
  --flow /mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt \
  --surrogate /mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt \
  --out outputs/shootingflow.json
```

UMA 力/应力 barrier 通过 `--uma-checkpoint` 启用。soft 约束仅用于梯度搜索；最终结构必须进行精确 PBC 距离、整数配位、力和应力验收，不执行独立生成后弛豫。

# NaGen

NaGen 是面向 Na–Fe–P–O 磷酸盐正极的统一 Dflow / ShootingFlow 晶体生成与约束优化项目。

唯一算法规范：

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
  --count 128 --batch-size 8 --optimizer hybrid-alm \
  --out outputs/shootingflow.json
```

`--batch-size` 控制单卡上同时完成 Flow ODE、surrogate 图推理与 source-space
反向优化的样本数；显存不足时减小该值。组成只从冻结 Flow 生成，不回退到训练集组成。
`hybrid-alm` 为每类约束维护独立的增广拉格朗日乘子，先用 Adam 搜索，
最后用短程 L-BFGS 精修；`--adam-steps` 与 `--lbfgs-steps` 控制两阶段长度。
`--restoration-steps` 可在前若干 Adam 步暂缓性质和 UMA 力/应力项，先将
距离、体积、配位及多面体约束拉回较合理区域，再恢复统一目标。

UMA 力/应力 barrier 通过 `--uma-checkpoint` 启用。soft 约束仅用于梯度搜索；最终结构必须进行精确 PBC 距离、整数配位、力和应力验收，不执行独立生成后弛豫。

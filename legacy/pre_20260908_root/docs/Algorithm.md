# NaGen DFlow 推理优化算法

## 已训练模块

- 无条件 Na–Fe–P–O Flow Matching：
  /mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt
- E_hull surrogate 初始模型：
  /mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt
- 优化问题规范：SimpleOptimization.md（见同目录）

## 推理流程

1. 训练多个独立 E_hull surrogate，用集成分歧建立并校准 OOD uncertainty。
2. 用冻结的无条件 Flow 从噪声生成初始结构和源变量 $$z_0$$。
3. 对每个样本固定 $$N,A$$，仅优化 $$z_X,z_L$$。
4. 将 $$z$$ 送入完整 Flow ODE，计算稳定概率、新颖性、E_hull 驻点条件和 OOD uncertainty。
5. 按 SimpleOptimization.md 的统一损失反向传播，仅更新 $$z_X,z_L$$，不更新 Flow 或 surrogate。
6. 重新完整积分 Flow，按稳定概率、不确定性和驻点容差验收并去重。

## 需要注意的实现修正

- 这是 source-space shooting：梯度必须穿过完整终点 ODE；不能直接优化终点坐标代替优化初始噪声。
- $$N,A$$ 是离散变量，普通梯度不能直接优化；当前正确做法是每个种子固定 $$N,A$$，只优化 $$z_X,z_L$$。若要优化 $$z_A$$，必须先引入可微的连续类别松弛并重新定义解码器。
- 驻点损失包含 surrogate 的一阶导数；继续反向传播到 $$z$$ 时需要二阶自动微分。
- 集成标准差表示 epistemic/OOD uncertainty；阈值必须由独立验证集上的预测误差校准。
- 统一目标经过非线性 Flow 与 surrogate 映射到 $$z$$ 后不保证全局凸，因此需要多个初始噪声和稳定性统计。
- 若图 surrogate 的邻接拓扑在优化中固定，几何大幅变化时可能产生梯度失真；应使用动态邻接或在迭代中周期性重建图。

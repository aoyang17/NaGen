# NaGen 统一 Dflow / ShootingFlow 算法

## 已训练模块

- 无条件 Na–Fe–P–O Flow Matching：
  /mnt/data2/aobo/NaGen/inverse_v1/checkpoints/naxl_unconditional_surrogate_ready_v2.best.pt
- E_hull surrogate：
  /mnt/data2/aobo/NaGen/surrogate_ehull/runs/final_mace_v1/best_model.pt
- 优化问题规范：HybridOptimization.md（见同目录）

## 推理流程

1. 用冻结的无条件 Flow 从噪声生成第一波 Na–Fe–P–O 晶体样本，得到每个样本的初始噪声 \(z\)。
2. 对每个样本固定离散变量 \(N,A\)，将 \(z\) 送入完整 Flow ODE，得到终点结构 \(\mathcal C(z)\)。
3. 用冻结的 E_hull surrogate 预测 \(E_{\mathrm{hull}}(\mathcal C(z))\)，并计算 HybridOptimization.md 中的可微约束代理、力/应力项、新颖性项和流形正则。
4. 对统一标量目标
   \[
   \mathcal L_{\mathrm{total}}(z)
   =\mathcal L_{\mathrm{prop}}
   +\lambda_{\mathrm{constraint}}\mathcal L_{\mathrm{constraint}}
   +\beta\lVert z-\bar z\rVert^2
   \]
   反向传播到初始噪声，仅更新 \(z\)，不更新 Flow 或 surrogate。
5. 得到 \(z^\ast\) 后重新完整积分冻结的 Flow，生成最终结构 \(\mathcal C(z^\ast)\)，再执行精确约束、去重和结果记录。

## 需要注意的实现修正

- 这是 source-space shooting：梯度必须穿过完整终点 ODE；不能直接优化终点坐标代替优化初始噪声。
- \(N,A\) 是离散变量，普通梯度不能直接优化；当前正确做法是每个种子固定 \(N,A\)，只优化 \(z_X,z_L\)。若要优化 \(z_A\)，必须先引入可微的连续类别松弛并重新定义解码器。
- surrogate 只提供 \(E_{\mathrm{hull}}\) 及其梯度；力、晶格应力等约束仍需要可微 UMA evaluator，不能由 E_hull surrogate 自动推出。
- soft-min、软配位和 barrier 只用于梯度搜索；最终必须在重新生成的结构上用精确 PBC 距离、整数配位、力和应力验收。
- 加权二次目标在 \(E_{\mathrm{hull}}\) 和新颖性两个标量输出上是凸的，但经过非线性 Flow 与 surrogate 映射到 \(z\) 后不保证全局凸；因此需要多种初始噪声和稳定性统计。
- 若图 surrogate 的邻接拓扑在优化中固定，几何大幅变化时可能产生梯度失真；应使用动态邻接或在迭代中周期性重建图。

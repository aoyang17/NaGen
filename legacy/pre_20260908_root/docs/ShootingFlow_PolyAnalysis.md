# ShootingFlow 多面体约束优化

## 目标

从 ShootingFlow 生成的 Na--Fe--P--O 初始结构出发，固定原子数和元素序列，优化 Flow 源变量

$$
\mathbf z=(\mathbf z_X,\mathbf z_L),\qquad
\mathcal C(\mathbf z)=\Phi_{\theta_0}(N,\mathbf A,\mathbf z).
$$

冻结 Flow 和基于原始训练数据训练的 $$E_{\mathrm{hull}}$$ surrogate。优化目标是在不偏离初始源变量过远的前提下，获得低 surrogate 能量、低 OOD 不确定性且化学可行的结构。

## 多面体化学原则

对当前结构用 `analysis_poly` 在 PBC 下建立 P--O 与 Fe--O 多面体网络 $$G$$。可接受网络满足

$$
\operatorname{CN}(P)=4,\qquad
\operatorname{CN}(Fe)\in\{4,5,6\},
$$

$$
P,Fe\in\operatorname{int}\!\left(\operatorname{conv}(O_{\mathrm{poly}})\right),
$$

$$
\left|O_{\mathrm{poly},i}\cap O_{\mathrm{poly},k}\right|\le2,
\qquad \forall\,Fe_i\ne Fe_k.
$$

最后一式表示 Fe--O 多面体允许角共享或边共享，但禁止面共享。它不需要额外的“面法向不共面”约束：三个共享 O 已足以定义面共享风险。

## L-BFGS 可行性表述

`analysis_poly` 含有截断邻居、凸包和图拓扑等离散操作，不能直接作为 L-BFGS 的可微约束。采用外层建图、内层连续优化：在外层迭代 $$t$$ 中固定有效网络 $$G_t$$ 及每个中心 $$i$$ 的 O 顶点集合 $$V_i$$。

对每个 P/Fe 中心引入凸组合权重，并用 softmax 参数化：

$$
\lambda_{ij}=\operatorname{softmax}(u_i)_j,\qquad j\in V_i.
$$

于是 $$\lambda_{ij}>0$$ 且 $$\sum_{j\in V_i}\lambda_{ij}=1$$ 自动成立。中心位于 O 多面体严格内部的连续残差为

$$
R_{\mathrm{inside}}
=\sum_{i\in P\cup Fe}
\left\|
\sum_{j\in V_i}\lambda_{ij}\mathbf r_{ij}^{\mathrm{PBC}}
\right\|_2^2.
$$

内层以 L-BFGS 最小化

$$
\begin{aligned}
\mathcal L=
&\;w_E\left(\frac{\widehat E_{\mathrm{hull}}(\mathcal C)}{s_E}\right)^2
+w_u[u(\mathcal C)-u_{\max}]_+^2
+\beta\|\mathbf z-\mathbf z_0\|_2^2\\
&+\rho R_{\mathrm{inside}}
+\lambda_d\mathcal L_{\mathrm{min\text{-}dist}}
+\lambda_b\mathcal L_{\mathrm{bond}}.
\end{aligned}
$$

$$\mathcal L_{\mathrm{min\text{-}dist}}$$ 只排除原子重叠；$$\mathcal L_{\mathrm{bond}}$$ 只保持已选 P--O、Fe--O 键在合理区间。配位数与 Fe 多面体不面共享是 $$G_t$$ 的离散属性，不在内层重复写成大量邻居约束。

## 外层流程与验收

1. 由 ShootingFlow 采样 $$\mathbf z_0$$ 并生成 $$\mathcal C_0$$。
2. 用 `analysis_poly` 建立 $$G_t$$；不满足配位、中心凸包或面共享规则的样本直接淘汰。
3. 固定 $$G_t$$，以 L-BFGS 优化 $$\mathbf z$$ 和 $$u$$；限制步长或使用信赖域，避免键跨越配位截断。
4. 每隔若干步或内层收敛后重建 $$G_{t+1}$$。若拓扑改变，重新开始该外层阶段；若仍不合法，淘汰样本。
5. 最终用 `analysis_poly` 作精确 PBC 验收，再报告 surrogate 能量、OOD 不确定性和结构新颖性。

该方案中，surrogate 提供可微的稳定性引导，L-BFGS 处理固定拓扑下的连续几何可行性，`analysis_poly` 保证离散化学拓扑正确。

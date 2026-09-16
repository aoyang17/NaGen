# DFlow 推理优化

## 1. 问题定义

固定原子数和元素序列 $$(N,\mathbf A)$$，仅优化 DFlow 源变量

$$
\mathbf z=(\mathbf z_X,\mathbf z_L),
\qquad
\mathcal C(\mathbf z)
=(N,\mathbf A,\mathbf X(\mathbf z),\mathbf L(\mathbf z))
=\Phi_{\phi_0}(N,\mathbf A,\mathbf z),
$$

其中冻结 DFlow 参数 $$\phi_0$$。晶格表示继承训练数据中的右手、非退化晶胞。

目标是同时获得高稳定概率和高结构新颖性。

## 2. Surrogate 与 OOD 不确定性

训练 $$K$$ 个独立的 $$E_{\mathrm{hull}}$$ surrogate：

$$
f_k(\mathcal C)=\widehat E_{\mathrm{hull}}^{(k)}(\mathcal C),
\qquad k=1,\ldots,K,
$$

各模型使用不同随机初始化、数据顺序和 bootstrap 训练集。集成预测为

$$
\bar f(\mathcal C)=\frac1K\sum_{k=1}^{K}f_k(\mathcal C).
$$

用模型间分歧定义 epistemic/OOD uncertainty：

$$
u(\mathcal C)
=\sqrt{\frac1{K-1}\sum_{k=1}^{K}
\left[f_k(\mathcal C)-\bar f(\mathcal C)\right]^2}.
$$

在按结构簇划分的验证集上校准 $$u_{\max}$$，使

$$
Q_{1-\alpha}\!\left(
\left|\bar f-E_{\mathrm{hull}}\right|
\;\middle|\;u\le u_{\max}
\right)
\le\delta_E.
$$

坐标扰动、晶格缩放和训练集外结构仅用于检验：OOD 样本应具有更大的 $$u$$。

## 3. 稳定概率

连续预测中用事件 $$E_{\mathrm{hull}}\le\varepsilon_E$$ 表示
$$E_{\mathrm{hull}}\to0$$。定义可微稳定概率

$$
p_{\mathrm{stab}}(\mathcal C)
=\frac1K\sum_{k=1}^{K}
\sigma\!\left(
\frac{\varepsilon_E-f_k(\mathcal C)}{\tau_E}
\right),
$$

其中 $$\sigma$$ 为 sigmoid，$$\tau_E>0$$ 为平滑温度。

## 4. 推理优化

记新颖性分数为

$$
s_{\mathrm{nov}}(\mathcal C;\mathcal D_{\mathrm{ref}}),
$$

分数越大表示结构越新颖。DFlow 推理阶段求解

$$
\boxed{
\begin{aligned}
\max_{\mathbf z_X,\mathbf z_L}\quad
&w_E\log p_{\mathrm{stab}}(\mathcal C(\mathbf z))
+w_Ns_{\mathrm{nov}}(\mathcal C(\mathbf z);\mathcal D_{\mathrm{ref}})
-\beta\lVert\mathbf z-\mathbf z_0\rVert_2^2\\
\mathrm{s.t.}\quad
&p_{\mathrm{stab}}(\mathcal C(\mathbf z))\ge p_{\min},\\
&\left\|\nabla_{\mathbf X}\bar f(\mathcal C(\mathbf z))\right\|_{\mathrm F}
\le\varepsilon_X,\\
&\left\|\nabla_{\mathbf L}\bar f(\mathcal C(\mathbf z))\right\|_{\mathrm F}
\le\varepsilon_L,\\
&u(\mathcal C(\mathbf z))\le u_{\max}.
\end{aligned}}
$$

其中 $$w_E,w_N,\beta\ge0$$。

实际反向传播使用无约束损失

$$
\begin{aligned}
\mathcal L(\mathbf z)
=&-w_E\log p_{\mathrm{stab}}
-w_Ns_{\mathrm{nov}}
+\beta\lVert\mathbf z-\mathbf z_0\rVert_2^2\\
&+\lambda_p[p_{\min}-p_{\mathrm{stab}}]_+^2
+\lambda_X[\lVert\nabla_{\mathbf X}\bar f\rVert_{\mathrm F}-\varepsilon_X]_+^2\\
&+\lambda_L[\lVert\nabla_{\mathbf L}\bar f\rVert_{\mathrm F}-\varepsilon_L]_+^2
+\lambda_u[u-u_{\max}]_+^2,
\end{aligned}
$$

其中 $$[a]_+=\max(0,a)$$。梯度驻点项对 $$\mathbf z$$ 的反向传播需要 surrogate 的二阶自动微分。

## 5. 实验流程

1. 训练并校准 $$K$$ 个 surrogate，确定 $$u_{\max}$$。
2. 冻结 DFlow 与全部 surrogate。
3. 采样 $$\mathbf z_0$$，固定对应的 $$(N,\mathbf A)$$。
4. 完整积分 DFlow，计算 $$p_{\mathrm{stab}}$$、新颖性、驻点约束和 OOD uncertainty。
5. 梯度穿过 surrogate 与完整 DFlow ODE，仅更新 $$(\mathbf z_X,\mathbf z_L)$$。
6. 对最终结构按 $$p_{\mathrm{stab}}\ge p_{\min}$$、$$u\le u_{\max}$$ 和驻点容差验收并去重。

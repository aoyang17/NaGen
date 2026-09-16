# Na–Fe–P–O 晶体结构 IPOPT 非线性优化问题

## 1. 任务范围

本模块不生成晶体。输入为一个已经基本成形的初始周期结构

$$
\mathcal C_0=(N,\mathbf A,\mathbf X_0,\mathbf L_0)
$$

以及目标参数。优化期间固定原子数 $N$ 与元素序列
$\mathbf A=(a_i)_{i=1}^{N}$，IPOPT 只优化分数坐标和晶格：

$$
q=(\mathbf X,\mathbf L),\qquad
\mathbf X=(\mathbf x_i)_{i=1}^{N},\quad
\mathbf L\in\mathbb R^{3\times3}.
$$

这一定义是直接的晶体空间局部精调；不存在 Flow、源变量、生成模型或生成一致性约束。

## 2. 优化目标

生成任务已经结束，因此直接精调不再优化结构新颖性。唯一性质目标使用
NaGen 根据原始数据集训练的 $E_{\mathrm{hull}}$ surrogate：

$$
\boxed{
\min_{\mathbf X,\mathbf L}\quad
\mathcal L_{\mathrm{prop}}
=\left(
\frac{\widehat E_{\mathrm{hull}}(\mathcal C)-E_*}{s_E}
\right)^2,
\qquad E_*=0
}
$$

其中

$$
\mathcal C=(N,\mathbf A,\mathbf X,\mathbf L).
$$

$\widehat E_{\mathrm{hull}}$ 必须直接加载指定的已训练 surrogate checkpoint，
并保持其训练时的元素映射、图截断、归一化参数和模型权重。UMA 不提供
$E_{\mathrm{hull}}$ 目标；它只用于下面的原子力和晶格应力约束。

## 3. 固定参数的输入可行性检查

$N$ 和 $\mathbf A$ 是固定参数，不是 IPOPT 变量。输入结构应先标准化为待优化的原胞表示，此时 $N=N_{\mathrm{prim}}$。下列条件只需在求解前检查一次；任一条件不成立时，本次固定组成问题无可行解，模块直接拒绝输入，而不是尝试在优化中改变组成。

$$
N=\operatorname{len}(\mathbf A),\qquad
23\le N\le184,\qquad
\operatorname{supp}(\mathbf A)=\{\mathrm{Na},\mathrm{Fe},\mathrm P,\mathrm O\}.
$$

定义

$$
(m,n,y)=\left(
\frac{N_{\mathrm{Na}}}{N_{\mathrm P}},
\frac{N_{\mathrm{Fe}}}{N_{\mathrm P}},
4-\frac{N_{\mathrm O}}{N_{\mathrm P}}
\right),
\qquad
\bar z_{\mathrm{Fe}}=\frac{3-2y-m}{n}.
$$

要求

$$
0\le y\le0.5,\qquad m\ge1,\qquad \bar z_{\mathrm{Fe}}\ge2.
$$

理论比容量仍按

$$
\begin{aligned}
M_{\mathrm{fu}}&=mM_{\mathrm{Na}}+nM_{\mathrm{Fe}}+M_{\mathrm P}+(4-y)M_{\mathrm O},\\
\kappa_{\mathrm{Fe}}&=n(4.5-\bar z_{\mathrm{Fe}}),\qquad
k=\min\{m,\kappa_{\mathrm{Fe}}\},\\
C_{\mathrm{th}}&=\frac{kF}{3.6M_{\mathrm{fu}}}\ge130\ \mathrm{mAh\,g^{-1}}
\end{aligned}
$$

检查。这里取
$M_{\mathrm{Na}}=22.99$、$M_{\mathrm{Fe}}=55.85$、$M_{\mathrm P}=30.97$、
$M_{\mathrm O}=16.00\ \mathrm{g\,mol^{-1}}$ 和
$F=96485.33212\ \mathrm{C\,mol^{-1}}$。

初始结构还必须具有每个 P 恰好 4 个 O 邻居、每个 Fe 具有 4、5 或 6 个 O 邻居，并且任意相邻 Fe 配位多面体至多共享两个 O。模块由 $\mathcal C_0$ 建立固定的 P–O 和 Fe–O 周期邻接集合；因此每个中心的 O 邻居身份、配位数以及共享 O 数在优化期间均不得改变。为避免数值解落在成键截断面上，初始键满足 $d\le r_{M\mathrm O}-\delta_c$，初始非键满足 $d\ge r_{M\mathrm O}+\delta_c$，默认 $\delta_c=0.01\ \mathring{\mathrm A}$。该模块只精调已有配位拓扑，不负责修复错误配位或搜索新的成键拓扑。

## 4. IPOPT 约束

下列约束与 surrogate 性质目标始终属于同一个 NLP，并在每一次 IPOPT
函数与 Jacobian 评估中同时激活；不设置预弛豫、分阶段目标或分段求解。

对周期像定义

$$
\mathbf r_{ij\mathbf n}
=(\mathbf x_j-\mathbf x_i+\mathbf n)\mathbf L,
\qquad
d_{ij\mathbf n}^2=\mathbf r_{ij\mathbf n}\cdot\mathbf r_{ij\mathbf n}.
$$

周期像集合 $\mathcal S\subset\mathbb Z^3$ 在求解前建立，并且必须覆盖当前局部优化域内所有可能进入最大几何截断的周期像。约束使用每个周期像的平方距离，不在 IPOPT 计算图中使用最小值、最近像切换或整数邻居计数。如果精确周期检查发现集合外的新活动周期像，模块将其加入 $\mathcal S$ 并重启 IPOPT，直至没有遗漏的违反约束周期像。

对每个 P 或 Fe，以初始配位 O 的展开笛卡尔坐标建立其凸包三角面集合 $\mathcal F_i$。每个面的顶点顺序固定为初始结构中的内法向方向。记中心原子位于局部原点，面对中心的有向四面体积函数为

$$
\phi_{ijkl}(\mathbf X,\mathbf L)
=\frac{1}{6}
\left[
(\mathbf r_{ik}-\mathbf r_{ij})
\times
(\mathbf r_{il}-\mathbf r_{ij})
\right]\cdot(-\mathbf r_{ij}),
\qquad (j,k,l)\in\mathcal F_i .
$$

对每一对至少共享一个 O 的相邻 Fe 多面体，在初始结构中分别固定朝向对方 Fe 中心的凸包面，并以其单位法向 $\widehat{\mathbf n}_i$、$\widehat{\mathbf n}_k$ 定义夹角。固定邻接拓扑使“共享 O 不超过 2”在整个连续优化域内恒成立；另以法向夹角排除相邻面共面。

| 名称 | IPOPT 中的数学形式 | 实现说明 |
|---|---|---|
| 分数坐标范围 | $0\le x_{i\alpha}\le1$ | 作为变量上下界；输出时将 1 周期归一化为 0。 |
| 正体积与体积窗口 | $10.5N\le\det(\mathbf L)\le20.5N\ \mathring{\mathrm A}^3$ | 正的体积下界同时保证右手、非退化晶胞。 |
| PBC 最小距离 | $d_{ij\mathbf n}^2-d_{\min}^2(a_i,a_j)\ge0$，对全部 $i<j$ 及相关 $\mathbf n\in\mathcal S$ | 将一个非光滑的最小距离约束展开成一组光滑标量不等式。 |
| 固定 P–O 配位 | 对初始邻接集合中的键，$d_{ij\mathbf n}\le r_{\mathrm{P-O}}-\delta_c$；对其余 P–O 接触，$d_{ij\mathbf n}\ge r_{\mathrm{P-O}}+\delta_c$ | 固定初始的 4 配位 O 邻居身份，避免优化中出现整数 `cardinality` 或卡在截断面。 |
| 固定 Fe–O 配位 | 对初始邻接集合中的键，$d_{ij\mathbf n}\le r_{\mathrm{Fe-O}}-\delta_c$；对其余 Fe–O 接触，$d_{ij\mathbf n}\ge r_{\mathrm{Fe-O}}+\delta_c$ | 固定初始 Fe 的配位数和 O 邻居身份；若输入配位不合法则拒绝。 |
| P/Fe–O 多面体中心 | $\phi_{ijkl}(\mathbf X,\mathbf L)\ge\varepsilon_V>0$，对所有中心 $i$ 和固定凸包面 $(j,k,l)\in\mathcal F_i$ | 固定面拓扑下，以光滑的有向体积不等式保证 P 或 Fe 严格位于其 O 多面体内部；不引入额外设计变量。 |
| Fe–O 多面体相邻关系 | $|\widehat{\mathbf n}_i\cdot\widehat{\mathbf n}_k|\le\cos\theta_{\min}$，默认 $\theta_{\min}=30^\circ$；固定邻居集合满足 $|\mathcal N_i^{\mathrm O}\cap\mathcal N_k^{\mathrm O}|\le2$ | 对初始相邻 Fe 对固定面对，阻止两个相邻多面体面趋于共面；固定配位同时保证共享 O 数不变。 |
| Surrogate 凸包能 | $0\le\widehat E_{\mathrm{hull}}(\mathcal C)\le0.150\ \mathrm{eV\,atom^{-1}}$ | 下界保持凸包能的非负物理定义，上界为稳定性硬约束；预测值和 Jacobian 均来自同一个冻结 surrogate。 |
| 原子力收敛 | $\lVert\mathbf F_i(\mathcal C)\rVert_2^2-F_{\mathrm{tol}}^2\le0$，$i=1,\ldots,N$ | 逐原子展开后与 $F_{\max}\le F_{\mathrm{tol}}$ 等价，且不使用非光滑的 `max`；$F_{\mathrm{tol}}=0.01\ \mathrm{eV}\,\mathring{\mathrm A}^{-1}$。 |
| 晶格应力收敛 | $\lVert\boldsymbol\sigma_{\mathrm{UMA}}(\mathcal C)\rVert_{\mathrm F}^2-\sigma_{\mathrm{tol}}^2\le0$ | $\sigma_{\mathrm{tol}}=0.10\ \mathrm{GPa}=6.242\times10^{-4}\ \mathrm{eV}\,\mathring{\mathrm A}^{-3}$。 |

采用如下最小距离阈值：

| 元素对 | $d_{\min}$ (Å) |
|---|---:|
| P–O | 1.40 |
| Fe–O | 1.55 |
| Na–O | 2.00 |
| O–O | 2.00 |
| P–P | 2.60 |
| Na–Na | 2.20 |

其他元素对使用

$$
d_{\min}(a,b)=0.75\,[r_{\mathrm{cov}}(a)+r_{\mathrm{cov}}(b)].
$$

固定配位拓扑的截断为

$$
r_{\mathrm{P-O}}=2.00\ \mathring{\mathrm A},
\qquad
r_{\mathrm{Fe-O}}=2.50\ \mathring{\mathrm A}.
$$

为避免初始键恰好位于截断面造成邻接歧义，实现可在截断两侧设置很小的固定滞回宽度 $\delta_r$：初始键保持在 $r-\delta_r$ 内，初始非键保持在 $r+\delta_r$ 外。

## 5. 求解器导数接口与验收

Surrogate 输出、UMA 力和 UMA 应力必须对 $(\mathbf X,\mathbf L)$ 提供一致的一阶导数。由于力本身是 UMA 势能的一阶导数，原子力约束的 Jacobian 需要 UMA 势能关于结构的二阶导数。IPOPT 使用 limited-memory Hessian 近似时，不要求模块提供完整的拉格朗日 Hessian。

初始结构应尽量满足固定拓扑和几何约束；IPOPT 可以从轻度不可行点恢复，但本模块不承诺从严重原子重叠、错误配位或退化晶格中重建晶体。

求解结束后，模块使用精确周期邻居搜索重新计算最小距离和整数配位，并重新计算 surrogate 凸包能、UMA 逐原子力和 UMA 应力。只有全部精确检查均通过的结构才标记为可行解；分数坐标在输出前统一映射到 $[0,1)$。

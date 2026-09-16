# HybridOptimization：由晶体结构与能量模型确定的约束

本文整理两类约束：

1. 由 $$(N,A,X,L)$$ 直接确定的晶胞、距离、配位和多面体几何约束；
2. 由 $$(N,A,X,L,\widehat E_{\mathrm{hull}})$$ 确定的能量、力和应力约束。

其中，$$N$$ 为原子数，$$A=(a_1,\ldots,a_N)$$ 为元素序列，$$X$$ 为分数坐标矩阵，$$L$$ 为晶格矩阵。

## 一、由 $$(N,A,X,L)$$ 确定的约束

### 1. 结构维度一致

$$
X\in\mathbb R^{N\times3},
\qquad
L\in\mathbb R^{3\times3}.
$$

### 2. 分数坐标范围

$$
0\le x_{i\alpha}<1,
\qquad
\forall i\in\{1,\ldots,N\},
\quad
\alpha\in\{1,2,3\}.
$$

### 3. 有限结构

$$
x_{i\alpha}\in\mathbb R,
\qquad
L_{\alpha\beta}\in\mathbb R,
$$

并且 $$X$$ 和 $$L$$ 的所有分量均为有限数值。

### 4. 正晶胞体积

$$
V=\det L>0.
$$

### 5. 单原子体积

$$
10.5
\le
\frac{\det L}{N}
\le
20.5
\qquad
\mathrm{\mathring A^3/atom}.
$$

### 6. 规约胞原子数

$$
N_{\mathrm{prim}}
=
\operatorname{PrimitiveCount}
\left(N,A,X,L;\varepsilon_{\mathrm{sym}}\right),
$$

$$
23\le N_{\mathrm{prim}}\le184.
$$

该约束依赖对称性规约算法及其数值容差 $$\varepsilon_{\mathrm{sym}}$$。

### 7. 周期性边界条件下的原子间距

$$
d_{ij}^{\mathrm{PBC}}(X,L)
=
\min_{\mathbf n\in\mathbb Z^3}
\left\|
(\mathbf x_j-\mathbf x_i+\mathbf n)L
\right\|_2.
$$

### 8. 元素对最小距离

$$
d_{ij}^{\mathrm{PBC}}
\ge
d_{\min}(a_i,a_j),
\qquad
\forall i<j.
$$

当前主要元素对下界为：

$$
d_{\min}^{\mathrm{P-O}}=1.40\ \mathrm{\mathring A},
\qquad
d_{\min}^{\mathrm{Fe-O}}=1.55\ \mathrm{\mathring A},
$$

$$
d_{\min}^{\mathrm{Na-O}}=2.00\ \mathrm{\mathring A},
\qquad
d_{\min}^{\mathrm{O-O}}=2.00\ \mathrm{\mathring A},
$$

$$
d_{\min}^{\mathrm{P-P}}=2.60\ \mathrm{\mathring A},
\qquad
d_{\min}^{\mathrm{Na-Na}}=2.20\ \mathrm{\mathring A}.
$$

### 9. P--O 邻居集合

$$
\mathcal N_i^{\mathrm{P-O}}
=
\left\{
j:
a_j=\mathrm O,
\quad
d_{ij}^{\mathrm{PBC}}\le2.00\ \mathrm{\mathring A}
\right\}.
$$

### 10. P--O 成键距离

$$
a_i=\mathrm P,
\quad
j\in\mathcal N_i^{\mathrm{P-O}}
\Longrightarrow
1.40
\le
d_{ij}^{\mathrm{PBC}}
\le
2.00
\qquad
\mathrm{\mathring A}.
$$

### 11. P 四配位

$$
a_i=\mathrm P
\Longrightarrow
\left|\mathcal N_i^{\mathrm{P-O}}\right|=4.
$$

### 12. Fe--O 邻居集合

$$
\mathcal N_i^{\mathrm{Fe-O}}
=
\left\{
j:
a_j=\mathrm O,
\quad
d_{ij}^{\mathrm{PBC}}\le2.50\ \mathrm{\mathring A}
\right\}.
$$

### 13. Fe--O 成键距离

$$
a_i=\mathrm{Fe},
\quad
j\in\mathcal N_i^{\mathrm{Fe-O}}
\Longrightarrow
1.55
\le
d_{ij}^{\mathrm{PBC}}
\le
2.50
\qquad
\mathrm{\mathring A}.
$$

### 14. Fe 配位数

$$
a_i=\mathrm{Fe}
\Longrightarrow
\left|\mathcal N_i^{\mathrm{Fe-O}}\right|
\in\{4,5,6\}.
$$

### 15. P/Fe 位于 O 配位多面体内部

对于任意 P 或 Fe 中心 $$i$$，要求存在正权重 $$\lambda_{ij}$$：

$$
\lambda_{ij}>0,
\qquad
\sum_{j\in\mathcal N_i^{\mathrm O}}\lambda_{ij}=1,
$$

使得：

$$
\sum_{j\in\mathcal N_i^{\mathrm O}}
\lambda_{ij}
\mathbf r_{ij}^{\mathrm{PBC}}
=
\mathbf 0.
$$

### 16. Fe 多面体共享 O 数

$$
a_i=a_k=\mathrm{Fe}
\Longrightarrow
\left|
\mathcal N_i^{\mathrm{Fe-O}}
\cap
\mathcal N_k^{\mathrm{Fe-O}}
\right|
\le2.
$$

### 17. Fe 多面体非共面

$$
\left|
\mathbf n_i\cdot\mathbf n_k
\right|
\le
\cos\theta_{\min},
$$

其中 $$\mathbf n_i$$ 和 $$\mathbf n_k$$ 由共享 O 及对应的 Fe--O 周期几何向量确定。

## 二、由 $$(N,A,X,L,\widehat E_{\mathrm{hull}})$$ 确定的约束

### 1. 预测原子能

$$
\widehat e_\theta
=
\widehat e_\theta(N,A,X,L).
$$

### 2. Raw hull energy

固定组成对应的参考凸包能记为 $$e_{\mathrm{ref,hull}}(A)$$：

$$
\widehat e_{\mathrm{raw}}
=
\widehat e_\theta(N,A,X,L)
-
e_{\mathrm{ref,hull}}(A).
$$

### 3. 非负预测凸包能

$$
\widehat E_{\mathrm{hull}}
=
\max\left\{0,\widehat e_{\mathrm{raw}}\right\}.
$$

### 4. 凸包能阈值

$$
\widehat E_{\mathrm{hull}}
\le
0.150
\qquad
\mathrm{eV/atom}.
$$

### 5. 笛卡尔坐标

$$
R=XL.
$$

### 6. 原子力

固定组成时，参考凸包能与坐标无关，因此：

$$
\mathbf F_i
=
-\frac{\partial E_{\mathrm{total}}}{\partial\mathbf r_i}
=
-N
\frac{\partial\widehat e_{\mathrm{raw}}}
{\partial\mathbf r_i}.
$$

### 7. 最大原子力

$$
F_{\max}
=
\max_{1\le i\le N}
\left\|\mathbf F_i\right\|_2.
$$

### 8. 原子力收敛

$$
F_{\max}
\le
0.01
\qquad
\mathrm{eV/\mathring A}.
$$

### 9. 晶格应力

令晶胞体积为：

$$
V=\det L.
$$

晶格应力定义为：

$$
\boldsymbol\sigma
=
\frac{1}{V}
\frac{\partial\left(N\widehat e_{\mathrm{raw}}\right)}
{\partial\boldsymbol\varepsilon},
$$

其中 $$\boldsymbol\varepsilon$$ 为晶格应变。

### 10. 应力 Frobenius 范数

$$
\sigma_{\mathrm F}
=
\left\|\boldsymbol\sigma\right\|_{\mathrm F}.
$$

### 11. 晶格应力收敛

$$
\left\|\boldsymbol\sigma\right\|_{\mathrm F}
\le
0.10
\qquad
\mathrm{GPa}.
$$

## 三、汇总

由 $$(N,A,X,L)$$ 确定：

$$
\boxed{
\text{晶胞、周期距离、元素对距离、P/Fe--O 配位、多面体几何}
}
$$

加入固定的能量预测模型和参考凸包后，由 $$(N,A,X,L,\widehat E_{\mathrm{hull}})$$ 确定：

$$
\boxed{
\widehat E_{\mathrm{hull}},\quad
\mathbf F,\quad
F_{\max},\quad
\boldsymbol\sigma
}
$$

力的计算应优先对 $$\widehat e_{\mathrm{raw}}$$ 或原始预测能量求导。直接对

$$
\max\left\{0,\widehat e_{\mathrm{raw}}\right\}
$$

求导会在零点产生不可微问题，并在 $$\widehat e_{\mathrm{raw}}<0$$ 时产生零梯度。

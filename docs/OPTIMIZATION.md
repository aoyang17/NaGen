# 磷酸盐 Na 正极统一优化问题（Na–Fe–P–O）

| 类别 | 名称 | 数学定义 | 解释 |
|---|---|---|---|
| 优化目标 | 统一标量目标 | \(\displaystyle \min_{z}\ \mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{hull}}-\lambda_{\mathrm{nov}}f_{\mathrm{nov}}+\lambda_{\mathrm{constraint}}\mathcal L_{\mathrm{constraint}}+\beta\lVert z-\bar z\rVert^2\) | 在冻结 Flow 参数下仅优化初始噪声；凸包能、新颖性、全部可微约束和流形正则在同一目标中处理。 |
| 设计变量 | 原子数 | \(N\in\mathbb Z_{>0}\) | 周期晶胞中的原子总数。 |
| 设计变量 | 元素序列 | \(\mathbf A=(a_i)_{i=1}^{N}\) | 指定每个原子位点的元素。 |
| 设计变量 | 分数坐标 | \(\mathbf X=(\mathbf x_i)_{i=1}^{N},\ \mathbf x_i\in[0,1)^3\) | 指定晶胞内的原子位置。 |
| 设计变量 | 晶格矩阵 | \(\mathbf L\in\mathbb R^{3\times3}\) | 指定晶胞大小和形状。 |
| 约束条件 | 结构一致性 | \(N=\operatorname{len}(\mathbf A)=\operatorname{rows}(\mathbf X),\ \mathbf X,\mathbf L\ \text{finite},\ \det(\mathbf L)>0\) | \(N,\mathbf A,\mathbf X,\mathbf L\) 必须共同定义有限、正体积的周期晶体。 |
| 约束条件 | 元素组成 | \(\operatorname{supp}(\mathbf A)=\{\mathrm{Na},\mathrm{Fe},\mathrm P,\mathrm O\}\) | 候选必须且只能包含 Na、Fe、P、O，四种元素均须出现。 |
| 约束条件 | 组成域 | \(\displaystyle 0\le y\le0.5,\quad m\ge1,\quad m\ge n,\quad n\le1\) | 限制 Na、Fe 与 O 的归一化化学计量。[A1] |
| 约束条件 | 形式电中性 | \(\displaystyle \sum_{i:a_i=\mathrm{Fe}}\bar z_{\mathrm{Fe}}+N_{\mathrm{Na}}+5N_{\mathrm P}-2N_{\mathrm O}=0\iff \bar z_{\mathrm{Fe}}=\frac{3-2y-m}{n}\in[2,4.5]\subset\mathbb R_{>0}\) | \(\bar z_{\mathrm{Fe}}\) 为 Fe 平均形式价态，不要求整数价态分配。[A3] |
| 约束条件 | Fe 可氧化当量 | \(\displaystyle \kappa_{\mathrm{Fe}}:=n(4.5-\bar z_{\mathrm{Fe}})=m+4.5n+2y-3\) | 每摩尔归一化化学式中，Fe 氧化至平均 \(+4.5\) 可提供的电子当量。[A4] |
| 约束条件 | 可脱 Na 当量 | \(\displaystyle k:=\min\{m,\kappa_{\mathrm{Fe}}\}\) | 可脱 Na 量同时受 Na 含量与 Fe 可氧化当量限制。[A4] |
| 约束条件 | 理论比容量 | \(\displaystyle M_{\mathrm{fu}}:=mM_{\mathrm{Na}}+nM_{\mathrm{Fe}}+M_{\mathrm P}+(4-y)M_{\mathrm O},\quad C_{\mathrm{th}}:=\frac{kF}{\gamma_{\mathrm{mAh}}M_{\mathrm{fu}}}\ge C_{\min}\) | \(M_\alpha\) 为元素 \(\alpha\) 的原子摩尔质量，\(F\) 为 Faraday 常数，\(\gamma_{\mathrm{mAh}}\) 为电量单位换算因子。[A4][^capacity-constants] |
| 约束条件 | PBC 最小距离 | \(\displaystyle d_{ij}^{\mathrm{PBC}}\ge d_{\min}(a_i,a_j),\quad\forall i<j\) | 排除原子重叠和异常短接触。[A5] |
| 约束条件 | 体积与规约胞大小 | \(\displaystyle 10.5\le\frac{\det(\mathbf L)}N\le20.5\ \mathring{\mathrm A}^{3}\,\mathrm{atom}^{-1},\quad23\le N_{\mathrm{prim}}\le184\) | 体积窗口来自数据审计外扩，原子数为 Na–Fe–P–O 子集实测范围。[A2] |
| 约束条件 | P–O 配位 | \(\displaystyle a_i=\mathrm P\Rightarrow\operatorname{card}(\mathcal N_i^{\{\mathrm O\}})=4\) | 每个 P 必须与 4 个 O 成键。[A6] |
| 约束条件 | Fe–O 配位 | \(\displaystyle a_i=\mathrm{Fe}\Rightarrow\operatorname{card}(\mathcal N_i^{\{\mathrm O\}})\in\{4,5,6\}\) | 每个 Fe 只允许四、五或六配位氧环境。[A6] |
| 约束条件 | P/Fe–O 多面体中心 | \(\displaystyle \exists\lambda_{ij}>0:\ \sum_{j\in\mathcal N_i^{\mathrm O}}\lambda_{ij}=1,\ \sum_{j\in\mathcal N_i^{\mathrm O}}\lambda_{ij}\mathbf r_{ij}=\mathbf0,\quad a_i\in\{\mathrm P,\mathrm{Fe}\}\) | P 或 Fe 位于其 O 配位多面体内部；梯度优化使用 soft-neighbor 权重。[A10] |
| 约束条件 | Fe–O 多面体相邻关系 | 见 A10 的陈列公式 | 相邻 Fe 多面体不得共用三个 O，且不得共面。[A10] |
| 约束条件 | Fe–Fe 距离 | \(\displaystyle d_{ik}^{\mathrm{PBC}}\ge d_{\mathrm{FeFe}}^{\min}:=2d_{\mathrm{FeO}}^{\min}\sin(\theta_{\min}/2),\quad a_i=a_k=\mathrm{Fe}\) | 由 Fe–O 最小距离和 Fe–O–Fe 最小夹角推出。[A10] |
| 约束条件 | 非重复性 | \(\displaystyle \mathcal C\not\simeq\mathcal C_k,\quad\forall\mathcal C_k\in\mathcal D_{\mathrm{ref}}\) | 候选不得与参考集中的结构等价。[A7] |
| 约束条件 | 凸包能 | \(\displaystyle E_{\mathrm{hull}}\le0.150\ \mathrm{eV\,atom^{-1}}\) | 将凸包能上限作为候选必须满足的强约束。[A8] |
| 约束条件 | UMA 原子力收敛 | \(\displaystyle F_{\max}(\mathcal C):=\max_i\left\lVert-\nabla_{\mathbf r_i}E_{\mathrm{UMA}}(\mathcal C)\right\rVert_2\le F_{\mathrm{tol}},\quad F_{\mathrm{tol}}:=0.01\ \mathrm{eV\,\mathring A^{-1}}\) | 将 FIRE/BFGS 的最终原子力收敛条件直接作为统一优化的可行约束；不再定义独立的生成后弛豫阶段。[A9] |
| 约束条件 | UMA 晶格应力收敛 | \(\displaystyle \lVert\boldsymbol\sigma_{\mathrm{UMA}}(\mathcal C)\rVert_{\mathrm F}\le\sigma_{\mathrm{tol}},\quad \sigma_{\mathrm{tol}}:=0.10\ \mathrm{GPa}\) | 将晶胞自由度的收敛条件直接并入同一优化问题；应力由 UMA 对晶格的导数给出。[A9] |

| 类别 | 名称 | 数学定义 | 解释 |
|---|---|---|---|
| Appendix | A1：组成归一化 | \(\displaystyle N_\alpha=\sum_i\mathbf1[a_i=\alpha],\quad (m,n,y)=\left(\frac{N_{\mathrm{Na}}}{N_{\mathrm P}},\frac{N_{\mathrm{Fe}}}{N_{\mathrm P}},4-\frac{N_{\mathrm O}}{N_{\mathrm P}}\right)\) | 归一化化学式为 \(\mathrm{Na}_m\mathrm{Fe}_n\mathrm P\mathrm O_{4-y}\)；原式中的 \(c\) 统一记为 \(y\)。 |
| Appendix | A2：数据支持口径 | \(4196\ \text{structures},\ 116\ \text{reduced formulas}\) | 精确 Na–Fe–P–O 子集实测：\(23\le N_{\mathrm{prim}}\le184\)，\(11.3885\le V/N\le17.7468\ \mathring{\mathrm A}^{3}\,\mathrm{atom}^{-1}\)；体积硬约束采用外扩窗口 10.5–20.5。 |
| Appendix | A3：形式价态假设 | \(\mathrm{Na}^{+},\ \mathrm P^{5+},\ \mathrm O^{2-},\ \mathrm{Fe}^{\bar z_{\mathrm{Fe}}+},\quad \bar z_{\mathrm{Fe}}\in[2,4.5]\subset\mathbb R_{>0}\) | \(+4.5\) 是形式电中性的允许上限；Fe 平均价态按总电荷计算。 |
| Appendix | A4：容量口径 | \(\displaystyle \kappa_{\mathrm{Fe}}=n(4.5-\bar z_{\mathrm{Fe}}),\quad k=\min\{m,\kappa_{\mathrm{Fe}}\},\quad C_{\mathrm{th}}=\frac{kF}{\gamma_{\mathrm{mAh}}M_{\mathrm{fu}}}\ge C_{\min}\) | 容量计算以 Fe 氧化至平均 \(+4.5\) 为终点；数值见“理论比容量”脚注。 |
| Appendix | A5：最小距离阈值 | \(d_{\min}^{\mathrm{P-O}}=1.40\), \(d_{\min}^{\mathrm{Fe-O}}=1.55\), \(d_{\min}^{\mathrm{Na-O}}=2.00\), \(d_{\min}^{\mathrm{O-O}}=2.00\), \(d_{\min}^{\mathrm{P-P}}=2.60\), \(d_{\min}^{\mathrm{Na-Na}}=2.20\ \mathring{\mathrm A}\) | 其他元素对使用 \(d_{\min}(a,b)=0.75[r_{\mathrm{cov}}(a)+r_{\mathrm{cov}}(b)]\)。 |
| Appendix | A6：成键与配位截断 | \(r_{\mathrm{P-O}}=2.00\ \mathring{\mathrm A},\quad r_{\mathrm{Fe-O}}=2.50\ \mathring{\mathrm A}\) | \(\mathcal N_i^B=\{j\ne i:a_j\in B,\ d_{ij}^{\mathrm{PBC}}\le r_{a_i-a_j}\}\)；配位数和 P–Fe–O 周期成键图都按此固定截断计算。 |
| Appendix | A7：去重口径 | \(\mathcal C\simeq\mathcal C_k\) | 使用固定容差的 StructureMatcher 或等价结构指纹，并先对晶胞做统一规约化。 |
| Appendix | A8：凸包能口径 | \(\displaystyle E_{\mathrm{hull}}(\mathcal C):=\max\!\left\{0,\ E_{\mathrm{UMA}}(\mathcal C)+C_{\mathrm{MP2020}}(\operatorname{comp}(\mathcal C))-E_{\mathrm{MP,hull}}(\operatorname{comp}(\mathcal C))\right\}\) | 与 `relaxation_n_ehull.py` 一致；固定组成时修正项和参考凸包为常数，梯度只由 UMA 能量经冻结 Flow 回传。候选在统一优化得到的同一结构上直接重算。 |
| Appendix | A9：弛豫收敛的统一化 | \(\displaystyle F_{\max}\le0.01\ \mathrm{eV\,\mathring A^{-1}},\quad \lVert\boldsymbol\sigma_{\mathrm{UMA}}\rVert_{\mathrm F}\le0.10\ \mathrm{GPa}\) | FIRE 本质上是对原子位置/晶胞的优化器；因此只保留其物理收敛条件作为统一可行域。梯度阶段可用 \(\mathcal L_F=\sum_i\operatorname{softplus}((\lVert\mathbf F_i\rVert-F_{\mathrm{tol}})/\tau_F)^2\) 与 \(\mathcal L_\sigma=\operatorname{softplus}((\lVert\boldsymbol\sigma\rVert_F-\sigma_{\mathrm{tol}})/\tau_\sigma)^2\) 的连续 barrier，不再执行独立 FIRE/BFGS 后处理。 |
| Appendix | A10：可微约束总项 | 见下方陈列公式 | 统一约束项为 \(\mathcal L_{\mathrm{constraint}}=\mathcal L_{\mathrm{A5}}+\mathcal L_{\mathrm{A6}}+\mathcal L_{\mathrm{new}}+\lambda_F\mathcal L_F+\lambda_\sigma\mathcal L_\sigma\)；软代理仅用于梯度优化，最终直接检查精确 PBC 距离、整数配位、力和应力。实现索引见 `SHOOTINGFLOW_CONSTRAINT_EQUATIONS.md`。 |

### A10 陈列公式

周期 soft-min 距离与 O 邻居软权重：

\[
\begin{aligned}
\widetilde d_{ij}
  &=-\tau_d\log\!\sum_{\mathbf n\in\{-1,0,1\}^3}
    \exp\!\left[-\frac{\lVert(\mathbf x_j-\mathbf x_i+\mathbf n)\mathbf L\rVert_2}{\tau_d}\right],\\
s_{ij}^{(M)}
  &=\mathbf 1[a_j=\mathrm O]\,
    \sigma\!\left(\frac{r_{M\mathrm O}-\widetilde d_{ij}}{\tau_c}\right),
    \qquad M\in\{\mathrm P,\mathrm{Fe}\}.
\end{aligned}
\]

对 Fe 位点 (i,k)，多面体邻近约束为：

\[
\begin{aligned}
S_{ik}
  &=\sum_{j:a_j=\mathrm O}s_{ij}^{(\mathrm{Fe})}s_{kj}^{(\mathrm{Fe})}
    \le 2,\\
\left|\mathbf n_i\!\cdot\!\mathbf n_k\right|
  &\le \cos\theta_{\min},
\qquad a_i=a_k=\mathrm{Fe},\ i<k.
\end{aligned}
\]

第一式禁止两个 Fe–O 多面体共享三个 O，第二式禁止其配位面共面。

[^capacity-constants]: 取 \(M_{\mathrm{Na}}=22.99\)、\(M_{\mathrm{Fe}}=55.85\)、\(M_{\mathrm P}=30.97\)、\(M_{\mathrm O}=16.00\ \mathrm{g\,mol^{-1}}\)，\(F=96485.33212\ \mathrm{C\,mol^{-1}}\)，\(\gamma_{\mathrm{mAh}}=3.6\ \mathrm{C\,mAh^{-1}}\)；\(C_{\min}=130\ \mathrm{mAh\,g^{-1}}\) 为容量筛选阈值。由此 \(F/\gamma_{\mathrm{mAh}}\approx26801\ \mathrm{mAh\,mol^{-1}}\)。

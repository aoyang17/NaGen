# 磷酸盐Na正极统一混合优化问题表（Na–Fe–P–O）

| 类别 | 名称 | 数学定义 | 解释 |
|---|---|---|---|
| 优化目标 | 稳定性–新颖性加权二次目标 | \(\displaystyle \min_{z}\ \mathcal L_{\mathrm{prop}}=w_E\left(\frac{E_{\mathrm{hull}}^{\mathrm{UMA}}-E_*}{s_E}\right)^2+w_N\left(\frac{f_* - f_{\mathrm{nov}}}{s_N}\right)^2,\quad w_E,w_N\ge0,\ w_E+w_N=1\) | 将凸包能最小化与新颖性最大化统一为一个标量目标；对两个标量输出均为二次凸形式。取 \(E_*=0\)，\(f_*\) 为预设新颖性目标，\(s_E,s_N\) 为固定尺度。[A8] |
| 设计变量 | 原子数 | \(N\in\mathbb Z_{>0}\) | 周期晶胞中的原子总数。 |
| 设计变量 | 元素序列 | \(\mathbf A=(a_i)_{i=1}^{N}\) | 指定每个原子位点的元素。 |
| 设计变量 | 分数坐标 | \(\mathbf X=(\mathbf x_i)_{i=1}^{N},\ \mathbf x_i\in[0,1)^3\) | 指定晶胞内的原子位置；也是原弛豫任务更新的原子位置变量。 |
| 设计变量 | 晶格矩阵 | \(\mathbf L\in\mathbb R^{3\times3}\) | 指定晶胞大小和形状；也是原弛豫任务通过晶胞过滤器更新的变量。 |
| 设计变量 | Flow 源变量 | \(\mathbf z=(\mathbf z_A,\mathbf z_X,\mathbf z_L)\in\mathcal Z\) | 冻结生成模型后实际交给统一求解器更新的连续源变量。[A10] |
| 约束条件 | Flow 生成一致性 | \(\displaystyle \mathcal C=(N,\mathbf A,\mathbf X,\mathbf L)=\Phi_\theta(N,\mathbf z),\quad\theta=\theta_0\) | 晶体由冻结的 Flow 映射产生；优化期间不更新模型参数。 |
| 约束条件 | 结构一致性 | \(N=\operatorname{len}(\mathbf A)=\operatorname{rows}(\mathbf X),\ \mathbf X,\mathbf L\ \text{finite},\ \det(\mathbf L)>0\) | 所有变量必须共同定义有限、正体积的周期晶体。 |
| 约束条件 | 元素组成 | \(\operatorname{supp}(\mathbf A)=\{\mathrm{Na},\mathrm{Fe},\mathrm P,\mathrm O\}\) | 候选必须且只能包含 Na、Fe、P、O，四种元素均须出现。 |
| 约束条件 | 组成域 | \(\displaystyle 0\le y\le0.5,\quad m\ge1\) | 独立限制氧缺位范围和最低 Na 含量；\(m\ge n\) 与 \(n\le1\) 可由本行和 Fe 价态下界推出，故不重复列入。[A1][A3] |
| 约束条件 | Fe 平均形式价态 | \(\displaystyle \bar z_{\mathrm{Fe}}:=\frac{3-2y-m}{n}\ge2\) | 电中性决定 Fe 平均形式价态；容量约束已通过可氧化当量隐含 \(\bar z_{\mathrm{Fe}}<4.5\)，故不重复设置上界。[A3][A4] |
| 约束条件 | 理论比容量 | \(\displaystyle M_{\mathrm{fu}}:=mM_{\mathrm{Na}}+nM_{\mathrm{Fe}}+M_{\mathrm P}+(4-y)M_{\mathrm O},\quad C_{\mathrm{th}}:=\frac{kF}{\gamma_{\mathrm{mAh}}M_{\mathrm{fu}}}\ge C_{\min}\) | 容量计算口径与原优化问题保持一致。[A4][^constants] |
| 约束条件 | PBC 最小距离 | \(\displaystyle d_{ij}^{\mathrm{PBC}}\ge d_{\min}(a_i,a_j),\quad\forall i<j\) | 排除原子重叠和异常短接触。[A5] |
| 约束条件 | 体积与规约胞大小 | \(\displaystyle 10.5\le\frac{\det(\mathbf L)}N\le20.5\ \mathring{\mathrm A}^{3}\,\mathrm{atom}^{-1},\quad23\le N_{\mathrm{prim}}\le184\) | 体积窗口与原子数范围沿用现有问题定义。[A2] |
| 约束条件 | P–O 配位 | \(\displaystyle a_i=\mathrm P\Rightarrow\operatorname{card}(\mathcal N_i^{\{\mathrm O\}})=4\) | 每个 P 必须与 4 个 O 成键。[A6] |
| 约束条件 | Fe–O 配位 | \(\displaystyle a_i=\mathrm{Fe}\Rightarrow\operatorname{card}(\mathcal N_i^{\{\mathrm O\}})\in\{4,5,6\}\) | 每个 Fe 只允许四、五或六配位氧环境。[A6] |
| 约束条件 | P/Fe–O 多面体中心 | \(\displaystyle \exists\lambda_{ij}>0:\ \sum_{j\in\mathcal N_i^{\mathrm O}}\lambda_{ij}=1,\ \sum_{j\in\mathcal N_i^{\mathrm O}}\lambda_{ij}\mathbf r_{ij}=\mathbf0,\quad a_i\in\{\mathrm P,\mathrm{Fe}\}\) | P 或 Fe 位于其 O 配位多面体内部；梯度优化使用 soft-neighbor 权重。[A11] |
| 约束条件 | Fe–O 多面体相邻关系 | 见 A11 陈列公式 | 相邻 Fe 多面体不得共用三个 O，且不得共面。[A11] |
| 约束条件 | 非重复性 | \(\displaystyle \mathcal C\not\simeq\mathcal C_k,\quad\forall\mathcal C_k\in\mathcal D_{\mathrm{ref}}\) | 候选不得与参考集中的结构等价。[A7] |
| 约束条件 | 全 UMA 凸包能 | \(\displaystyle E_{\mathrm{hull}}^{\mathrm{UMA}}(\mathcal C)\le0.150\ \mathrm{eV\,atom^{-1}}\) | 以全 UMA 同能标凸包能作为稳定性强约束。[A8] |
| 约束条件 | **【原弛豫约束】原子力收敛** | \(\displaystyle F_{\max}(\mathcal C):=\max_i\left\lVert-\nabla_{\mathbf r_i}E_{\mathrm{UMA}}(\mathcal C)\right\rVert\le F_{\mathrm{tol}},\quad F_{\mathrm{tol}}:=0.01\ \mathrm{eV}\,\mathring{\mathrm A}^{-1}\) | 原 FIRE/BFGS 弛豫任务的最终力收敛条件，现直接并入统一可行域。[A9] |
| 约束条件 | **【新增弛豫约束】晶格应力收敛** | \(\displaystyle \left\lVert\boldsymbol\sigma_{\mathrm{UMA}}(\mathcal C)\right\rVert_{\mathrm F}\le\sigma_{\mathrm{tol}},\quad\sigma_{\mathrm{tol}}:=0.10\ \mathrm{GPa}\approx6.242\times10^{-4}\ \mathrm{eV}\,\mathring{\mathrm A}^{-3}\) | 将晶格弛豫显式写入统一可行域；该阈值用于判断数值收敛，不代表 UMA 的应力预测误差。[A9] |

统一求解器实际使用：

\[
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{prop}}
+\lambda_{\mathrm{constraint}}\mathcal L_{\mathrm{constraint}}
+\beta\lVert z-\bar z\rVert_2^2,
\qquad \lambda_{\mathrm{constraint}},\beta\ge0 .
\]

| 类别 | 名称 | 数学定义 | 解释 |
|---|---|---|---|
| Appendix | A1：组成归一化 | \(\displaystyle N_\alpha:=\sum_i\mathbf1[a_i=\alpha],\quad(m,n,y):=\left(\frac{N_{\mathrm{Na}}}{N_{\mathrm P}},\frac{N_{\mathrm{Fe}}}{N_{\mathrm P}},4-\frac{N_{\mathrm O}}{N_{\mathrm P}}\right)\) | 归一化化学式为 \(\mathrm{Na}_m\mathrm{Fe}_n\mathrm P\mathrm O_{4-y}\)。 |
| Appendix | A2：数据支持口径 | \(4196\ \text{structures},\quad116\ \text{reduced formulas}\) | 四元子集实测 \(23\le N_{\mathrm{prim}}\le184\)，体积硬约束使用外扩窗口 10.5–20.5。 |
| Appendix | A3：形式价态与派生关系 | \(\displaystyle \mathrm{Na}^{+},\ \mathrm P^{5+},\ \mathrm O^{2-},\ \mathrm{Fe}^{\bar z_{\mathrm{Fe}}+},\quad \bar z_{\mathrm{Fe}}=\frac{3-2y-m}{n}\ge2\) | 因 \(m\ge1\)、\(y\ge0\) 且 \(n>0\)，可直接推出 \(m\ge n\) 与 \(n\le1\)；二者不是独立约束。 |
| Appendix | A4：容量口径 | \(\displaystyle \kappa_{\mathrm{Fe}}:=n(4.5-\bar z_{\mathrm{Fe}}),\quad k:=\min\{m,\kappa_{\mathrm{Fe}}\},\quad C_{\mathrm{th}}:=\frac{kF}{\gamma_{\mathrm{mAh}}M_{\mathrm{fu}}}\ge C_{\min}\) | \(\kappa_{\mathrm{Fe}}\) 和 \(k\) 仅是容量的派生量，不是独立约束。由于 \(C_{\min}>0\)，容量约束蕴含 \(k>0\)、\(\kappa_{\mathrm{Fe}}>0\) 和 \(\bar z_{\mathrm{Fe}}<4.5\)。 |
| Appendix | A5：最小距离阈值 | \(d_{\min}^{\mathrm{P-O}}=1.40,\ d_{\min}^{\mathrm{Fe-O}}=1.55,\ d_{\min}^{\mathrm{Na-O}}=2.00,\ d_{\min}^{\mathrm{O-O}}=2.00,\ d_{\min}^{\mathrm{P-P}}=2.60,\ d_{\min}^{\mathrm{Na-Na}}=2.20\ \mathring{\mathrm A}\) | 其他元素对使用 \(d_{\min}(a,b)=0.75[r_{\mathrm{cov}}(a)+r_{\mathrm{cov}}(b)]\)。 |
| Appendix | A6：成键与配位截断 | \(r_{\mathrm{P-O}}=2.00\ \mathring{\mathrm A},\quad r_{\mathrm{Fe-O}}=2.50\ \mathring{\mathrm A}\) | \(\mathcal N_i^B:=\{j\ne i:a_j\in B,\ d_{ij}^{\mathrm{PBC}}\le r_{a_i-a_j}\}\)。 |
| Appendix | A7：去重口径 | \(\mathcal C\simeq\mathcal C_k\) | 使用固定容差的 StructureMatcher 或等价结构指纹，并先统一规约晶胞。 |
| Appendix | A8：全 UMA 凸包定义 | \(\displaystyle E_{\mathrm{hull}}^{\mathrm{UMA}}(\mathcal C):=\max\!\left\{0,E_{\mathrm{UMA}}(\mathcal C)-E_{\mathrm{ref,hull}}^{\mathrm{UMA}}(\operatorname{comp}(\mathcal C))\right\}\) | 候选与全部竞争相使用相同 UMA checkpoint、任务头、每原子归一化及结构优化口径；不使用 MP 能量查表值或 MP2020 候选能量修正。 |
| Appendix | A9：原弛豫任务的统一化 | \(\displaystyle F_{\max}\le0.01\ \mathrm{eV}\,\mathring{\mathrm A}^{-1},\quad \lVert\boldsymbol\sigma_{\mathrm{UMA}}\rVert_{\mathrm F}\le0.10\ \mathrm{GPa}\) | 不再单列最小化 \(E_{\mathrm{UMA}}\)：固定组成时 \(E_{\mathrm{hull}}^{\mathrm{UMA}}\) 与 \(E_{\mathrm{UMA}}\) 对 \(\mathbf X,\mathbf L\) 的梯度相同。原子力和新增应力条件直接定义统一问题的结构收敛可行域；FIRE、BFGS、IPOPT 仅是候选求解器。 |
| Appendix | A10：统一求解空间 | \(\displaystyle (N,\mathbf A,\mathbf X,\mathbf L)=\Phi_{\theta_0}(N,\mathbf z_A,\mathbf z_X,\mathbf z_L)\) | 优化变量可在晶体空间或 Flow 源空间参数化，但必须对应同一个统一目标与可行域，不再设置独立的生成后弛豫阶段。 |
| Appendix | A11：新增多面体几何约束 | 见下方陈列公式 | 软距离/邻居用于梯度优化；Fe–Fe 下界仅作为由 Fe–O 距离和角度约束导出的可选 barrier，不再单列独立硬约束；最终使用精确 PBC 距离与配位验收。 |

### A11 陈列公式

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

\[
\begin{aligned}
S_{ik}
  &=\sum_{j:a_j=\mathrm O}s_{ij}^{(\mathrm{Fe})}s_{kj}^{(\mathrm{Fe})}\le2,\\
\left|\mathbf n_i\!\cdot\!\mathbf n_k\right|
  &\le\cos\theta_{\min},
\qquad a_i=a_k=\mathrm{Fe},\ i<k.
\end{aligned}
\]

[^constants]: 取 \(M_{\mathrm{Na}}=22.99\)、\(M_{\mathrm{Fe}}=55.85\)、\(M_{\mathrm P}=30.97\)、\(M_{\mathrm O}=16.00\ \mathrm{g\,mol^{-1}}\)，\(F=96485.33212\ \mathrm{C\,mol^{-1}}\)，\(\gamma_{\mathrm{mAh}}=3.6\ \mathrm{C\,mAh^{-1}}\)；\(C_{\min}=130\ \mathrm{mAh\,g^{-1}}\)、\(E_{\mathrm{hull,max}}^{\mathrm{UMA}}=0.150\ \mathrm{eV\,atom^{-1}}\)、\(F_{\mathrm{tol}}=0.01\ \mathrm{eV}\,\mathring{\mathrm A}^{-1}\) 和 \(\sigma_{\mathrm{tol}}=0.10\ \mathrm{GPa}\approx6.242\times10^{-4}\ \mathrm{eV}\,\mathring{\mathrm A}^{-3}\) 均为设计筛选阈值；应力换算采用 \(1\ \mathrm{eV}\,\mathring{\mathrm A}^{-3}=160.21766208\ \mathrm{GPa}\)。

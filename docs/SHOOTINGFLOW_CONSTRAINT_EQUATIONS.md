# ShootingFlow 约束实现索引

> 本文件不再独立定义约束、阈值或符号。统一优化问题的唯一规范来源是
> [`OPTIMIZATION.md`](./OPTIMIZATION.md)；所有约束及其物理口径只在该文件维护。

## 梯度优化接口

ShootingFlow 阶段只对初始噪声 (z) 求梯度，Flow 参数冻结。实现时将
[`OPTIMIZATION.md`](./OPTIMIZATION.md) 的 A5、A6、A9 和 A10 转换为连续惩罚项：

- A5：PBC 最小距离使用周期 soft-min 距离，并用 softplus barrier 逼近下界；
- A6：P–O 与 Fe–O 配位使用软邻居权重，作为整数配位的可微代理；
- A9：UMA 原子力与晶格应力使用连续 barrier，直接约束结构收敛；
- A10：使用多面体中心、Fe 多面体共享面/共面和 Fe–Fe 距离三项惩罚。

三类代理项直接按 `OPTIMIZATION.md` A10 的定义组成
\(\mathcal L_{\mathrm{constraint}}\)，并代入该文档顶部的统一标量目标；本文件不重复目标函数定义。

软代理仅用于梯度搜索；最终验收仍严格执行 `OPTIMIZATION.md` 中 A5/A6
的精确 PBC 距离、整数配位、力/应力及 A10 几何条件。若实现需要展开式，直接以
`OPTIMIZATION.md` 的 A10 为准，不在本文件复制定义。

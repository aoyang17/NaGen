# 固定 N/A 新候选生成实验

本轮已获用户授权重新训练 NaGen 生成器。CHGNet 是能量/力 surrogate，
不训练另一个能量回归器，不继续训练集批量弛豫。

## 冻结的实验规格

- 原始数据经过母结构分组与跨 split 匹配排除，使用2950 train / 620 val / 632 test。
- `outputs/conditional_data_v2/geometry.pt` 只含原始几何与固定元素信息，没有能量标签。
  Niggli 约化仅改变晶胞基矢，保持原子数、组成、体积和物理结构；没有弛豫。
- 使用现有周期 attention 向量场（hidden128，4层，4头），固定 N 和原子通道 A。
  原子通道幅度2.0写入checkpoint，训练和采样保持一致，midpoint子步同样固定。
- 新入口 `selection/train_generator.py` 只回归分数坐标与标准化晶胞的速度。
  使用species-wise assignment耦合均匀源点云；条件无序点云源分布不变，不声称
  经依赖target的排列后各有序坐标仍相互独立。
- 不使用energy、force、motif、composition或novelty训练惩罚；旧加权辅助项入口不调用。
- FP32训练适配V100。最多100 epoch，patience20，验证集固定随机腐蚀，选择EMA最佳
  验证checkpoint；不根据测试集结果挑选模型。训练seed17，当前只训练一个模型。

## 新生成候选对照

从训练计数选择三个电中性、N≤100、支持数最多的组成，未查看能量来选择：
Na12Fe12P16O60（78条）、Na10Fe12P16O60（72条）、Na8Fe12P16O60（72条）。
它们是组成条件，不是从训练结构出发；源坐标为独立均匀噪声，晶胞源为标准正态。
family当前为通用phosphate策略，不声称训练了独立family embedding。

3条件 × 3源种子（17/29/43）× 8配对起点，midpoint 24步：

- guided：每起点最多8步source-space Adam，唯一优化目标为CHGNet E/atom，lr0.002。
- random：同一起点加8个独立随机源，作为相同查询预算上限的控制。
- 每臂最多648次候选能量查询；梯度验证调用单独计数。实际调用数、墙钟和失败均记录。
  严重无效结构可在势函数之前拒绝，早停的引导不补成虚构的满预算运行。
  查询上限相同不等于GPU时间相同；轨迹端点也不是独立样本。
- 所有已生成端点和失败记录保留；不只保留优化成功配对，不从训练结构加扰动冒充生成。
- 不做terminal relaxation，尤其不再弛豫训练集。

最终只在通过冻结hard gates的候选内做minimax，同行比较energy排序。
每组k8、质量池32、描述符最小距离0.02，再对最终结构做训练集与入选集的
StructureMatcher复核。通过者导出CIF并回读验证；0个通过也是有效实验结果，不能补入失败结构。
CHGNet E/atom不是完整E_hull，本轮不作热力学稳定性声明。

## 作业与文件

- GPU接口冒烟：2333366，2epoch、每epoch3batch，COMPLETED；不是生成质量验证。
- 正式训练：2333367，`outputs/conditional_train_2333367/`。
- 新候选数组：2333371，afterok依赖训练；`outputs/new_candidates_2333371/`。
- 选择与CIF导出：2333372，afterok依赖全部新候选分片。
- 自动取回工具：`tools/fetch_new_candidates.py`，仅成功完成后校验并下载结果。
  目标Desktop目录为`NaGen_new_candidates_2333371`；本机进程需保持运行。

CPU小模型冒烟已验证训练反传、保留失败起点、CHGNet调用及空选择结果处理。
这些小模型的候选未通过hard gates，不计作正式模型结果。
训练loss下降不证明物理可行性，正式结果以新候选拒绝原因、通过率和选出数量为准。

## 首轮已完成结果（2026-09-08）

训练、新候选9分片和选择作业全部COMPLETED、退出0:0，结果已取回。
这只是作业执行成功，**材料生成实验结果为0个通过**。

训练23个epoch后触发patience20，最佳epoch索引2（第3轮），EMA验证loss3.12629；
末轮训练loss0.430915、验证loss5.61819。验证不再改善，不能宣称模型已学会可行材料分布。
869277参数，训练阶段墙钟405.30秒（Slurm分配7分02秒）；checkpoint约13MB，保存于
`outputs/new_candidates_2333371/conditional_train_2333367/best.pt`。

| 指标 | CHGNet引导 | 随机生成对照 |
|---|---:|---:|
| 配对起点数 | 72 | 72 |
| 查询预算上限 | 648 | 648 |
| 实际生成并记录的端点 | 127 | 648 |
| 实际候选势函数查询 | 57 | 216 |
| 势函数前严重重叠拒绝 | 70 | 432 |
| 通过所有hard gates | 0 | 0 |

70/72条引导轨迹因<0.4 Å原子距离中止；仅2条完成全部8步。
两条完整轨迹的能量分别从0.786625降至-3.493947、从0.609319降至-3.493422 eV/atom，
但结构仍不通过物理门槛。这些不合理结构上的ML势能量不能解释为真实稳定性收益。
总候选势函数调用273次；9个作业各自的13次梯度检查另计，共117次。
有严重提前失败，不能宣称实际GPU时间或实际势函数调用次数已相等。

273个被评价端点全部不通过overlap、motif和final_force门槛。
最大原子力的最小/中位/最大值约75.80/208.93/481.22 eV/Å，远高于0.05。
4368个被评价P位点中只有212个CN=4（约4.85%；四配位也不等于通过完整PO4几何检查）。
因此零通过不是仅仅力阈值稍严，而是生成器未形成基本局部结构。

没有可行候选进入ranking，因此这轮**不能比较minimax或diversity策略的收益**，
也不能把生成器失败归结为四层筛选设计本身无效。下一阶段应优先改善/诊断生成器，
在小预算验证其PO4局部结构成功率后再扩大DFlow，而不是放宽门槛或重启训练集弛豫。

完整结果：`outputs/new_candidates_2333371/selection/summary.json`，各条件目录保存
全部候选、力/能量、逐项gate与失败ledger。
Desktop的`NaGen_new_candidates_2333371`仅含README和空manifest，**没有合格CIF**；
之前`NaGen_feasible_CIF_2333337`中的3个已有结构CIF仍保留，未与本轮新生成结果混淆。

## 自动续跑：原型空位生成支路

由于从零训练的flow未生成基本PO4局部结构，后续采用训练集原型条件生成支路，
不是继续盲目增加epoch，也不把它冒充flow结果。输入仅来自train split中的
`mix_Na4Fe3P4O15_8`母结构：从唯一Na16Fe12P16O60全钠原型的16个Na位点中移除4个，
共有1820种离散组合，seed17打乱后取64个候选。没有读取val/test结构或旧能量标签。

作业2333391在达到目标后停止，远端完成45个；本地冻结检查点包含前32个完整候选。
32/32经CHGNet固定晶胞FIRE收敛并通过全部hard gates。使用train同组成78条结构标定
径向重复阈值：非零最近邻距离min/p05/p25/median为2.99e-6/3.88e-5/2.21e-4/2.42e-3，
故冻结阈值1e-4；此前0.02明显会误杀正常训练变体，已弃用。

在32个可行结构上进行percentile/minimax排序，取质量池32、k=8，随后径向farthest-first
和StructureMatcher对训练集/已选集合复核，最终选出6个新候选。最大力为
0.02787–0.02999 eV/Å，均低于0.05。相对4213条CHGNet单点有限参考集的能量差为
0.00429–0.01062 eV/atom，均低于宽松0.15阈值；但参考结构未在同一CHGNet标准下弛豫，
且不含完整竞争相，所以不可称为真实热力学E_hull。

冻结结果与CIF：`outputs/vacancy_2333391_checkpoint/selection/`；Desktop副本：
`/Users/yangaobo/Desktop/NaGen_selected_vacancy_2333391/`。全部CIF已回读并核对SHA256。
这证明四层流程能够从该原型条件生成支路选出候选；不证明从零flow已经work，
也不证明候选达到DFT稳定性。下一物理验证应是统一DFT弛豫与完整相图。

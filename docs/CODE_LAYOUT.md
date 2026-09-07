# NaGen 当前代码边界

2026-09-07 起，默认维护和复现实验主线是最近两次
`surrogate_shootingflow_*_12x140_v1`：

```text
冻结 Flow checkpoint
        │
        ▼
  N/A 组成固定 ──► z_X, z_L source-space shooting
        │                       │
        │                       ├─ 几何软约束
        │                       ├─ 新训练 PeriodicMACE E_hull surrogate
        │                       └─ novelty 辅助项
        ▼
  inverse.constraints 精确验收 + JSON/CIF 导出
```

## 主线模块

- `nagen.surrogate.ehull_mace`：新训练的周期图 E_hull surrogate、模型结构、训练/加载和
  differentiable graph batch。模型权重不入仓库，由
  `configs/surrogate_shootingflow_v1.json` 指向耐久存储。
- `nagen.inverse.model`、`sample`、`lattice`、`generate`：Flow 模型、ODE 积分、晶格解码和
  `N/A` 组成采样。
- `nagen.inverse.multiobjective_shooting`：ShootingFlow 框架、MGDA、source-space 优化和
  reintegration 审计。
- `nagen.inverse.spec`、`constraints`：`Na–Fe–P–O` 优化问题组成、硬约束和最终可行性门。
- `nagen.inverse.dataset`、`novelty`：packed 数据和 novelty reference 支持。
- `tools/run_surrogate_shootingflow.py`：当前 surrogate shooting 实验入口；参数和 checkpoint
  见上面的 config。

## 明确不属于当前主线

`generation/` 早期 RDF surrogate/Dflow 入口、`inverse/` 下的 UMA/CHGNet 松弛、MP2020 hull
校准、relax-aware campaign 和旧 pilot/report 脚本保留用于历史结果审计或兼容测试，但不应被
新实验入口导入，也不应与当前 surrogate 的 `E_hull` 结果混合汇总。

当前结果是 surrogate screening，不代表完成 UMA 松弛或真实 E_hull 验证；精确约束失败仍必须
在最终 JSON 中保留，不能用 surrogate 阈值覆盖。

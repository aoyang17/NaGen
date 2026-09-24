# ShootingCSP 约束配置

ShootingCSP 使用 `shootingcsp.constraints.v1` JSON 作为生成优化问题的唯一物理约束来源。

默认配置：

```text
configs/constraints/na_fe_po_default.json
```

JSON Schema：

```text
configs/constraints/shootingcsp.constraints.schema.json
```

## 离线约束编辑器

打开：

```text
docs/shootingflow_constraint_builder.html
```

该页面完全离线运行，不依赖 CDN、MathJax 或网络字体。用户可以：

- 修改任意约束的数值；
- 启用或关闭单个约束；
- 导入已有 JSON；
- 导出 `shootingcsp.constraints.json`；
- 复制对应的主生成命令。

## 校验配置

```bash
python -m shootingcsp.config \
  --constraints shootingcsp.constraints.json \
  --out resolved.json
```

命令会输出配置名称、schema 版本和 SHA256。

## 运行生成

```bash
shootingcsp-generate \
  --constraints shootingcsp.constraints.json \
  --flow /path/to/flow.pt \
  --profile /path/to/profile.json \
  --surrogate uma \
  --surrogate-checkpoint /path/to/uma.pt \
  --out /path/to/output
```

输出目录会保存：

```text
constraints.input.json
constraints.resolved.json
constraints.sha256
manifest.json
generated_all.jsonl
generated_safe.jsonl
```

## Python API

```python
from shootingcsp.config import (
    load_constraint_config,
    to_optimization_spec,
    optimization_weights,
)

config = load_constraint_config("shootingcsp.constraints.json")
spec = to_optimization_spec(config)
weights = optimization_weights(config)
```

约束值会映射到：

- source-space ShootingFlow 软目标；
- 生成前安全预检查；
- 精确 hard gates；
- 有限参考 hull；
- 新颖性与选择规则；
- 固定晶胞弛豫参数。

当约束变化后，`manifest.json` 中的 `constraint_config_sha256` 会变化，campaign 续跑时可以据此拒绝混用不同物理问题。

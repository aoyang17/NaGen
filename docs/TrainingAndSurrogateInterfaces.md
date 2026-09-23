# Flow 训练、数据集与 surrogate 接口

本文档描述当前 `src/` 中新增的扩展接口。它不改变已有 UMA campaign 的默认运行方式。

## 1. 重新训练或微调 Flow

内置入口：

```bash
python -m shootingcsp.training.train_flow --help
```

Packed ShootingCSP 数据集：

```bash
python -m shootingcsp.training.train_flow \
  --dataset /path/to/packed.pt \
  --dataset-kind packed \
  --train-split train \
  --val-split val \
  --out /path/to/flow.pt \
  --coupling polyhedral \
  --geometry-only \
  --geometry-validity-weight 0.0
```

JSONL 数据集：

```bash
python -m shootingcsp.training.train_flow \
  --dataset /path/to/dataset.jsonl \
  --dataset-kind jsonl \
  --train-split train \
  --val-split val \
  --out /path/to/flow.pt
```

JSONL 每条记录支持：

```json
{
  "material_id": "id",
  "split": "train",
  "A": ["Na", "Fe", "P", "O"],
  "X_frac": [[0.0, 0.0, 0.0]],
  "L_matrix": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
}
```

也可以使用 pymatgen 字典：

```json
{"material_id": "id", "split": "train", "structure": {}}
```

训练支持：

- `--resume` / `--init`
- EMA
- `--geometry-only`
- `--coupling noise|assignment|polyhedral`
- `--fe-coordination 4,5`
- 单卡或 `torchrun` 分布式训练
- JSONL checkpoint 日志

## 2. 自定义数据集适配器

适配器需要返回如下对象：

```python
class Dataset:
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> dict: ...
    @property
    def atom_counts(self) -> tuple[int, ...]: ...
    @property
    def element_to_index(self) -> dict[str, int]: ...
    @property
    def metadata(self) -> dict: ...
```

每个 item 必须包含：

```python
{
    "index": int,
    "types": torch.Tensor,       # canonical element indices
    "frac_coords": torch.Tensor, # (N, 3)
    "lattice": torch.Tensor,     # (3, 3)
    "N": int,
}
```

注册代码示例：

```python
from shootingcsp.training.data import register_dataset

def load_my_dataset(path, *, split, **options):
    return MyDataset(path, split=split, **options)

register_dataset("my-format", load_my_dataset)
```

也可以通过外部模块直接加载：

```bash
python -m shootingcsp.training.train_flow \
  --dataset /path/to/data \
  --dataset-factory my_package.datasets:load_my_dataset \
  --dataset-option key=value \
  --out /path/to/flow.pt
```

## 3. Surrogate 接口

生成时使用的 surrogate adapter 需要实现：

```python
class Surrogate:
    @property
    def provenance(self) -> dict: ...

    def guidance_energy(self, atomic_numbers: torch.Tensor):
        """Return energy_fn(frac, lattice) with shape (1, N, 3), (1, 3, 3)."""
        ...

    def evaluator(self, inference_settings: str = "default"):
        """Return an ASE-calculator-compatible evaluation backend when needed."""
        ...
```

当前内置 UMA：

```text
backend = "uma"
```

注册自定义 backend：

```python
from shootingcsp.surrogate import register_surrogate

def load_my_surrogate(spec):
    return MySurrogate(spec)

register_surrogate("my-surrogate", load_my_surrogate)
```

单批生成时选择 backend：

```bash
python tools/generate_uma_guided_framework_candidates.py \
  --flow runs/nacathode_broad_flow.best.pt \
  --condition configs/framework_condition_orthophosphate.json \
  --profile runs/nafe_po_profile.json \
  --surrogate my-surrogate \
  --surrogate-checkpoint /path/to/model \
  --surrogate-factory my_package.surrogate:load_my_surrogate \
  --surrogate-option key=value \
  --out /path/to/output
```

UMA 命令保持向后兼容：

```bash
python tools/generate_uma_guided_framework_candidates.py \
  --flow runs/nacathode_broad_flow.best.pt \
  --condition configs/framework_condition_orthophosphate.json \
  --profile runs/nafe_po_profile.json \
  --uma-checkpoint /mnt/data2/aobo/UMA/uma-m-1p1.pt \
  --out /path/to/output
```

## 4. src 中的公共入口

```text
shootingcsp.training.train_flow.main
shootingcsp.training.data.load_crystal_dataset
shootingcsp.training.data.register_dataset
shootingcsp.generation.optimize_source_with_surrogate
shootingcsp.surrogate.load_surrogate
shootingcsp.surrogate.register_surrogate
```

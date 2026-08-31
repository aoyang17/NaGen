# DFLOW / ShootingFlow Strategy Notes

This document collects the core algorithms used in FlowDesign / Dflow-SUR for
inference-time, physics-guided generative design. It is intended as a quick
reference for experiments that need to call DFlow, ShootingFlow, MultipleShooting,
or solver-guided variants.

## 1. Background and goal

The main idea is:

> Do **not** retrain a conditional generative model every time the physical
> objective changes. Instead, train an unconditional flow matching model once,
> then, during inference, differentiate through the learned flow and through a
> differentiable surrogate/physics evaluator to optimize the initial noise
> `x0`.

This is D-Flow applied to engineering design:

```text
x0 ~ N(0, I)
x1 = Phi_[0,1](x0)          # learned flow map
loss = || SUR(x1) - y ||^2  # physics/surrogate objective
x0 <- x0 - lr * d(loss)/d(x0)
```

The key advantages are:

- physics is injected only at the final generated design `x1`;
- flow-matching inference and physical-loss optimization are decoupled;
- no gradient collision between the velocity field and the physical loss;
- the generated sample stays close to the learned data manifold;
- surrogate uncertainty is lower than evaluating noisy intermediate states.

## 2. Algorithm overview

### 2.1 DFlow core algorithm

Given a learned velocity field `v_theta(x, t)` and a differentiable evaluator
`SUR(x)`:

1. Sample initial source:

   ```text
   x0 ~ N(0, I)
   ```

2. Solve the flow ODE forward:

   ```text
   x1 = ODESolve(v_theta, x0, t=0 -> 1)
   ```

3. Compute the final design loss:

   ```text
   L(x1) = || SUR(x1) - y ||^2
   ```

   For airfoil/wing experiments this is often:

   ```text
   L = (CL - CL_target)^2 + beta * max(CD - CD_max, 0)^2
   ```

4. Back-propagate to the source:

   ```text
   grad_x0 = dL/dx1 @ dx1/dx0
   ```

5. Update the source with an optimizer:

   ```text
   x0 <- OptimizerStep(x0, grad_x0)
   ```

6. Repeat for `optim_steps`/`guidance_steps` iterations. Diversity comes from
   different random `x0` initializations.

### 2.2 ShootingFlow

ShootingFlow is the same source-space optimization idea, but expressed as a
problem-level strategy:

```text
source = initial samples
for step in range(guidance_steps):
    terminal = flow_map(source, t0 -> t1)
    outputs = problem.evaluate(terminal)
    loss = problem.loss(terminal, outputs).mean()
    loss.backward()
    optimizer.step()
samples = flow_map(source.detach(), t0 -> t1)
```

If no flow/vector-field model is supplied, `flow_map` becomes the identity and
ShootingFlow reduces to direct gradient-based design-variable optimization.

### 2.3 MultipleShootingFlow

Multiple shooting splits `[t0, t1]` into several segments. The optimization
variables are all segment node states `z_0 ... z_M`:

```text
terminal_loss = problem.loss(z_M)
defect_loss = mean_i || z_{i+1} - flow_map(z_i, t_i -> t_{i+1}) ||^2
loss = terminal_loss + defect_weight * defect_loss
```

This is useful when long flow integration causes unstable or vanishing
gradients.

### 2.4 SolverGuidedDFlow

For non-autograd evaluators such as XFoil, gradients are estimated by central
finite differences:

```text
grad[:, j] = (loss(x + eps e_j) - loss(x - eps e_j)) / (2 eps)
x <- x - lr * grad
```

### 2.5 PostOpt

PostOpt first samples initial designs from the flow model without guidance,
then performs local Adam optimization in the design-variable space for each
sample.

## 3. Main implementation links

All links point to the `main` branch of the public FlowDesign repository:
`https://github.com/aoyang17/FlowDesign`.

### Core DFlow

- [`flowdesign/strategy/dflow.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/strategy/dflow.py)
  - `ode_integrate()` at line 11
  - `get_init_x()` at line 47
  - `dflow()` at line 99

The low-level `dflow()` function is the closest implementation of the DFlow
algorithm. It uses `torchdiffeq.odeint`, evaluates `H(x1)`, computes a cost
function, and optimizes `x0` with LBFGS/SGD/Adam.

### Strategy layer

- [`flowdesign/strategy/generative.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/strategy/generative.py)
  - `DFlowStrategy` at line 12
  - `PostOptProblemStrategy`
  - `BanditDFlowStrategy`
  - `ConditionalStrategy`

- [`flowdesign/strategy/shooting.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/strategy/shooting.py)
  - `ShootingFlowStrategy` at line 15
  - `flow_map()` at line 149
  - `flow_trajectory()` at line 181

- [`flowdesign/strategy/multiple_shooting.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/strategy/multiple_shooting.py)
  - `MultipleShootingFlowStrategy` at line 23
  - `_defect_loss()` at line 209

- [`flowdesign/strategy/solver_guided.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/strategy/solver_guided.py)
  - `SolverGuidedDFlowStrategy` at line 12

- [`flowdesign/strategy/postopt.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/strategy/postopt.py)
  - `PostOptStrategy` at line 11

### Problem abstraction

- [`flowdesign/core/problem.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/core/problem.py)
  - `DesignProblem` at line 24
  - `evaluate()` at line 33
  - `loss()` at line 37
  - `is_feasible()` at line 56

The central abstraction is:

```text
Problem  = geometry + evaluator + objectives + constraints
Strategy = generation / guidance / optimization behavior
Workflow = train / generate / optimize / evaluate
```

## 4. Flow model and surrogate models

### Flow matching training

- [`flowdesign/trainer.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/trainer.py)
  - `FlowMatchingTrainer` at line 8

- [`flowdesign/vf_model/mlp.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/vf_model/mlp.py)
  - `MLP` at line 14

- [`flowdesign/vf_model/resnet_mlp.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/vf_model/resnet_mlp.py)
  - `ResNetMLP` at line 26

The training objective is a standard flow matching regression:

```text
x_t = (1 - t) x0 + t x1
target_velocity = x1 - x0
loss = || v_theta(x_t, t) - target_velocity ||^2
```

### Airfoil surrogates

- [`flowdesign/surrogate/cuneuralfoil_surrogate.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/surrogate/cuneuralfoil_surrogate.py)
  - `cuNeuralFoilSUR` at line 65
  - outputs `[CL, CD, CM]`

- [`flowdesign/surrogate/airfoilSUR.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/surrogate/airfoilSUR.py)
  - `AirfoilSUR` at line 174
  - includes custom autograd finite-difference sensitivities

### Wing surrogate

- [`flowdesign/surrogate/aero_transformer_surrogate.py`](https://github.com/aoyang17/FlowDesign/blob/main/flowdesign/surrogate/aero_transformer_surrogate.py)
  - `AeroTransformerSurrogate` at line 210
  - `predict_fields_and_coefficients()` at line 448

The AeroTransformer path is:

```text
SuperWing design vector
-> differentiable mesh builder
-> ATsurf_S predicts [Cp, Cf_tau, Cf_z]
-> differentiable surface integration
-> [CL, CD, CM]
```

## 5. Recommended call patterns

### Low-level DFlow call

```python
from flowdesign.strategy.dflow import dflow

x1, trajectory, metrics = dflow(
    ode_func=ode_func,          # ode_func(t, x) -> dx/dt
    y=target,                   # target physics values
    H=H,                        # x1 -> observed/compared quantities
    cost_func=cost_func,        # (H(x1), y) -> scalar loss
    x_dims=(batch, design_dim),
    init_method="random",
    optimizer_type="LBFGS",
    lr=1.0,
    ode_opts={"method": "midpoint", "options": {"step_size": 0.1}},
    optim_steps=50,
    max_iter=20,
)
```

Use this pattern when you already have a trained flow model and a custom
physics evaluator.

### Problem + Strategy call

```python
from flowdesign.core import DesignProblem
from flowdesign.geometry import CSTAirfoilGeometry
from flowdesign.surrogate import DummyAirfoilEvaluator
from flowdesign.loss import EnergyLoss
from flowdesign.constraints import CLConstraint, CDConstraint
from flowdesign.strategy import DFlowStrategy

problem = DesignProblem(
    geometry=CSTAirfoilGeometry(n_upper=8, n_lower=8),
    evaluator=DummyAirfoilEvaluator(alpha=2.0, mach=0.15),
    objectives=[EnergyLoss(target="CD", mode="minimize")],
    constraints=[
        CLConstraint(target=0.5, tolerance=0.02, weight=20.0),
        CDConstraint(max_value=0.015, weight=10.0),
    ],
)

result = DFlowStrategy(
    num_samples=256,
    guidance_steps=100,
    guidance_lr=0.01,
).run(problem)
```

If a flow model path is supplied, `DFlowStrategy.run()` internally routes to
`ShootingFlowStrategy`:

```python
result = DFlowStrategy(
    num_samples=64,
    guidance_steps=50,
    flow_model_path="results_dflow_airfoil/airfoil_flow_model.pt",
    flow_input_dim=16,
    flow_hidden_dim=512,
    flow_steps=50,
    integrator="rk4",
).run(problem)
```

### Multiple shooting call

```python
from flowdesign.strategy.multiple_shooting import MultipleShootingFlowStrategy

result = MultipleShootingFlowStrategy(
    num_samples=64,
    guidance_steps=50,
    segments=5,
    flow_steps_per_segment=10,
    defect_weight=100.0,
    flow_model_path="results_dflow_airfoil/airfoil_flow_model.pt",
    flow_input_dim=16,
).run(problem)
```

## 6. End-to-end aircraft experiment

The main Dflow-SUR experiment is in:

- [`main_dflow_aerotransformer.py`](https://github.com/aoyang17/FlowDesign/blob/main/main_dflow_aerotransformer.py)
  - `aero_cl_cd_cost()` at line 79
  - `load_or_train_flow_model()` at line 444
  - `main()` at line 586
  - surrogate `H(x)` at line 639
  - DFlow call at line 660

This script trains or loads a SuperWing flow model, builds an AeroTransformer
surrogate, and optimizes source noise with DFlow against `CL` and `CD`
targets.

## 7. Useful configs

- [`configs/airfoil/dflow.yaml`](https://github.com/aoyang17/FlowDesign/blob/main/configs/airfoil/dflow.yaml)
- [`configs/airfoil/dflow_pretrained.yaml`](https://github.com/aoyang17/FlowDesign/blob/main/configs/airfoil/dflow_pretrained.yaml)
- [`configs/airfoil/multiple_shooting_pretrained.yaml`](https://github.com/aoyang17/FlowDesign/blob/main/configs/airfoil/multiple_shooting_pretrained.yaml)
- [`configs/airfoil/xfoil_dflow.yaml`](https://github.com/aoyang17/FlowDesign/blob/main/configs/airfoil/xfoil_dflow.yaml)
- [`configs/superwing/dflow.yaml`](https://github.com/aoyang17/FlowDesign/blob/main/configs/superwing/dflow.yaml)
- [`configs/superwing/multiple_shooting.yaml`](https://github.com/aoyang17/FlowDesign/blob/main/configs/superwing/multiple_shooting.yaml)

## 8. References

- D-Flow:
  - Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, Yaron Lipman.
  - “D-Flow: Differentiating through Flows for Controlled Generation.”
  - ICML 2024.
  - arXiv: [https://arxiv.org/abs/2402.14017](https://arxiv.org/abs/2402.14017)
  - PMLR: [https://proceedings.mlr.press/v235/ben-hamu24a.html](https://proceedings.mlr.press/v235/ben-hamu24a.html)

- Dflow-SUR / FlowDesign paper:
  - Aobo Yang, Zhen Wei, Rhea P. Liem, Pascal Fua.
  - “Dflow-SUR: Enhancing Generative Aerodynamic Inverse Design using Differentiation Throughout Flow Matching.”
  - arXiv: [https://arxiv.org/abs/2512.08336](https://arxiv.org/abs/2512.08336)
  - ar5iv HTML: [https://ar5iv.labs.arxiv.org/html/2512.08336](https://ar5iv.labs.arxiv.org/html/2512.08336)
  - Journal version: “Physics-guided generative design with gradient-based source-space optimization on flow matching.”
  - Springer DOI: [https://doi.org/10.1007/s00158-026-04318-6](https://doi.org/10.1007/s00158-026-04318-6)

- FlowDesign repository:
  - [https://github.com/aoyang17/FlowDesign](https://github.com/aoyang17/FlowDesign)

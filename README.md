# NaGen

NaGen generates constrained Na–Fe–P–O orthophosphate crystal candidates.
This repository's production workflow is the UMA campaign that produced the
September 10–11, 2026 candidate set. Earlier root-level implementations are
preserved under `legacy/` and are not part of the active runtime path.

## Crystal representation

I adopt the $(N, \mathbf{A}, \mathbf{X}, \mathbf{L})$ crystal representation strategy, where:

- $N \in \mathbb{Z}_{>0}$ is the total number of atoms in the periodic cell;
- $\mathbf{A} = (a_i)_{i=1}^{N}$ specifies the element at each atomic site;
- $\mathbf{X} = (x_i)_{i=1}^{N}$, with $x_i \in [0,1)^3$, specifies the atomic positions within the cell, and is also the atomic-position variable updated by the original relaxation task;
- $\mathbf{L} \in \mathbb{R}^{3\times3}$ specifies the cell size and shape, and is the variable updated by the original relaxation task through the cell filter.

These four components together form the design variables of the generative optimization problem.

## Production architecture

1. Fix the `Na6Fe6P8O32` composition and draw a polyhedral source state.
2. Optimize source coordinates and lattice for 12 steps with UMA energy and
   differentiable geometric constraints.
3. Integrate the frozen geometry DFlow for 48 midpoint steps.
4. Relax at fixed cell with UMA/FIRE, then validate promising candidates using
   default UMA inference.
5. Require exact coordination, geometry, force, finite same-UMA reference-hull,
   framework novelty, cross-candidate uniqueness, and CIF round-trip checks.

The finite reference hull is a same-UMA proxy, not a complete DFT phase diagram.


# ShootingFlow · Optimization problem formulation | Take Na-Fe-P-O generation for example

| Type | Item | Requirement / Definition | Current Result / Notes |
|---|---|---|---|
| ![Objective](https://img.shields.io/badge/-Objective-2ea44f?style=flat-square) | Stability: minimize $E_{\mathrm{hull}}$ | $\le 150\ \text{meV/atom}$ | ✓ Result range: 122–168 meV/atom; 50% compliant |
| ![Objective](https://img.shields.io/badge/-Objective-2ea44f?style=flat-square) | Novelty: maximize $f_{\mathrm{nov}}$ | Fe–P–O framework after alkali-metal removal | ✓ No match to known structures |
| ![Design Variable](https://img.shields.io/badge/-Design_Variable-8250df?style=flat-square) | Number of atoms | $N\in\mathbb Z_{>0}$ | Total number of atoms in the periodic cell. |
| ![Design Variable](https://img.shields.io/badge/-Design_Variable-8250df?style=flat-square) | Element sequence | $\mathbf A=(a_i)_{i=1}^{N}$ | Specifies the element at each atomic site. |
| ![Design Variable](https://img.shields.io/badge/-Design_Variable-8250df?style=flat-square) | Fractional coordinates | $\mathbf X=(\mathbf x_i)_{i=1}^{N},\ \mathbf x_i\in[0,1)^3$ | Specifies atomic positions within the cell; also the atomic-position variable updated by the original relaxation task. |
| ![Design Variable](https://img.shields.io/badge/-Design_Variable-8250df?style=flat-square) | Lattice matrix | $\mathbf L\in\mathbb R^{3\times3}$ | Specifies cell size and shape; also the variable updated by the original relaxation task through the cell filter. |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Elemental composition | Only Na, Fe, P, O, and all four elements present | ✓ Na<sub>6</sub>Fe<sub>6</sub>P<sub>8</sub>O<sub>32</sub> |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Average formal valence of Fe | $\bar z_{\mathrm{Fe}} \ge 2$ | ✓ $\bar z_{\mathrm{Fe}} = 3.0$ |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Theoretical specific capacity | $C_{\mathrm{th}} \ge 130$ mAh g$^{-1}$ | ✓ 130.4 mAh g$^{-1}$ |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Volume | 10.5–20.5 Å$^3$/atom | ✓ 12.91–16.41 Å$^3$/atom |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Number of atoms in primitive cell | $23 \le N_{\mathrm{prim}} \le 184$ | ✓ Current cell $N = 52$ |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | PBC minimum distance | P–O $\ge 1.40$; Fe–O $\ge 1.55$; Na–O/O–O $\ge 2.00$; P–P $\ge 2.60$; Na–Na $\ge 2.20$ Å | ✓ No PBC overlaps; P–O 1.474–1.652 Å, Fe–O 1.791–2.428 Å |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | P–O coordination | $r_{\mathrm{P-O}} \le 2.00$ Å; CN(P) = 4 | ✓ All PO<sub>4</sub>; 8/8 P pass |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Fe–O coordination | $r_{\mathrm{Fe-O}} \le 2.50$ Å; CN(Fe) ∈ {4,5,6} | ✓ 6 Fe per cell; all FeO<sub>4</sub>/FeO<sub>5</sub>/FeO<sub>6</sub> |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | P/Fe polyhedron centers | P and Fe strictly located inside their respective O polyhedra | ✓ 14/14 P and Fe centers pass |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Fe polyhedron adjacency | Any Fe–Fe shared O count $\le 2$ | ✓ No two Fe–O coordination polyhedra share the same oxygen triangular face |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Oxygen coverage | All O covered by framework polyhedra | ✓ 32/32 O covered |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | UMA $E_{\mathrm{hull}}$ | $E_{\mathrm{hull}}^{\mathrm{UMA}} \le 150$ meV/atom | ✓ 100.00–146.21 meV/atom |
| ![Constraint](https://img.shields.io/badge/-Constraint-0969da?style=flat-square) | Atomic force convergence | $F_{\max} \le 0.03$ eV Å$^{-1}$ | ✓ 0.01874–0.02996 eV Å$^{-1}$ |

## Entrypoints

- `tools/run_uma_campaign.py`: resumable multi-worker generation, relaxation,
  gates, hull proxy, and incremental selection.
- `tools/generate_uma_guided_framework_candidates.py`: single-batch generation
  and UMA-guided source optimization.
- `tools/select_framework_candidates.py`: offline strict selection and ranking.
- `tools/audit_uma_campaign.py`: independent final-CIF validation.
- `tools/audit_crossmetal_novelty.py`: the September 11 cross-metal novelty audit.

`runs/` is a symlink to the immutable 9.10–9.11 experiment assets, including
the flow checkpoint, conditioning profile, reference labels, campaign records,
and selected structures.

## Run the campaign

```bash
tools/launch_uma_campaign.sh
```

Use `tools/launch_uma_campaign.sh --audit` to revalidate the campaign's
exported CIFs. See [UMACampaign24.md](docs/UMACampaign24.md) for the frozen
protocol and artifact layout.

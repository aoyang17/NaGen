# NaGen four-stage framework-generation experiment handoff

Last updated: 2026-09-09 06:57 CST  
Git branch: `experiment/framework-flow-four-stage`  
Base experiment commit before this handoff: `a81a5c69d1a1e1d4bfdcf2c541b06feb351b4e82`

This document is the authoritative handoff for the current Na–Fe–P–O experiment. It distinguishes verified facts, provisional results, and work that is still running. Do not infer final performance from the seven provisional CIFs.

## 1. Experimental objective

Test whether a conditional crystal Flow Matching model, followed by physically explicit feasibility and selection stages, can generate Na–Fe–P–O phosphate cathode structures whose **Fe–P–O framework is not equivalent to a known training structure**. Moving only Na sites, changing Na occupancy, relaxing a copied parent, or merely failing a continuous descriptor threshold does not establish framework generation.

The requested architecture is:

```text
Conditioning -> Hard Feasibility Gates -> Robust Ranking -> Diversity Selection
```

The current energy/force backend is CHGNet 0.4.2, used as a temporary open surrogate because UMA weights were unavailable. Historical energy fields in the JSONL dataset are ignored.

## 2. Main algorithm

### 2.1 Conditioning: specify what to generate

The condition fixes the atom count, species counts, stoichiometry, Na content, and material family. For the formal run:

```json
{"counts":{"Na":6,"Fe":6,"P":8,"O":32},"family":"phosphate"}
```

Thus `N=52` and the atom-type sequence `A` are fixed. Composition is not an optimization penalty. The generator produces all fractional coordinates `X` for Na/Fe/P/O and the lattice `L`.

The geometry Flow is trained with a `polyhedral` coupling. Its source is freshly randomized for every sample and supplies an inductive bias toward local PO4-like arrangements. It does **not** copy coordinates or a lattice from a training parent. The Flow integrates this random source to `t=1` while the conditioned atom types are frozen.

### 2.2 Hard feasibility gates: reject loss of physical meaning

The gate implementation is in `src/nagen/selection/pipeline.py`. A candidate must satisfy all applicable checks:

- exact conditioning/species counts and supported phosphate family;
- a finite, nondegenerate periodic cell;
- integer-compatible charge neutrality under the declared model Na+, P5+, O2-, Fe2+/Fe3+;
- no extreme periodic overlap or implausibly short bond;
- final maximum CHGNet residual force below the configured threshold;
- phosphate motif: every P has supported PO4 oxygen coordination, acceptable P–O lengths and angles, and P lies inside its oxygen polyhedron;
- Fe motif: Fe oxygen coordination numbers and geometry must lie in distributions calibrated from training structures, rather than assuming a single Fe coordination number;
- oxygen coverage and optional external-filter result, when configured.

The formal generation command first applies only a catastrophic precheck (`d_min >= 0.4 A`, finite cell, volume/atom and condition-number bounds) before calling CHGNet. Full gates are reevaluated after relaxation. Gate failure is never compensated by a good energy.

### 2.3 Robust ranking: compare only feasible candidates

Only hard-feasible candidates enter ranking. The continuous metrics are:

- CHGNet finite-reference `E_above_reference_hull` proxy;
- worst P and Fe off-centering;
- bond-length distortion;
- angle distortion;
- coordination rarity/quality.

Each metric is converted to an empirical percentile within the comparable group. `robust_rank` orders the sorted percentile vector lexicographically (bottleneck/minimax behavior), so an excellent energy cannot hide a severely distorted local environment. There is no weighted scalar sum of unrelated constraints.

Important terminology: the current value is a **finite-reference CHGNet hull proxy**, not a complete thermodynamic convex-hull result. It is valid only when candidate and reference energies use the identical CHGNet model/hash. The current loose selection threshold is 0.15 eV/atom.

### 2.4 Diversity selection: framework novelty after quality

Novelty is not rewarded in the generation loss. Selection removes Na before novelty analysis so that Na rearrangement cannot masquerade as a new framework. It then requires:

1. no Fe–P–O match to known structures under `pymatgen.StructureMatcher`;
2. no isomorphic Fe/P–O coordination graph match;
3. a minimum framework-descriptor distance to known structures;
4. no exact structural duplicates among selected candidates;
5. farthest-first selection relative to the training set and already selected candidates, after physical quality filtering.

The descriptor distance is a screening coordinate, not by itself proof of novelty. Exact structure matching and graph-topology checks are retained as independent audits.

## 3. Data provenance

Original dataset source:

- <https://github.com/aoyang17/NaGen/blob/main/parent_RDF_opt.jsonl>
- the `structure` field contains already optimized/steady structures;
- stored historical `energy` values are not used as labels in this experiment.

BSCC prepared dataset:

```text
/data02/home/scv7eyx/projects/NaGen/outputs/full_dataset_v3/
  train.jsonl   2,950 structures
  val.jsonl
  test.jsonl
  target.jsonl  4,213 structures total
```

The full geometry audit found P CN=4 for all 86,238 target P sites and data-supported Fe CN values 4, 5, and 6. Only the training split calibrates motif distributions.

The geometry Flow was trained on the orthophosphate subset selected by `O = 4P`:

```text
/data02/home/scv7eyx/projects/NaGen/outputs/orthophosphate_data/geometry.pt
count: 618
split: 496 train / 62 validation / 60 test
packed SHA256: 98ea89bd620f659552b7c935e4a30495d83058db67632ca55aa020013336110f
contains energy labels: false
split seed: 20260909
```

Its recorded parent-group counts are 23 `ortho_Na3Fe3P4O16_13`, 7 `ortho_NaFePO4_maricite_14`, 541 `ortho_Na3VP2O8_24`, 7 `ortho_NaFePO4_olivine_15`, and 40 `NaSICON_Na4Fe2P3O12_10`. These labels describe dataset parent groups; they must not be interpreted as independently sampled framework families.

## 4. Versioned artifacts and hashes

The following required artifacts are committed on this Git branch:

| Artifact | SHA256 |
|---|---|
| `outputs/polyhedral_flow_2333476.best.pt` | `455d4437d651b1b5522b91ffbc167066dec25cda187b349f802f036241a48396` |
| `outputs/full_audit_v3/profile.json` | `67da71a46a64c48f1d417ace9135a75b2023e4fce28ea4fce8344827d1de6775` |
| `configs/framework_condition_orthophosphate.json` | `52c759d89aad189a5c877c1de6383542350793c0e9c64f8a7eef678e2582e58f` |

Checkpoint metadata records geometry-only training, `geometry_validity_weight=0`, and `coupling=polyhedral`. CHGNet evaluation used package version 0.4.2, pretrained model name 0.3.0, and model-state SHA256 `818dbd975187b2df03eb56d0c4f28fdd3bd05aa97758b2c530be80b72cfdc335`.

## 5. Jobs and current results

BSCC account and repository:

```text
ssh -p 22 scv7eyx@BSCC-N26@ssh.cn-zhongwei-1.paracloud.com
/data02/home/scv7eyx/projects/NaGen
Python: /data02/home/scv7eyx/.conda/envs/nagen-chgnet/bin/python
Slurm partition: gpu
```

### Completed

- `2333476` (`nagen-poly-flow`): completed successfully in 21m56s; trained 120-epoch polyhedral-source geometry Flow. The best checkpoint is committed in Git.
- `2333477` (`nagen-framework-gen`): completed successfully; attempted 128, sent 93 safe endpoints to CHGNet, relaxed all 93, and produced 8 current hard-gate passes. Seven met `fmax <= 0.03 eV/A`; 12 met 0.05 and 22 met 0.1. A downstream provisional analysis found all eight below the 0.15 eV/atom finite-reference hull threshold and selected six apparently nonmatching frameworks.

**Do not use job 2333477 as final evidence.** A random-number provenance bug was subsequently found: the polyhedral source did not consistently consume the explicit generator passed by the generation path. The candidates may be scientifically interesting, but the run was demoted to provisional because exact replay was not guaranteed.

### Formal reproducible run in progress

- `2333529` (`nagen-framework-gen`): running at the timestamp above.
  - generated all 128 candidates;
  - 82 passed the catastrophic precheck and entered CHGNet relaxation;
  - 46 failed because an interatomic distance was below 0.4 A;
  - 50/82 relaxation records were complete at 2026-09-09 06:57 CST;
  - final hard-feasible count, ranking, framework novelty, and selected CIFs are not yet available.
- `2333530` (`nagen-replay`): pending with dependency on `2333529`. It will replay all 128 sources and requires maximum fractional-coordinate and lattice error <= `1e-6`.

The seven CIFs in `provisional_7/` are simply the first seven completed relaxation records from job 2333529. At export time only candidate `framework_flow_seed271828_0007` passed the current hard gates. These seven are for visual inspection and are not final Top-7 structures.

## 6. Reproduction commands

Clone the isolated experiment branch:

```bash
git clone -b experiment/framework-flow-four-stage https://github.com/aoyang17/NaGen.git
cd NaGen
python -m pip install -e '.[experiment]'
export PYTHONPATH="$PWD/src"
```

Verify committed artifacts:

```bash
sha256sum \
  outputs/polyhedral_flow_2333476.best.pt \
  outputs/full_audit_v3/profile.json \
  configs/framework_condition_orthophosphate.json
python -m pytest -q
```

The committed state passed 35 local tests.

Rebuild the orthophosphate geometry dataset when `full_dataset_v3` is available:

```bash
python tools/pack_orthophosphate_data.py \
  --data outputs/full_dataset_v3 \
  --out outputs/orthophosphate_data/geometry.pt \
  --seed 20260909
```

Retrain the geometry Flow:

```bash
python -m nagen.inverse.train \
  --dataset outputs/orthophosphate_data/geometry.pt \
  --out outputs/polyhedral_flow_retrained.pt \
  --epochs 120 --atom-budget 832 --hidden 128 --layers 4 --heads 4 \
  --geometry-only --coupling polyhedral --geometry-validity-weight 0 \
  --seed 20260909
```

Generate complete fixed-composition structures:

```bash
python tools/generate_framework_candidates.py \
  --flow outputs/polyhedral_flow_2333476.best.pt \
  --condition configs/framework_condition_orthophosphate.json \
  --profile outputs/full_audit_v3/profile.json \
  --out outputs/framework_generation_replay/generation \
  --count 128 --batch-size 4 --seed 271828 --ode-steps 48 \
  --source-mode polyhedral --flow-end-time 1.0 --device cuda
```

Relax and reevaluate hard gates:

```bash
python -m nagen.selection.relax_audit \
  --input outputs/framework_generation_replay/generation/generated_safe.jsonl \
  --profile outputs/full_audit_v3/profile.json \
  --out outputs/framework_generation_replay/relax \
  --device cuda --steps 400 --fmax 0.03
```

Replay-audit the random generation exactly:

```bash
python tools/replay_framework_generation.py \
  --run outputs/framework_generation_replay \
  --flow outputs/polyhedral_flow_2333476.best.pt \
  --condition configs/framework_condition_orthophosphate.json \
  --out outputs/framework_generation_replay/generation/replay_audit.json \
  --device cuda
```

After a complete relaxation and with same-model reference labels, run four-stage selection:

```bash
python tools/select_framework_candidates.py \
  --run outputs/framework_generation_replay \
  --known outputs/full_dataset_v3 \
  --hull-labels /path/to/same_CHGNet_reference_labels.jsonl \
  --out outputs/framework_generation_replay/selection \
  --k 6 --max-reference-hull 0.15
```

On BSCC the equivalent submitted workflows are `deployment/bscc/train_polyhedral_flow.slurm`, `framework_generation.slurm`, and `replay_framework.slurm`.

## 7. Files that are not in Git and must be migrated separately

Git contains the source, profile, condition, and 13.4 MB Flow checkpoint. It intentionally does not contain generated runs or the prepared datasets because `/outputs/` and `*.pt` are normally ignored; the two required committed artifacts were explicitly versioned.

Copy these separately when continuing on another platform:

```text
/data02/home/scv7eyx/projects/NaGen/outputs/full_dataset_v3/
/data02/home/scv7eyx/projects/NaGen/outputs/orthophosphate_data/
/data02/home/scv7eyx/projects/NaGen/outputs/framework_generation_2333529/
/data02/home/scv7eyx/projects/NaGen/logs/framework-gen-2333529.log
/data02/home/scv7eyx/projects/NaGen/logs/replay-2333530.log   # after it runs
```

Also preserve an environment lock (`pip freeze`), CUDA/PyTorch versions, and either the CHGNet cache or the ability to redownload exactly the same pretrained CHGNet model.

## 8. Known limitations

1. CHGNet replaces the requested UMA backend. Its energies and forces are surrogate predictions, not DFT or UMA validation.
2. The reported hull quantity is against a finite same-model reference set; it is not a complete phase-diagram `Delta E_hull`.
3. The current formal run is incomplete, so no final success rate or final candidate set can yet be claimed.
4. Most generated candidates still fail catastrophic overlap or local-motif gates; generation quality is not yet adequate.
5. Fixed-cell CHGNet relaxation is capped at 400 steps. A record can have a low residual force without ASE's optimizer convergence flag, and vice versa near the numerical threshold; report both.
6. Charge neutrality assumes Na+, P5+, O2-, Fe2+/Fe3+. Oxygen redox, Fe4+, vacancies, disorder, and partial occupancy need separate chemical models.
7. P–O motif constraints are deliberately strict. Fe CN support is empirical, but the current cutoff and percentile margins can still reject uncommon valid environments.
8. The framework descriptor is radial/statistical and is not a metric proof of structural novelty. StructureMatcher and coordination-graph audits remain necessary.
9. The 618 orthophosphate records are correlated through a small number of parent groups. Apparent train/validation/test sample counts overstate independent framework coverage.
10. The target composition `Na6Fe6P8O32` appears only sparsely, if at all, in the training composition distribution; extrapolation risk must be reported.
11. No external `PolyAnionCathodeFilter` has been invoked in the formal run; the current motif backend is the repository's native periodic implementation.
12. Generation currently samples and then relaxes. CHGNet energy is not yet used as a differentiable DFlow source-space objective in the formal framework run.

## 9. Next steps

1. Let job `2333529` finish without modifying its inputs or checkpoint.
2. Run and inspect dependent replay job `2333530`; reject the formal run if either replay maximum exceeds `1e-6`.
3. Freeze and archive the complete generation/relax directories, logs, hashes, Slurm metadata, package versions, and GPU information.
4. Apply hard gates to all 82 relaxed endpoints and report rejection counts per gate. Never backfill failed structures to reach a desired `k`.
5. Build finite-reference hull labels with the identical CHGNet state hash, then apply the 0.15 eV/atom loose threshold.
6. Run minimax ranking and Fe–P–O-only novelty audits against all known train/validation/test frameworks; export only final selected CIFs with a manifest and round-trip verification.
7. Compare pre- and post-relax frameworks. Reject novelty claims if relaxation merely changes Na or if a candidate collapses onto a known Fe–P–O framework.
8. Diagnose overlap and PO4/FeOx failure modes, then improve the source/Flow if the formal pass rate is too low. Do not weaken hard gates simply to increase yield.
9. Add a matched ablation: uniform source versus polyhedral source, identical condition/seeds/query budget, repeated over multiple seeds.
10. Add energy-guided source optimization only after the pure-generation baseline is closed. Keep hard feasibility and novelty outside the scalar optimization objective.
11. When available, replace CHGNet with UMA or validate finalists using DFT/another independent potential before making materials claims.

## 10. Claim boundary

At handoff time the defensible claim is only:

> A conditional geometry Flow with a randomized polyhedral source can generate complete Na/Fe/P/O coordinates and lattices, and the code implements a non-scalar four-stage evaluation pipeline. A prior non-replay-certified run produced a small set of apparently feasible and framework-novel candidates, while the corrected reproducible run and its replay audit are still in progress.

Do not yet claim discovery of a stable new cathode, validated thermodynamic stability, or a final improvement over the training set.

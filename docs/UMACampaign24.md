# UMA campaign: accumulate 24 audited structures

The campaign uses `/mnt/data2/aobo/envs/NaGen/bin/python` and its existing
site-packages. It does not create another environment. A runtime replay of the
original finalist with fairchem-core 2.11.0 and torch 2.11.0+cu128 gave energy
and maximum-force differences of 3.67e-8 eV/atom and 4.88e-8 eV/Angstrom.
The original finalist still passes the strict 0.03 eV/Angstrom gates.
See `runs/campaign_benchmark.json` for the measured results.

## Execution and artifacts

```bash
bash tools/launch_uma_campaign.sh
```

The launch script starts a single coordinator owning eight GPU workers. Each
worker keeps one Flow and one proposal UMA model resident; its UMA calculator is
shared between the source-space force/stress VJP and independent fixed-cell FIRE.
The production launch now uses `batch_tf32` with four independent FIRE states
per worker. Promising endpoints trigger a lazily loaded second UMA evaluator
using the original `default` inference settings for strict final refinement.
Only those default-inference labels can qualify an accelerated candidate.
Workers process globally disjoint source seeds, starting at 2026091000. No new
training or reference-phase generation is performed.

Artifacts under `runs/campaign_24/`:

- `configuration.json`: frozen input hashes, known-reference split hashes,
  Python executable, random-seed allocation and numerical protocol.
- `processes.json`, `worker_N.log`, `worker_N.json`: process ownership and state.
- `records/NNNNNNNN.json`: atomic per-candidate generation history, final
  structure, force/energy labels, gates, hull and stage timings.
- `decisions/NNNNNNNN.json`: centralized novelty and duplicate decisions.
- `selected/*.cif` and `selected/*.audit.json`: accepted structures with full
  per-structure audits and verified CIF hashes/round trips.
- `selected/manifest.json`: collection manifest and current count.
- `summary.json`: live progress; only `status=completed` means 24 were reached.
- `STOP`: target completion marker. A human-created STOP also requests workers
  finish their current candidate and exit; inspect before removing/resuming.
- `final_cif_audit.json`: independent default-UMA predictions on the exported
  CIF files, including repeated strict gates, hull and duplicate checks. This
  is produced by `tools/audit_uma_campaign.py` after reaching the target.

The previous single finalist is re-audited and counts as the first structure.
The collector does not count a new candidate until all strict gates, the frozen
finite-reference hull threshold, known-framework novelty, cross-candidate
StructureMatcher uniqueness, radial separation and CIF round trip pass.

## Performance changes

1. Persistent per-GPU models remove repeated load/prepare/compile costs between
   generation and relaxation stages and between batches.
2. Streaming independent workers remove stage-wide barriers; one slow
   relaxation does not stall another GPU's generation.
3. FIRE terminal energy/force/stress are read from its existing Atoms/calculator
   cache, avoiding repeated predictions caused by reconstructing wrapped Atoms.
4. Finite-reference linear-program solutions are cached by composition. The
   reference optimum and decomposition are independent of candidate energy.
5. Periodic image vectors are evaluated in bounded chunks with NumPy. On the
   original finalist the neighbor benchmark measured 1.72x speedup, preserving
   the original ordering, cutoffs, vectors and nonzero self images.
6. Known framework structures, graphs and descriptors are prepared once by
   the collector. Only candidates passing force/geometry/hull enter novelty.
7. Four independent structures share one batched force call. Each FIRE keeps
   its own velocity, timestep, convergence test and 400-step cap. A regression
   test matches independent ASE trajectories and verifies early retirement.

The batch TF32 proposal path is deliberately not numerically identical to
default FP32 inference. On the eight-structure benchmark, the maximum energy
difference was 6.52e-5 eV/atom and the maximum force-component difference was
0.00391 eV/Angstrom. This is acceptable only for proposals and initial
relaxation: final force/energy/hull decisions use original default FP32
predictions. The production configuration records the inference-mode change
and source hashes; each new candidate records its proposal inference mode
and final validation model. The first 328 samples used default inference.

Benchmarks are retained in `runs/uma_inference_benchmark.json`,
`runs/uma_compiled_inference_benchmark.json`, `runs/uma_batch_benchmark.json`
and `runs/uma_batch_tf32_benchmark.json`. The tests ran alongside the active
campaign, so their timings include GPU contention. FP32 compilation was not
chosen for production: it took about seven minutes to initialize and its
measured hot-call gain was only about 13%. Four-structure batches had similar
per-structure throughput to eight while requiring less memory and waiting.

The neighbor speedup is a component benchmark, not an end-to-end speedup
claim. End-to-end stage timings are in each completed candidate record.

## Protocol and continuation

The Flow checkpoint, condition Na6Fe6P8O32, profile, UMA checkpoint and 4,385
frozen reference labels are the same as the original UMA campaign. Source
optimization uses 12 Adam DFlow steps, lr=0.03, and 48 midpoint ODE steps.
Initial fixed-cell FIRE uses maxstep=0.1, fmax=0.03, steps=400. An endpoint
passing geometry, residual force <=0.05 and hull <=0.15 receives the original
style of extra strict FIRE refinement, capped at 1,000 steps. Acceptance always
requires residual force <=0.03 and reference hull <=0.15 after refinement.
No threshold is relaxed to meet the count.

For batch records, `timing.relaxation_seconds` is the batch wall time divided
by the number of structures. `batch_relaxation_seconds` and `batch_size` retain
the measured batch time and denominator. These are amortized compute times,
not each sample's queue-to-completion latency.

Restart the launch command to resume a prematurely stopped campaign with the
same target, worker count and seed. Completed candidate records and collection
audits are reused. An incomplete in-flight candidate is regenerated from its
recorded deterministic seed. A coordinator file lock prevents simultaneous
writers; input-hash checks reject a changed checkpoint/profile/reference set.
Worker runtime errors surface in records/logs and stop that worker instead of
silently consuming an unlimited series of samples. The coordinator terminates
its own workers on completion or coordinator failure.

Selection is incremental acceptance from the physically feasible pool, with
exact cross-campaign duplicate checks. It is not a global farthest-first Top-K
over an unknown future pool. The existing batch minimax/farthest-first selector
remains available for retrospective ranking of completed candidate sets.

Novelty uses the same 22 matching-count Fe-P-O references and StructureMatcher/
coordination-graph criteria as the previous experiment; Na is removed. This is
novelty relative to that frozen reference scope. Hull remains a finite same-UMA
reference proxy, not a complete DFT thermodynamic phase diagram.

## Verification

```bash
/mnt/data2/aobo/envs/NaGen/bin/python -m pytest -q tests
/mnt/data2/aobo/envs/NaGen/bin/python tools/benchmark_uma_campaign.py \
  --gpu --out runs/campaign_benchmark.json
/mnt/data2/aobo/envs/NaGen/bin/python tools/audit_uma_campaign.py \
  --run runs/campaign_24
```

Regression coverage includes periodic self images/skew cells, hull cache
composition changes and varying candidate energies, missing hull coverage,
atomic record serialization, and the existing generation/gate tests.
Before the batch production phase, the full suite passed 38 tests with one
optional test skipped.

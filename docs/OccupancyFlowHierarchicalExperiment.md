# Conditional occupancy Flow + hierarchical constraints

## Outcome

Run `2333403` demonstrates a narrow but genuine successful case of the proposed
architecture: a trained conditional rectified-flow model maps independent
Gaussian sources to Na-site occupancy logits on a fixed Fe–P–O phosphate
backbone. Exact composition is imposed by top-k decoding, and physical quality
or novelty never enters the flow loss. CHGNet relaxation and the four-layer
selection pipeline subsequently yield eight training-set-non-equivalent
structures.

This is **family/backbone-conditioned configurational generation**, not a claim
of unconstrained crystal-topology generation. CHGNet is the requested temporary
surrogate; the reported hull is a frozen, finite-reference same-model envelope,
not a complete thermodynamic convex hull or UMA result.

## Layer-by-layer evidence

1. **Conditioning.** `N=100`, `Na12Fe12P16O60`, phosphate family, and the
   `mix_Na4Fe3P4O15_8` backbone family are fixed. The standard rectified-flow
   velocity MSE is the only training loss. A 1.0 Å mapping-quality gate retained
   77 unique TRAIN occupancy labels over 16 Na sites.
2. **Generation.** Checkpoint SHA-256 is
   `a255ebcc47e61932f6a8dfcdead533eef997b02f89236dd5ff9cd55c80c2335e`.
   Seed `314160`, 96 ODE steps, and 256 Gaussian sources produced 103 unique
   top-k-decoded configurations. Twenty-nine occupancy patterns were absent
   from the flow training patterns.
   A same-budget diagnostic confirms the checkpoint is not behaving like an
   untrained random top-k sampler: 223/256 trained-flow draws exactly hit a
   TRAIN pattern, versus 12/256 for Gaussian top-k. This also reveals strong
   mode concentration—novelty yield is only 29 unique patterns—so the result
   supports the layered novelty stage but does not claim an optimal generator.
3. **Hard feasibility.** All 103 candidates received the same fixed-cell CHGNet
   relaxation. There were no evaluation failures, 102 reached 0.03 eV/Å, and
   101 passed every hard gate (composition, charge, overlaps, force <=0.05
   eV/Å, P–O motif, and empirically calibrated Fe–O motif).
4. **Robust ranking.** The 101 feasible candidates passed the frozen 0.15
   eV/atom finite-reference hull threshold and were ranked lexicographically by
   their worst metric percentiles rather than a weighted scalar sum. Metrics
   were reference-hull energy and P/Fe off-centering, bond distortion, angle
   distortion, and coordination rarity.
5. **Diversity selection.** Novelty was applied only after ranking. Of the 101,
   72 had an exact TRAIN occupancy pattern, 17 additional candidates were
   equivalent under strict post-relax `StructureMatcher`, and 12 remained
   novelty-eligible. Farthest-first selection returned eight mutually
   non-equivalent structures.

## Selected structures

| ID | source index | max force (eV/Å) | finite-reference hull (meV/atom) |
|---|---:|---:|---:|
| `occflow_seed314160_0175` | 175 | 0.02657 | 4.191 |
| `occflow_seed314160_0002` | 2 | 0.02921 | 5.846 |
| `occflow_seed314160_0006` | 6 | 0.02692 | 6.941 |
| `occflow_seed314160_0136` | 136 | 0.02741 | 7.977 |
| `occflow_seed314160_0104` | 104 | 0.02699 | 5.590 |
| `occflow_seed314160_0101` | 101 | 0.02360 | 6.031 |
| `occflow_seed314160_0230` | 230 | 0.02245 | 7.535 |
| `occflow_seed314160_0030` | 30 | 0.02732 | 7.232 |

The independent audit verifies exact composition, all hard checks, the force
and hull thresholds, absence of a TRAIN structure match, mutual distinctness,
file hashes, and CIF round-trip for all eight structures.

## Authoritative artifacts

- Generation provenance: `outputs/occupancy_flow_2333403/generation/manifest.json`
- Flow checkpoint: `outputs/occupancy_flow_2333403/generation/flow.pt`
- Learned-vs-random diagnostic:
  `outputs/occupancy_flow_2333403/generation/distribution_baseline.json`
- Relaxation records: `outputs/occupancy_flow_2333403/relax/summary.json` and
  `pairs.jsonl`
- Ranking/diversity result: `outputs/occupancy_flow_2333403/selection_v2/summary.json`
- Independent audit: `outputs/occupancy_flow_2333403/selection_v2/independent_audit.json`
- CIF files and hashes: `outputs/occupancy_flow_2333403/selection_v2/CIF/`
- Desktop copy: `/Users/yangaobo/Desktop/NaGen_FlowMatching_selected_2333403/`

## Reproduction

Submit `deployment/bscc/occupancy_flow.slurm` on BSCC-N26. The implementation is
in `src/nagen/selection/occupancy_flow.py`, generation CLI in
`tools/run_occupancy_flow.py`, and hierarchical selector in
`tools/select_relaxed_candidates.py`.

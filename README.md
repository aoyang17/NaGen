# NaGen

NaGen generates constrained Na–Fe–P–O orthophosphate crystal candidates.
This repository's production workflow is the UMA campaign that produced the
September 10–11, 2026 candidate set. Earlier root-level implementations are
preserved under `legacy/` and are not part of the active runtime path.

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

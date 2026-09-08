"""Fixed-N/A source-space CHGNet guidance using a frozen NaGen flow.

Clamping a legacy unconditional checkpoint is experimental, not evidence of
conditional training. Family currently selects phosphate policy, not a new
learned family embedding. No terminal relaxation or novelty loss is used.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from nagen.inverse.model import CrystalState, CrystalVectorField
from nagen.inverse.sample import decode_terminal
from .energy import CHGNetEvaluator
from .optimization import optimize_source
from .pipeline import Crystal, Conditioning
from .experiment import sha256


def fixed_rollout(model, atom, frac, lattice, mask, steps):
    state = CrystalState(atom, frac, lattice)
    dt = 1. / steps
    for step in range(steps):
        time = frac.new_full((1, 1), step * dt)
        first = model(*state, time, mask)
        middle = CrystalState(atom, (state.frac + .5*dt*first.frac) % 1,
                              state.lattice + .5*dt*first.lattice)
        velocity = model(*middle, time + .5*dt, mask)
        state = CrystalState(atom, (state.frac + dt*velocity.frac) % 1,
                             state.lattice + dt*velocity.lattice)
    _, final_frac, final_cell = decode_terminal(model, state, mask)
    return final_frac, final_cell


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--count", type=int, default=16)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--ode-steps", type=int, default=24)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--gradient-structure", required=True,
                   help="Real structure JSONL for mandatory pre-optimization VJP check")
    args = p.parse_args()
    if args.count < 1 or args.steps < 0 or args.ode_steps < 1:
        p.error("invalid generation budget")
    condition = Conditioning(**json.loads(Path(args.condition).read_text()))
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.flow, map_location="cpu", weights_only=False)
    model = CrystalVectorField.from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint.get("ema_model", checkpoint["model"]))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    elements = [e for e in ("Na", "Fe", "P", "O") for _ in range(condition.counts.get(e,0))]
    indices = torch.tensor([checkpoint["element_to_index"][e]-1 for e in elements], device=device)
    atom = F.one_hot(indices, num_classes=model.config.vocab_size).float()[None] * checkpoint.get('atom_scale', 2.0)
    mask = torch.ones(1, len(elements), dtype=torch.bool, device=device)
    evaluator = CHGNetEvaluator(args.device)
    from .audit import load_crystals
    gradient = evaluator.gradient_check(next(load_crystals(args.gradient_structure)))
    if not gradient["passed"]:
        raise RuntimeError("CHGNet gradient verification failed")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("baseline.jsonl", "optimized.jsonl", "manifest.json"):
        if (out/name).exists():
            raise FileExistsError("use a fresh generation output directory")
    histories = []
    with (out/"baseline.jsonl").open("w") as baseline, (out/"optimized.jsonl").open("w") as optimized:
        for i in range(args.count):
            source = [torch.rand(1, len(elements), 3, device=device), torch.randn(1,6,device=device)]
            generate = lambda frac, cell: fixed_rollout(model, atom, frac, cell, mask, args.ode_steps)
            try:
                # Save every baseline before potential evaluation/optimization;
                # otherwise failed guidance silently censors the control arm.
                with torch.no_grad():
                    initial_frac, initial_cell = generate(*source)
                baseline.write(json.dumps({'material_id': f'seed{args.seed}_{i}_baseline',
                    'A': elements, 'X_frac': initial_frac[0].cpu().tolist(),
                    'L_matrix': initial_cell[0].cpu().tolist(), 'family': condition.family})+'\n')
                baseline.flush()
                snapshots = optimize_source(source, generate, evaluator.energy_function(elements), args.steps, args.lr)
                for handle, snapshot, arm in ((optimized,snapshots[-1],"optimized"),):
                    handle.write(json.dumps({"material_id": f"seed{args.seed}_{i}_{arm}", "A": elements,
                        "X_frac": snapshot["frac"][0].cpu().tolist(), "L_matrix": snapshot["lattice"][0].cpu().tolist(),
                        "family": condition.family, "energy_eV_atom": snapshot["energy_eV_atom"]})+"\n")
                    handle.flush()
                histories.append({"index": i, "energies_eV_atom": [s["energy_eV_atom"] for s in snapshots]})
            except (ValueError, RuntimeError) as error:
                histories.append({"index": i, "error": f"{type(error).__name__}: {error}"})
    completed = sum("error" not in h for h in histories)
    (out/"manifest.json").write_text(json.dumps({"configuration": vars(args), "model": evaluator.provenance,
        "flow_sha256": sha256(args.flow), "gradient_check": gradient, "histories": histories,
        "completed_pairs": completed, "failed_pairs": args.count-completed,
        "status": "complete" if completed == args.count else "partial" if completed else "all_failed",
        "potential_calls": evaluator.calls, "conditioning_mode": checkpoint.get('training_mode', 'experimental_fixed_atom_channel'),
        'atom_scale': checkpoint.get('atom_scale', 2.0),
        'baseline_policy': 'Retained even when potential or optimization fails; failures count in requested denominator.'}, indent=2))
    if not completed:
        raise SystemExit("all generation starts failed; inspect manifest.json")


if __name__ == "__main__":
    main()

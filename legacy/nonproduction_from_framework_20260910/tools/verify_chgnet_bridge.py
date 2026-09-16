"""Real-potential NAXL derivative smoke, not a learned-generator benchmark."""
import argparse
import json
from pathlib import Path
import torch
from nagen.selection.audit import load_crystals
from nagen.selection.energy import CHGNetEvaluator
from nagen.selection.optimization import optimize_source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    c = next(load_crystals(args.structure))
    evaluator = CHGNetEvaluator(args.device)
    gradient = evaluator.gradient_check(c)
    f = torch.tensor(c.frac[None], dtype=torch.float64)
    f = f + .002*torch.randn_like(f)
    h = torch.tensor(c.lattice[None], dtype=torch.float64)
    # Identity geometry map isolates NAXL VJP from generator quality.
    snapshots = optimize_source([f,h], lambda x,l:(x,l), evaluator.energy_function(c.elements), steps=5, lr=.0002)
    energies = [s["energy_eV_atom"] for s in snapshots]
    report = {"scope": "CHGNet NAXL bridge only, not learned DFlow", "gradient_check": gradient,
              "energies_eV_atom": energies, "energy_decreased": energies[-1] < energies[0],
              "potential_calls": evaluator.calls}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not gradient["passed"] or not report["energy_decreased"]:
        raise SystemExit("bridge verification failed")


if __name__ == "__main__":
    main()

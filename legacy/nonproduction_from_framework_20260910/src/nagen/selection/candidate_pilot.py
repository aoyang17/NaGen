"""New generated candidates only: paired DFlow and query-budget random controls.

No dataset relaxation. Every endpoint is independently gated; all failures
remain in the requested-start denominator. No property loss trains the flow.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F

from nagen.inverse.model import CrystalVectorField
from .audit import load_crystals
from .energy import CHGNetEvaluator
from .experiment import sha256, json_default
from .generate import fixed_rollout
from .optimization import optimize_source
from .pipeline import Crystal, Conditioning, hard_gates, neighbors


class RecordingEvaluator(CHGNetEvaluator):
    def evaluate(self, *args, **kwargs):
        self.latest = super().evaluate(*args, **kwargs)
        return self.latest


def catastrophic_check(c):
    volume = abs(np.linalg.det(c.lattice))/len(c.elements)
    if not np.isfinite(c.frac).all() or not np.isfinite(c.lattice).all() or not 2 <= volume <= 60 or np.linalg.cond(c.lattice) > 40:
        raise ValueError('unsafe generated cell; potential not called')
    if any(distance < .4 for i in range(len(c.elements)) for _,_,distance in neighbors(c, i, .4)):
        raise ValueError('catastrophic overlap <0.4 A; potential not called')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flow', required=True)
    p.add_argument('--condition', required=True)
    p.add_argument('--profile', required=True)
    p.add_argument('--gradient-structure', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--ode-steps', type=int, default=24)
    p.add_argument('--seed', type=int, default=17)
    args = p.parse_args()
    if args.count < 1 or args.steps < 0 or args.ode_steps < 1:
        p.error('positive generation count/ODE steps and nonnegative optimization steps required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.flow, map_location='cpu', weights_only=False)
    if checkpoint.get('training_mode') != 'fixed_NA_geometry':
        raise ValueError('pilot requires the explicitly conditional generator')
    model = CrystalVectorField.from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint['ema_model'])
    model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    condition = Conditioning(**json.loads(Path(args.condition).read_text()))
    condition_tag = sha256(args.condition)[:8]
    elements = [e for e in ['Na','Fe','P','O'] for _ in range(condition.counts.get(e, 0))]
    indices = torch.tensor([checkpoint['element_to_index'][e]-1 for e in elements], device=args.device)
    atom = checkpoint['atom_scale']*F.one_hot(indices, model.config.vocab_size).float()[None]
    mask = torch.ones(1, len(elements), device=args.device, dtype=torch.bool)
    profile = json.loads(Path(args.profile).read_text())
    evaluator = RecordingEvaluator(args.device)
    gradient = evaluator.gradient_check(next(load_crystals(args.gradient_structure)))
    if not gradient['passed']:
        raise RuntimeError('CHGNet VJP check failed')
    checks_calls = evaluator.calls
    start = time.monotonic()
    ledger, counters = [], Counter()
    generator = lambda x,h: fixed_rollout(model, atom, x, h, mask, args.ode_steps)
    # Per-pair independent RNG: guidance failures cannot shift later controls.
    with (out/'candidates.jsonl').open('x') as structures, (out/'evaluations.jsonl').open('x') as reports:
        for pair in range(args.count):
            rng = torch.Generator(device=args.device).manual_seed(args.seed*100000+pair)
            def noise():
                return [torch.rand(1,len(elements),3,device=args.device,generator=rng),
                        torch.randn(1,6,device=args.device,generator=rng)]
            initial = noise()
            control_sources = [initial]+[noise() for _ in range(args.steps)]
            arm, endpoint = 'guided', 0
            fn = evaluator.energy_function(elements)
            def objective(frac, cell):
                nonlocal endpoint
                identifier = f'{condition_tag}_seed{args.seed}_pair{pair}_{arm}_{endpoint}'
                endpoint += 1
                c = Crystal(identifier, elements, frac[0].detach().cpu().numpy(), cell[0].detach().cpu().numpy())
                record = {'id': identifier, 'arm': arm, 'pair': pair, 'endpoint': endpoint-1}
                structures.write(json.dumps({'material_id': identifier, 'A': elements,
                    'X_frac': c.frac, 'L_matrix': c.lattice, 'family': condition.family},
                    default=json_default)+'\n')
                structures.flush()
                try:
                    catastrophic_check(c)
                    energy = fn(frac, cell)
                    prediction = evaluator.latest
                    gate = hard_gates(c, condition, profile, forces=prediction['forces_eV_A'], motif_backend='native')
                    gate['metrics']['energy_eV_atom'] = prediction['energy_eV_atom']
                    record.update(status='evaluated', gate=gate, prediction=prediction)
                    counters[f'{arm}_evaluated'] += 1
                    counters[f'{arm}_passed'] += int(gate['passed'])
                    return energy
                except (ValueError, RuntimeError) as error:
                    record.update(status='failed', error=f'{type(error).__name__}: {error}')
                    counters[f'{arm}_failed'] += 1
                    raise
                finally:
                    reports.write(json.dumps(record, default=json_default, allow_nan=False)+'\n')
                    reports.flush()
            calls = evaluator.calls
            try:
                snapshots = optimize_source(initial, generator, objective, steps=args.steps, lr=.002)
                ledger.append({'pair': pair, 'arm': arm, 'status': 'completed',
                    'energies': [s['energy_eV_atom'] for s in snapshots], 'calls': evaluator.calls-calls})
            except (ValueError, RuntimeError) as error:
                ledger.append({'pair': pair, 'arm': arm, 'status': 'failed',
                    'error': f'{type(error).__name__}: {error}', 'calls': evaluator.calls-calls})
            arm, endpoint = 'random', 0
            for source in control_sources:
                calls = evaluator.calls
                try:
                    with torch.no_grad():
                        frac, cell = generator(*source)
                        energy = objective(frac, cell)
                    ledger.append({'pair': pair, 'arm': arm, 'endpoint': endpoint-1,
                                   'status': 'completed', 'energy': float(energy), 'calls': evaluator.calls-calls})
                except (ValueError, RuntimeError) as error:
                    ledger.append({'pair': pair, 'arm': arm, 'endpoint': endpoint-1,
                        'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'calls': evaluator.calls-calls})
            print(json.dumps({'pair': pair, 'counts': dict(counters)}), flush=True)
    summary = {'status': 'completed', 'configuration': vars(args), 'counts': dict(counters), 'ledger': ledger,
        'flow_sha256': sha256(args.flow), 'profile_sha256': sha256(args.profile), 'model': evaluator.provenance,
        'gradient_check': gradient, 'gradient_check_calls': checks_calls,
        'potential_calls': evaluator.calls-checks_calls, 'elapsed_seconds': time.monotonic()-start,
        'maximum_query_budget_per_arm': args.count*(args.steps+1),
        'note': 'Paired new-source starts. Random arm has steps+1 draws per pair; guided arm has '
                'steps+1 trajectory endpoints. Actual calls recorded; aborted guidance may underuse budget. '
                'All endpoints retained, no training structure perturbation, no terminal relaxation, '
                'no stability claim from CHGNet energy alone.'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()

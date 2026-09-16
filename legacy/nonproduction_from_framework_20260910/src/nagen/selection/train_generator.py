"""Train an N/A-conditioned geometry generator; CHGNet is not a training label."""
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from nagen.inverse.dataset import PackedCrystalDataset, collate_crystals
from nagen.inverse.model import CrystalVectorField, FlowConfig
from nagen.inverse.train import AtomBudgetBatchSampler, lattice_statistics, prepare_batch
from .conditional_flow import conditional_flow_loss, ATOM_SCALE
from .experiment import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--atom-budget', type=int, default=1024)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-batches', type=int, help='smoke only; absent for full epochs')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    train = PackedCrystalDataset(args.dataset, 'train')
    val = PackedCrystalDataset(args.dataset, 'val')
    if not len(train) or not len(val):
        raise ValueError('nonempty training and clean validation required')
    forbidden = [k for k in train.data if 'energy' in k or 'ehull' in k]
    if forbidden:
        raise ValueError(f'geometry training must not carry energy labels: {forbidden}')
    mean, std = lattice_statistics(train)
    model = CrystalVectorField(FlowConfig(hidden_dim=args.hidden, layers=args.layers, heads=4), mean, std).to(device)
    # Fixed discrete channels: unused atom-velocity head is never trained.
    for parameter in model.atom_velocity.parameters():
        parameter.requires_grad_(False)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-5)
    loaders = {name: DataLoader(dataset, batch_sampler=AtomBudgetBatchSampler(
        dataset, args.atom_budget, name == 'train', args.seed), collate_fn=collate_crystals,
        num_workers=0) for name,dataset in [('train',train), ('val',val)]}
    mean, std = mean.to(device), std.to(device)
    dataset_hash = sha256(args.dataset)
    best, best_epoch, history = float('inf'), -1, []
    start = time.monotonic()
    (out/'configuration.json').write_text(json.dumps(vars(args), indent=2)+'\n')
    for epoch in range(args.epochs):
        epoch_start = time.monotonic()
        lr = 2e-4 * min(1., (epoch+1)/5) * (.1+.9*.5*(1+math.cos(math.pi*epoch/max(1,args.epochs-1))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        totals, examples = np.zeros(3), 0
        for i, raw in enumerate(loaders['train']):
            if args.max_batches is not None and i >= args.max_batches:
                break
            batch = prepare_batch(raw, device, mean, std)
            # Exact uniform global translation augmentation on periodic sites.
            batch['frac_coords'] = (batch['frac_coords']+torch.rand(len(batch['N']),1,3,device=device)) % 1
            optimizer.zero_grad(set_to_none=True)
            loss, metric = conditional_flow_loss(model, batch['types'], batch['frac_coords'], batch['lattice_state'], batch['mask'])
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(norm):
                raise RuntimeError('nonfinite parameter gradient')
            optimizer.step()
            with torch.no_grad():
                for ep, mp in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(mp, .01)
            n = len(batch['N'])
            totals += n*np.array([loss.item(), metric['frac_mse'].item(), metric['lattice_mse'].item()])
            examples += n
        validation, val_examples = 0., 0
        devices = [device.index or 0] if device.type == 'cuda' else []
        # Identical validation corruption/ordering every epoch; checkpoint the
        # same EMA model used at inference, not an unrelated raw-model score.
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(9917)
            for i, raw in enumerate(loaders['val']):
                if args.max_batches is not None and i >= args.max_batches:
                    break
                batch = prepare_batch(raw, device, mean, std)
                loss, _ = conditional_flow_loss(ema, batch['types'], batch['frac_coords'], batch['lattice_state'], batch['mask'])
                validation += loss.item()*len(batch['N'])
                val_examples += len(batch['N'])
        record = {'epoch': epoch, 'train_loss': totals[0]/examples, 'train_frac_mse': totals[1]/examples,
            'train_lattice_mse': totals[2]/examples, 'val_loss': validation/val_examples,
            'train_examples': examples, 'val_examples': val_examples, 'lr': lr,
            'seconds': time.monotonic()-epoch_start}
        if not np.isfinite(record['val_loss']):
            raise RuntimeError('nonfinite validation loss')
        history.append(record)
        payload = {**model.checkpoint_payload(), 'ema_model': ema.state_dict(),
            'element_to_index': train.data['element_to_index'], 'epoch': epoch,
            'training_mode': 'fixed_NA_geometry', 'atom_scale': ATOM_SCALE,
            'coupling': 'species_assignment_uniform_unordered', 'dataset_sha256': dataset_hash,
            'optimizer': optimizer.state_dict(), 'history': history, 'configuration': vars(args),
            'physical_penalties': [], 'energy_labels_used': False}
        torch.save(payload, out/'last.pt')
        if record['val_loss'] < best:
            best, best_epoch = record['val_loss'], epoch
            torch.save(payload, out/'best.pt')
        with (out/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        if epoch-best_epoch >= args.patience:
            break
    (out/'summary.json').write_text(json.dumps({'status': 'completed', 'epochs': len(history),
        'best_epoch': best_epoch, 'best_val_loss': best, 'dataset_sha256': dataset_hash,
        'elapsed_seconds': time.monotonic()-start, 'parameters': sum(p.numel() for p in model.parameters()),
        'training_count': len(train), 'validation_count': len(val), 'test_evaluated': False,
        'note': 'Density/velocity fitting is not evidence of valid material generation.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()

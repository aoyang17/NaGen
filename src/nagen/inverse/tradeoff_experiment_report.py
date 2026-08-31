"""Render the coordination trade-off experiment summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str):
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--prior-generation", required=True)
    parser.add_argument("--prior-relaxation", required=True)
    parser.add_argument("--unconstrained-generation", required=True)
    parser.add_argument("--unconstrained-relaxation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    audit = _load(args.audit)["datasets"]["quality_training"]
    old = _load(args.old)["summary"]
    prior_g = _load(args.prior_generation)["summary"]
    prior_r_doc = _load(args.prior_relaxation)
    prior_r = prior_r_doc["summary"]
    free_g = _load(args.unconstrained_generation)["summary"]
    free_r_doc = _load(args.unconstrained_relaxation)
    free_r = free_r_doc["summary"]
    prior_best = next(r for r in prior_r_doc["records"] if r["structurally_rankable"])
    free_best = next(r for r in free_r_doc["records"] if r["structurally_rankable"])
    text = f"""# Latest NaGen coordination trade-off experiment

## Decision

Use the population-weighted Fe coordination prior for the next pilot; do not
remove Fe guidance completely. Keep P coordination strongly guided. Continue
to rank post-UMA structures by site fractions and bounded distance shortfall,
while keeping `E_hull <= 0.150 eV/atom` as the unchanged final hard gate.

No final candidate is claimed because the absolute UMA--MP energy scale remains
unaligned and `E_hull` was not validly evaluated.

## Data audit

The quality training set contains {audit['n_structures']:,} structures,
{audit['n_p_sites']:,} P sites, and {audit['n_fe_sites']:,} Fe sites. Every P
site has CN=4. Fe site counts are:

- CN=4: {audit['fe_coordination_histogram']['4']:,};
- CN=5: {audit['fe_coordination_histogram']['5']:,};
- CN=6: {audit['fe_coordination_histogram']['6']:,}.

Thus P=4 is not data-starved. Fe is strongly imbalanced toward CN=6, which
justifies a prior-weighted soft choice instead of treating CN=4/5/6 equally.

## Paired eight-source generation

| Inference constraint | Exact geometry | distance | P CN=4 | Fe CN=4/5/6 |
|---|---:|---:|---:|---:|
| Old equal-choice Fe baseline | 3/8 | 3/8 | 3/8 | 6/8 |
| Population-weighted Fe prior | {prior_g['geometry_pass']}/8 | {prior_g['check_pass_counts']['minimum_pbc_distances']}/8 | {prior_g['check_pass_counts']['p_coordination_eq_4']}/8 | {prior_g['check_pass_counts']['fe_coordination_in_4_5_6']}/8 |
| Fe penalty removed | {free_g['geometry_pass']}/8 | {free_g['check_pass_counts']['minimum_pbc_distances']}/8 | {free_g['check_pass_counts']['p_coordination_eq_4']}/8 | {free_g['check_pass_counts']['fe_coordination_in_4_5_6']}/8 |

Removing Fe guidance entirely fails before relaxation. The prior-weighted rule
retains two exact structures and is the safer trade-off.

## UMA relaxation and structural ranking

The trade-off ranking requires at least 75% valid P and Fe sites, a positive
lattice, and no pair-distance shortfall above 10%. This is a ranking rule, not
a redefinition of final `E_hull` validity.

| Configuration | UMA converged | Exact geometry | Structurally rankable |
|---|---:|---:|---:|
| Old equal-choice baseline | {old['relaxation_converged']}/8 | {old['exact_geometry_pass']}/8 | {old['structurally_rankable']}/8 |
| Population-weighted prior | {prior_r['relaxation_converged']}/2 | {prior_r['exact_geometry_pass']}/2 | {prior_r['structurally_rankable']}/2 |
| Fe penalty removed | {free_r['relaxation_converged']}/3 | {free_r['exact_geometry_pass']}/3 | {free_r['structurally_rankable']}/3 |

Best prior-weighted structure: sample {prior_best['sample_id']}, P-site fraction
{prior_best['p_cn4_site_fraction']:.3f}, Fe-site fraction
{prior_best['fe_cn456_site_fraction']:.3f}, maximum pair shortfall
{prior_best['maximum_pair_shortfall_fraction']:.3%}. Best no-Fe structure:
sample {free_best['sample_id']}, P-site fraction
{free_best['p_cn4_site_fraction']:.3f}, Fe-site fraction
{free_best['fe_cn456_site_fraction']:.3f}, maximum pair shortfall
{free_best['maximum_pair_shortfall_fraction']:.3%}.

## Conclusion

The user's trade-off hypothesis is partially supported: site-level ranking
recovers useful UMA-relaxed structures that an all-sites Boolean gate discards.
However, the data audit rejects the premise that P coordination examples are
scarce, and completely removing Fe guidance is too aggressive. The recommended
next configuration is the population-weighted Fe prior with the existing strong
distance/P restoration, followed by UMA relaxation and site-level ranking.
"""
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    print(str(target))


if __name__ == "__main__":
    main()

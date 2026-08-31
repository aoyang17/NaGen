# Experiments

Generation results use one directory per run:

```text
experiments/exp_YYYY-MM-DD/
```

If more than one run is created on the same UTC date, use
`exp_YYYY-MM-DD_02`, `exp_YYYY-MM-DD_03`, and so on. Each run should contain a
short `README.md` with its purpose, inputs, command/configuration, code version,
random seed, and a summary of the result.

Experiment outputs are durable generated artifacts. New runs live under
`/mnt/data2/aobo/NaGen/experiments/` and are exposed in this directory through
symlinks. The 2026-08-19 legacy result keeps its original durable location under
`/mnt/data2/aobo/NaGen/inverse_v1/candidates/` to avoid breaking prior references.

- `exp_2026-08-13`: initial structural cluster analysis
- `exp_2026-08-19`: first public four-candidate generation result

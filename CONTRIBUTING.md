# Contributing to ShootingCSP

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Before opening a pull request

```bash
python -m pytest -q
```

Please keep changes focused, document new public interfaces, and add regression
tests for bug fixes or new behavior. Do not modify files under `legacy/` unless
the change is explicitly about historical reproducibility.

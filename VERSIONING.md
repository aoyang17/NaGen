# Versioning

ShootingCSP follows Semantic Versioning:

- `MAJOR` for incompatible API or data-format changes;
- `MINOR` for backward-compatible features;
- `PATCH` for backward-compatible fixes.

The package version is stored in `src/shootingcsp/__init__.py` and mirrored in
`pyproject.toml` and `CITATION.cff`. A release is created by tagging the commit
with `vX.Y.Z`; the release workflow builds the source and wheel distributions.

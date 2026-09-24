# Changelog

All notable changes to ShootingCSP are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Global `shootingcsp-name-output` naming helper for canonical
  `Sxx_LE_n4_n5_n6_Eh_vv_Q` CIF and VESTA PNG outputs.
- Headless `shootingcsp-render-vesta` exporter that enforces matching CIF/PNG
  stems and applies the published Fe-P-O gallery styling.

### Changed

- Campaign, offline-selection, and intermediate CIF writers now use the shared
  global output naming implementation.

## [0.2.0] - 2026-09-24

### Added

- Versioned `shootingcsp.constraints.v1` constraint configuration.
- Offline ShootingCSP constraint builder with JSON import/export.
- Primary `shootingcsp-generate` command driven by constraint JSON.
- `shootingcsp-validate-constraints` configuration validator.
- Configurable objectives, hard gates, relaxation thresholds, novelty policy,
  source prechecks, and ShootingFlow optimization weights.

### Changed

- Fe-O coordination defaults now follow the current task table (`{4,5}`).
- Exact hard gates consume the same normalized configuration as soft guidance.

## [0.1.0] - 2026-09-23

### Added

- Initial public ShootingCSP library structure.
- ShootingFlow source-space generation and constraint optimization.
- Configurable Fe-O coordination constraints.
- Flow retraining entrypoint and training loop.
- Swappable dataset adapters for packed and JSONL data.
- Swappable differentiable surrogate model registry.
- Built-in UMA surrogate adapter.
- Independent surrogate evaluation interface.
- Library metadata, Apache-2.0 license, citation metadata, packaging defaults,
  changelog, contribution policy, security policy, and CI/release workflows.

### Changed

- The active Python package is now `shootingcsp`.
- The previous implementations remain available under `legacy/` for reference.

[Unreleased]: https://github.com/aoyang17/ShootingCSP/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/aoyang17/ShootingCSP/releases/tag/v0.2.0
[0.1.0]: https://github.com/aoyang17/ShootingCSP/releases/tag/v0.1.0

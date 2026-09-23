# Changelog

All notable changes to ShootingCSP are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/aoyang17/ShootingCSP/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aoyang17/ShootingCSP/releases/tag/v0.1.0

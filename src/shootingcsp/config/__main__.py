"""Validate and print a ShootingCSP constraint configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .loader import constraint_config_sha256, load_constraint_config, load_constraint_mapping


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    mapping = load_constraint_mapping(args.constraints)
    config = load_constraint_config(args.constraints)
    digest = constraint_config_sha256(config)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(config.as_dict(), indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "name": config.name,
        "schema_version": config.schema_version,
        "sha256": digest,
        "output": str(args.out) if args.out else None,
    }, indent=2))


if __name__ == "__main__":
    main()

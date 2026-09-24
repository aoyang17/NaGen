"""Compatibility wrapper for ``shootingcsp-generate``."""

from shootingcsp.cli.generate import main
from shootingcsp.generation import optimize_source_with_surrogate, optimize_source_with_uma
from shootingcsp.generation.source import catastrophic_reason, make_source

__all__ = [
    "catastrophic_reason",
    "main",
    "make_source",
    "optimize_source_with_surrogate",
    "optimize_source_with_uma",
]


if __name__ == "__main__":
    main()

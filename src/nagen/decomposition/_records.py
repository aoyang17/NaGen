"""Shared JSON-lines input helpers for decomposition commands."""

from __future__ import annotations

import json


def load_records(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle]

"""Small, dependency-free helpers shared by inverse-workflow CLIs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return a file digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash the canonical UTF-8 JSON representation of ``payload``."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(
    path: str | Path, payload: Any, *, sort_keys: bool = False
) -> None:
    """Write indented JSON through a sibling temporary file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload, indent=2, sort_keys=sort_keys, ensure_ascii=False
        ) + "\n"
    )
    temporary.replace(target)

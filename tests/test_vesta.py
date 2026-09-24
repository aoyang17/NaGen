from __future__ import annotations

from pathlib import Path

import pytest

from shootingcsp.vesta import vesta_png_path


def test_vesta_png_path_uses_shared_stem() -> None:
    cif = Path("/tmp/S01_LE_0_5_1_Eh_143_T.cif")
    assert vesta_png_path(cif) == Path("/tmp/S01_LE_0_5_1_Eh_143_T.png")


def test_existing_output_is_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from shootingcsp.vesta import render_vesta_png

    cif = tmp_path / "S01_LE_0_5_1_Eh_143_T.cif"
    png = tmp_path / "S01_LE_0_5_1_Eh_143_T.png"
    cif.write_text("data")
    png.write_bytes(b"x" * 2001)
    assert render_vesta_png(cif) == png

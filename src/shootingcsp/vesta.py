"""Headless VESTA PNG export with the canonical ShootingCSP output stem."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from shootingcsp.naming import vesta_png_filename


DEFAULT_VESTA_ROOT = Path("/mnt/data2/aobo/vesta/VESTA-gtk3-x86_64")


def vesta_png_path(cif_path: str | Path) -> Path:
    """Return the default PNG path, enforcing the shared CIF/PNG stem."""

    cif_path = Path(cif_path)
    return cif_path.with_name(vesta_png_filename(cif_path.stem))


def resolve_vesta_root(explicit: str | Path | None = None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        Path(os.environ["VESTA_ROOT"]) if os.environ.get("VESTA_ROOT") else None,
        DEFAULT_VESTA_ROOT,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "VESTA").is_file() and (candidate / "VESTA-gui").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "VESTA installation not found; pass --vesta-root or set VESTA_ROOT"
    )


def _prepare_structure(cif_path: Path, destination: Path, strip_elements: tuple[str, ...]) -> None:
    if not strip_elements:
        shutil.copy2(cif_path, destination)
        return
    from pymatgen.core import Structure
    from pymatgen.io.cif import CifWriter

    structure = Structure.from_file(cif_path)
    present_elements = {str(element) for element in structure.composition.elements}
    present = [element for element in strip_elements if element in present_elements]
    if present:
        structure.remove_species(present)
    CifWriter(structure, significant_figures=10).write_file(destination)


def _vesta_environment(root: Path, preferences: Path) -> dict[str, str]:
    env = os.environ.copy()
    library_paths = [
        root / "usr/lib/x86_64-linux-gnu",
        root / "lib",
        root,
    ]
    env["LD_LIBRARY_PATH"] = ":".join(
        str(path) for path in library_paths if path.is_dir()
    )
    env["VESTA_PREF"] = str(preferences)
    env["LIBGL_ALWAYS_SOFTWARE"] = "1"
    env["GDK_BACKEND"] = "x11"
    return env


def _convert_to_scene(root: Path, source: Path, target: Path, env: dict[str, str]) -> None:
    process = subprocess.run(
        [str(root / "VESTA"), "-nogui", "-i", str(source), "-o", str(target)],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    if process.returncode != 0 or not target.is_file():
        raise RuntimeError(
            "VESTA scene conversion failed:\n" + process.stdout + process.stderr
        )


def _style_fe_p_o_scene(path: Path) -> None:
    """Apply the published gallery styling to a Fe-P-O scene."""

    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"SBOND\n.*?\nSITET",
        "SBOND\n"
        "  1 Fe O 0.0 2.5 0 1 1 0 1 0.09 1.0 90 110 120\n"
        "  2 P O 0.0 2.0 0 1 1 0 1 0.09 1.0 90 110 120\n"
        "  0 0 0 0\nSITET",
        text,
        flags=re.S,
    )

    def recolor(match: re.Match[str]) -> str:
        fields = match.group(0).split()
        element = re.sub(r"\d", "", fields[1])
        style = {
            "Fe": (0.43, [28, 131, 151]),
            "P": (0.32, [237, 170, 52]),
            "O": (0.18, [211, 70, 76]),
        }
        if element not in style:
            return match.group(0)
        radius, rgb = style[element]
        fields[2] = str(radius)
        fields[3:9] = list(map(str, rgb + rgb))
        fields[9] = "160"
        return "  " + " ".join(fields)

    text = re.sub(
        r"^\s*\d+\s+(?:Fe|P|O)\d*\s+\d+\.\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+(?:\s+0)?$",
        recolor,
        text,
        flags=re.M,
    )
    text = re.sub(r"MODEL\s+\d+\s+\d+\s+\d+", "MODEL 2 1 0", text)
    text = text.replace("POLYP\n 204 1  1.000 180 180 180", "POLYP\n 160 1  0.7 70 90 100")
    path.write_text(text, encoding="utf-8")


def _render_scene(
    root: Path,
    scene: Path,
    output: Path,
    preferences: Path,
    timeout: float,
    scale: float,
    rotation: tuple[float, float, float],
) -> None:
    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        raise FileNotFoundError("xvfb-run is required for headless VESTA rendering")
    output.parent.mkdir(parents=True, exist_ok=True)
    env = _vesta_environment(root, preferences)
    log_path = scene.with_suffix(".render.log")

    with log_path.open("a", encoding="utf-8") as log:
        command = [
            xvfb_run,
            "-a",
            "-s",
            "-screen 0 1400x1100x24 -nolisten tcp",
            str(root / "VESTA-gui"),
            "-open",
            str(scene),
            "-rotate_x",
            str(rotation[0]),
            "-rotate_y",
            str(rotation[1]),
            "-rotate_z",
            str(rotation[2]),
            "-flush",
            "-export_img",
            f"scale={scale:g}",
            str(output),
        ]
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if output.is_file() and output.stat().st_size > 2000:
                    time.sleep(0.8)
                    break
                if process.poll() is not None and not output.is_file():
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                    raise RuntimeError(
                        f"VESTA exited with status {process.returncode}:\n{tail}"
                    )
                time.sleep(0.5)
            else:
                raise TimeoutError(f"VESTA export timed out; see {log_path}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)


def render_vesta_png(
    cif_path: str | Path,
    png_path: str | Path | None = None,
    *,
    vesta_root: str | Path | None = None,
    strip_elements: tuple[str, ...] = ("Na",),
    overwrite: bool = False,
    timeout: float = 60.0,
    scale: float = 2.0,
    rotation: tuple[float, float, float] = (22.0, 28.0, -8.0),
) -> Path:
    """Render a CIF to PNG and enforce the canonical shared filename stem."""

    cif_path = Path(cif_path).resolve()
    if not cif_path.is_file():
        raise FileNotFoundError(cif_path)
    output = Path(png_path).resolve() if png_path is not None else vesta_png_path(cif_path)
    if output.stem != cif_path.stem:
        raise ValueError(
            "CIF and PNG stems must match: "
            f"{cif_path.stem!r} != {output.stem!r}"
        )
    if output.exists() and not overwrite:
        if output.stat().st_size <= 2000:
            raise RuntimeError(f"existing PNG is too small: {output}")
        return output

    root = resolve_vesta_root(vesta_root)
    with tempfile.TemporaryDirectory(prefix="shootingcsp-vesta-") as temporary:
        temporary = Path(temporary)
        prepared_cif = temporary / f"{cif_path.stem}.cif"
        scene = temporary / f"{cif_path.stem}.vesta"
        preferences = temporary / "preferences"
        preferences.mkdir()
        _prepare_structure(cif_path, prepared_cif, strip_elements)
        env = _vesta_environment(root, preferences)
        _convert_to_scene(root, prepared_cif, scene, env)
        if strip_elements and set(strip_elements) >= {"Na"}:
            _style_fe_p_o_scene(scene)
        _render_scene(
            root,
            scene,
            output,
            preferences,
            timeout,
            scale,
            rotation,
        )
    if not output.is_file() or output.stat().st_size <= 2000:
        raise RuntimeError(f"VESTA did not create a valid PNG: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", required=True)
    parser.add_argument("--output", help="Defaults to the same stem with .png")
    parser.add_argument("--vesta-root", default=None)
    parser.add_argument(
        "--strip-elements",
        default="Na",
        help="Comma-separated elements to remove; use an empty value for the full structure.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--rotate-x", type=float, default=22.0)
    parser.add_argument("--rotate-y", type=float, default=28.0)
    parser.add_argument("--rotate-z", type=float, default=-8.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    strip_elements = tuple(
        element.strip()
        for element in args.strip_elements.split(",")
        if element.strip()
    )
    output = render_vesta_png(
        args.cif,
        args.output,
        vesta_root=args.vesta_root,
        strip_elements=strip_elements,
        overwrite=args.overwrite,
        timeout=args.timeout,
        scale=args.scale,
        rotation=(args.rotate_x, args.rotate_y, args.rotate_z),
    )
    print(output)


__all__ = ["render_vesta_png", "resolve_vesta_root", "vesta_png_path"]


if __name__ == "__main__":
    main()

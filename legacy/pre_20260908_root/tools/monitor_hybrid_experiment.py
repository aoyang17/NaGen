#!/usr/bin/env python3
"""Live terminal dashboard for a sharded HybridOptimization experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def run(command: list[str]) -> str:
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return ""


def completed_records(log_path: Path) -> tuple[int, bool]:
    completed = 0
    failed = False
    if not log_path.exists():
        return completed, failed
    with log_path.open(errors="replace") as handle:
        for line in handle:
            failed = failed or "Traceback (most recent call last)" in line
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            completed += int("sample_id" in row)
    return completed, failed


def process_rows(experiment_dir: Path) -> dict[int, tuple[str, str, str]]:
    rows: dict[int, tuple[str, str, str]] = {}
    output = run(["ps", "-eo", "pid=,etime=,stat=,args="])
    marker = experiment_dir.name
    for line in output.splitlines():
        if "run_surrogate_shootingflow.py" not in line or marker not in line:
            continue
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        pid, elapsed, state, command = fields
        for shard in range(4):
            if f"shard_gpu{shard}.json" in command:
                rows[shard] = (pid, elapsed, state)
    return rows


def gpu_rows() -> dict[int, str]:
    output = run([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    rows: dict[int, str] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, util, used, total, temperature, power = fields
        rows[int(index)] = (
            f"util {util:>3}% | mem {used:>6}/{total} MiB | "
            f"temp {temperature:>2} C | power {power:>6} W"
        )
    return rows


def bar(value: int, total: int, width: int = 28) -> str:
    filled = min(width, int(width * value / max(1, total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(experiment_dir: Path, shards: int, per_shard: int) -> str:
    processes = process_rows(experiment_dir)
    gpus = gpu_rows()
    lines = [
        "NaGen HybridOptimization live monitor",
        f"directory: {experiment_dir}",
        f"updated:   {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    total_done = 0
    any_failure = False
    for shard in range(shards):
        done, failed = completed_records(experiment_dir / f"gpu{shard}.log")
        total_done += done
        any_failure = any_failure or failed
        process = processes.get(shard)
        if failed:
            status = "FAILED"
        elif process:
            status = f"RUNNING pid={process[0]} elapsed={process[1]} state={process[2]}"
        elif done >= per_shard:
            status = "COMPLETE"
        else:
            status = "NOT RUNNING"
        lines.extend([
            f"GPU {shard}  {bar(done, per_shard)} {done:>2}/{per_shard}  {status}",
            f"       {gpus.get(shard, 'GPU metrics unavailable')}",
        ])
    total = shards * per_shard
    lines.extend([
        "",
        f"TOTAL  {bar(total_done, total, 40)} {total_done}/{total} "
        f"({100.0 * total_done / max(1, total):.1f}%)",
        "WARNING: traceback detected; inspect gpu*.log" if any_failure else "errors: none detected",
        "",
        "Progress advances when a candidate finishes all optimization steps.",
        "Press Ctrl-C to exit the monitor; the experiment will keep running.",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--per-shard", type=int, default=32)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    while True:
        dashboard = render(experiment_dir, args.shards, args.per_shard)
        if args.once:
            print(dashboard)
            return
        print("\033[2J\033[H" + dashboard, end="", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

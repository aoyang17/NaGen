"""Dispatch UMA hull relaxation shards only onto GPUs with no compute process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def gpu_inventory() -> tuple[dict[int, str], set[str]]:
    inventory = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    devices: dict[int, str] = {}
    for line in inventory.stdout.splitlines():
        if not line.strip():
            continue
        index, uuid = (value.strip() for value in line.split(",", maxsplit=1))
        devices[int(index)] = uuid
    processes = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True, capture_output=True, text=True,
    )
    occupied = {line.strip() for line in processes.stdout.splitlines() if line.strip()}
    return devices, occupied


def shard_complete(root: Path, shard: int, shard_count: int) -> bool:
    path = root / f"manifest_shard_{shard:03d}.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    counts = payload.get("counts", {})
    return bool(
        payload.get("shard_index") == shard
        and payload.get("shard_count") == shard_count
        and counts.get("selected") == counts.get("converged")
        and counts.get("failed") == 0
    )


def command(args: argparse.Namespace, shard: int) -> list[str]:
    return [
        args.python, "-m", "nagen.inverse.uma_hull_campaign", "relax",
        "--phase-cache", args.phase_cache,
        "--structure-cache", args.structure_cache,
        "--uma-checkpoint", args.uma_checkpoint,
        "--out-directory", args.out_directory,
        "--device", "cuda",
        "--task-name", args.task_name,
        "--shard-index", str(shard),
        "--shard-count", str(args.shard_count),
        "--fire-fmax", str(args.fire_fmax),
        "--fire-steps", str(args.fire_steps),
        "--bfgs-fmax", str(args.bfgs_fmax),
        "--bfgs-steps", str(args.bfgs_steps),
        "--resume",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--phase-cache", required=True)
    parser.add_argument("--structure-cache", required=True)
    parser.add_argument("--uma-checkpoint", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--shard-count", type=int, default=32)
    parser.add_argument("--task-name", default="omat")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--free-confirmations", type=int, default=2)
    parser.add_argument("--fire-fmax", type=float, default=0.05)
    parser.add_argument("--fire-steps", type=int, default=1500)
    parser.add_argument("--bfgs-fmax", type=float, default=0.01)
    parser.add_argument("--bfgs-steps", type=int, default=1500)
    args = parser.parse_args()
    if (
        args.shard_count <= 0 or args.poll_seconds <= 0
        or args.free_confirmations <= 0
    ):
        raise ValueError("shard count, poll interval, and confirmations must be positive")

    root = Path(args.out_directory)
    scheduler = root / "scheduler"
    scheduler.mkdir(parents=True, exist_ok=True)
    pending = [
        shard for shard in range(args.shard_count)
        if not shard_complete(root, shard, args.shard_count)
    ]
    running: dict[int, dict[str, Any]] = {}
    results: dict[int, dict[str, Any]] = {}
    handles: dict[int, Any] = {}
    free_streak: dict[int, int] = {}

    def save(status: str, inventory_error: str | None = None) -> None:
        atomic_json(scheduler / "state.json", {
            "version": "uma-hull-free-gpu-dispatch-v1",
            "status": status,
            "shard_count": args.shard_count,
            "pending": pending,
            "running": {
                str(shard): {
                    "gpu": info["gpu"], "pid": info["process"].pid,
                    "log": str(info["log"]),
                }
                for shard, info in running.items()
            },
            "results": {str(key): value for key, value in results.items()},
            "inventory_error": inventory_error,
            "free_streak": {str(key): value for key, value in free_streak.items()},
        })

    while pending or running:
        for shard, info in list(running.items()):
            returncode = info["process"].poll()
            if returncode is None:
                continue
            handles.pop(shard).close()
            complete = shard_complete(root, shard, args.shard_count)
            results[shard] = {
                "gpu": info["gpu"], "returncode": returncode,
                "complete": complete, "log": str(info["log"]),
            }
            running.pop(shard)

        inventory_error = None
        try:
            devices, occupied_uuids = gpu_inventory()
        except Exception as error:
            devices, occupied_uuids = {}, set()
            inventory_error = repr(error)
        busy_indices = {info["gpu"] for info in running.values()}
        for index, uuid in devices.items():
            free_now = uuid not in occupied_uuids and index not in busy_indices
            free_streak[index] = free_streak.get(index, 0) + 1 if free_now else 0
        free_indices = [
            index for index in sorted(devices)
            if free_streak.get(index, 0) >= args.free_confirmations
        ]
        while pending and free_indices:
            shard = pending.pop(0)
            gpu = free_indices.pop(0)
            log = scheduler / f"shard_{shard:03d}.stdout.log"
            handle = log.open("a")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(args.project) / "src")
            environment["MPLCONFIGDIR"] = "/tmp/nagen-mpl"
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command(args, shard), cwd=args.project, env=environment,
                stdout=handle, stderr=subprocess.STDOUT, text=True,
            )
            handles[shard] = handle
            running[shard] = {
                "gpu": gpu, "process": process, "log": log,
            }
            free_streak[gpu] = 0
            print(json.dumps({
                "event": "launched", "shard": shard,
                "gpu": gpu, "pid": process.pid,
            }), flush=True)
        status = "running" if running else "waiting_for_free_gpu"
        save(status, inventory_error)
        if pending or running:
            time.sleep(args.poll_seconds)

    failed = sorted(shard for shard, result in results.items() if not result["complete"])
    save("complete" if not failed else "failed")
    print(json.dumps({
        "status": "complete" if not failed else "failed",
        "completed_shards": sum(result["complete"] for result in results.values()),
        "failed_shards": failed,
    }, indent=2), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Each scenario gets a fresh process so CUDA and ORT allocator state cannot affect later cache checks.
"""Run each benchmark scenario in a separate process."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument(
        "--cuda_visible_devices",
        help="Comma-separated GPU IDs; scenarios are assigned round-robin and each sees one GPU.",
    )
    parser.add_argument(
        "--verbose", "--versbose", dest="verbose", action="store_true", help="Print each benchmark process's output."
    )
    args = parser.parse_args()
    benchmark_start = time.perf_counter()

    scenarios = json.loads(args.config.read_text())
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config must contain a non-empty JSON array of scenarios")

    gpu_ids = None
    if args.cuda_visible_devices is not None:
        gpu_ids = [gpu.strip() for gpu in args.cuda_visible_devices.split(",") if gpu.strip()]
        if not gpu_ids:
            raise ValueError("--cuda_visible_devices must contain at least one GPU ID")

    args.out.mkdir(parents=True, exist_ok=True)

    def run_scenario(item):
        index, scenario = item
        with tempfile.TemporaryDirectory() as scenario_dir:
            scenario_dir = Path(scenario_dir)
            temp_config = scenario_dir / "scenario.json"
            scenario_out = scenario_dir / "out"
            scenario_out.mkdir()
            temp_config.write_text(json.dumps([scenario], indent=2) + "\n")

            environment = os.environ.copy()
            gpu_id = None
            if gpu_ids is not None:
                gpu_id = gpu_ids[(index - 1) % len(gpu_ids)]
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            label = f"{index}/{len(scenarios)} {scenario['scenario']}"
            print(f"Starting {label}" + (f" on GPU {gpu_id}" if gpu_id is not None else ""), flush=True)
            completed = subprocess.run(
                [str(args.executable), "--config", str(temp_config), "--out", str(scenario_out)],
                env=environment,
                capture_output=not args.verbose,
                text=True,
            )

            result_path = scenario_out / f"{scenario['scenario']}_results_001.json"
            if result_path.exists():
                shutil.move(result_path, args.out / f"{scenario['scenario']}_results_{index:03d}.json")
            state = "completed" if completed.returncode == 0 else f"failed ({completed.returncode})"
            print(f"{state.capitalize()} {label}", flush=True)
            return completed.returncode != 0

    work = list(enumerate(scenarios, 1))
    max_workers = len(gpu_ids) if gpu_ids is not None else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        failed = any(executor.map(run_scenario, work))

    elapsed_seconds = time.perf_counter() - benchmark_start
    elapsed_minutes, elapsed_remainder = divmod(elapsed_seconds, 60)
    print(
        f"Benchmark {'failed' if failed else 'completed'} in {int(elapsed_minutes)}m {elapsed_remainder:.2f}s",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

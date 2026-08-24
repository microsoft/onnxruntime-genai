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
from queue import Queue
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cuda_visible_devices", required=True,
                        help="Comma-separated GPU IDs; each scenario waits for and uses one available GPU.")
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", help="Print each benchmark process's output."
    )
    args = parser.parse_args()
    benchmark_start = time.perf_counter()

    scenarios = json.loads(args.config.read_text())
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config must contain a non-empty JSON array of scenarios")

    gpu_ids = [gpu.strip() for gpu in args.cuda_visible_devices.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("--cuda_visible_devices must contain at least one GPU ID")
    # One queue token per GPU; taking a token reserves that GPU for one scenario.
    available_gpus = Queue()
    for gpu_id in gpu_ids:
        available_gpus.put(gpu_id)

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
            # Block until a GPU is free, then return it after the subprocess exits.
            gpu_id = available_gpus.get()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            label = f"{index}/{len(scenarios)} {scenario['scenario']}"
            try:
                print(f"Starting {label} on GPU {gpu_id}", flush=True)
                completed = subprocess.run(
                    [str(args.executable), "--config", str(temp_config), "--out", str(scenario_out)],
                    env=environment,
                    capture_output=not args.verbose,
                    text=True,
                )
            finally:
                available_gpus.put(gpu_id)

            result_path = scenario_out / f"{scenario['scenario']}_results_001.json"
            if result_path.exists():
                shutil.move(result_path, args.out / f"{scenario['scenario']}_results_{index:03d}.json")
            state = "completed" if completed.returncode == 0 else f"failed ({completed.returncode})"
            print(f"{state.capitalize()} {label}", flush=True)
            return completed.returncode != 0

    work = list(enumerate(scenarios, 1))
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        scenario_failed = list(executor.map(run_scenario, work))

    failed_count = sum(1 for failed in scenario_failed if failed)
    completed_count = len(scenario_failed) - failed_count
    failed = failed_count > 0

    elapsed_seconds = time.perf_counter() - benchmark_start
    elapsed_minutes, elapsed_remainder = divmod(elapsed_seconds, 60)
    print(
        f"Benchmark {'failed' if failed else 'completed'} in {int(elapsed_minutes)}m {elapsed_remainder:.2f}s "
        f"(completed={completed_count}, failed={failed_count})",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

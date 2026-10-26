#!/usr/bin/env python3
# Each scenario gets a fresh process so provider and ORT allocator state cannot affect later cache checks.
"""Run each benchmark scenario in a separate process."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cuda_visible_devices",
        help="Comma-separated GPU IDs, required for CUDA scenarios. WebGPU scenarios run serially.",
    )
    parser.add_argument("--model_path", type=Path, help="Override model_path in every scenario.")
    parser.add_argument(
        "--execution_provider_library", type=Path, help="Override the provider library in every scenario."
    )
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print each benchmark process's output.")
    args = parser.parse_args()
    benchmark_start = time.perf_counter()

    scenarios = json.loads(args.config.read_text())
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config must contain a non-empty JSON array of scenarios")

    uses_cuda = any(scenario.get("execution_provider", "cuda") == "cuda" for scenario in scenarios)
    gpu_ids = [gpu.strip() for gpu in (args.cuda_visible_devices or "").split(",") if gpu.strip()]
    if uses_cuda and not gpu_ids:
        raise ValueError("--cuda_visible_devices must contain at least one GPU ID")
    # One queue token per GPU; taking a token reserves that GPU for one scenario.
    available_gpus = Queue()
    for gpu_id in gpu_ids:
        available_gpus.put(gpu_id)
    available_webgpu = Queue()
    available_webgpu.put("default adapter")
    for scenario in scenarios:
        if args.model_path:
            scenario["model_path"] = str(args.model_path.expanduser().resolve())
        if args.execution_provider_library:
            scenario["execution_provider_library"] = str(args.execution_provider_library.expanduser().resolve())

    args.out.mkdir(parents=True, exist_ok=True)

    def run_scenario(item):
        index, scenario = item
        with tempfile.TemporaryDirectory() as temporary_directory:
            scenario_dir = Path(temporary_directory)
            temp_config = scenario_dir / "scenario.json"
            scenario_out = scenario_dir / "out"
            scenario_out.mkdir()
            temp_config.write_text(json.dumps([scenario], indent=2) + "\n")

            environment = os.environ.copy()
            is_cuda = scenario.get("execution_provider", "cuda") == "cuda"
            devices = available_gpus if is_cuda else available_webgpu
            gpu_id = devices.get()
            if is_cuda:
                environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            elif scenario.get("execution_provider") == "webgpu":
                environment.pop("CUDA_VISIBLE_DEVICES", None)
            label = f"{index}/{len(scenarios)} {scenario['scenario']}"
            log_path = args.out / f"{scenario['scenario']}_{index:03d}.log"
            try:
                print(f"Starting {label} on GPU {gpu_id}", flush=True)
                with log_path.open("w") as log:
                    completed = subprocess.run(
                        [str(args.executable), "--config", str(temp_config), "--out", str(scenario_out)],
                        check=False,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
            finally:
                devices.put(gpu_id)

            result_path = scenario_out / f"{scenario['scenario']}_results_001.json"
            failed = completed.returncode != 0
            if result_path.exists():
                failed |= json.loads(result_path.read_text()).get("status") != "success"
                shutil.move(result_path, args.out / f"{scenario['scenario']}_results_{index:03d}.json")
            else:
                print(f"Missing result for {label}", flush=True)
                failed = True
            if args.verbose or failed:
                print(log_path.read_text(), end="", flush=True)
            state = "completed" if not failed else f"failed (exit={completed.returncode}; see results and log)"
            print(f"{state.capitalize()} {label}", flush=True)
            return failed

    work = list(enumerate(scenarios, 1))
    with ThreadPoolExecutor(max_workers=max(1, len(gpu_ids))) as executor:
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

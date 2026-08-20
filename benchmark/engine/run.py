#!/usr/bin/env python3
# Each scenario gets a fresh process so CUDA and ORT allocator state cannot affect later cache checks.
"""Run each benchmark scenario in a separate process."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    scenarios = json.loads(args.config.read_text())
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("config must contain a non-empty JSON array of scenarios")

    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = Path(temp_dir) / "scenario.json"
        for index, scenario in enumerate(scenarios, 1):
            temp_config.write_text(json.dumps([scenario], indent=2) + "\n")
            completed = subprocess.run([str(args.executable), "--config", str(temp_config), "--out", str(args.out)])
            result_name = f"{scenario['scenario']}_results_001.json"
            result_path = args.out / result_name
            if result_path.exists():
                shutil.move(result_path, args.out / f"{scenario['scenario']}_results_{index:03d}.json")
            failed |= completed.returncode != 0

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

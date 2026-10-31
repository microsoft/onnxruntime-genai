#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Run a Qwen offload command with a bounded expert cache and optional cgroup RAM enforcement.

Example: python run_with_ram_budget.py --ram-budget-gib 64 -- python bench_engine_mtp.py PACKAGE --out result.json

The default 8 GiB reserve covers non-expert allocations, pageable prefill waves and runtime overhead.
Only the expert cache allocation is bounded internally. --enforce additionally uses a systemd user scope to
cap the entire command (including children and charged file cache); it requires a working user manager.
"""

import argparse
import json
import math
import os
import subprocess
import sys


def plan_environment(ram_gib, reserve_gib, layers, environment):
    if not all(math.isfinite(value) and value >= 0 for value in (ram_gib, reserve_gib)) or layers <= 0:
        raise ValueError("RAM sizes must be finite and nonnegative, and layers must be positive")
    env = environment.copy()
    if ram_gib == 0:
        env["ORT_QMOE_HOST_CACHE_MIB"] = "0"
        env.pop("ORT_QMOE_HOST_CACHE_LAYERS", None)
        return env, 0
    cache_mib = math.floor((ram_gib - reserve_gib) * 1024)
    # This launcher targets Qwen's 2560 x 640 NVFP4 experts, including aligned global scales.
    if cache_mib <= 0 or cache_mib >= 2**31 or (cache_mib << 20) // layers < 2765312:
        raise ValueError("RAM budget minus reserve must fit at least one NVFP4 expert per layer")
    for name in (
        "ORT_QMOE_HOST_COMPLEMENT_MODE",
        "ORT_QMOE_SSD_CACHE_SLOTS",
        "ORT_QMOE_SHARED_PIN_EXPERTS",
        "ORT_QMOE_SHARED_STREAM_ROWS",
        "ORTGENAI_PIN_CPU_POOL",
    ):
        if env.get(name, "0") != "0":
            raise ValueError(f"Unset {name} before enabling the bounded host cache")
    env["ORT_QMOE_HOST_CACHE_MIB"] = str(cache_mib)
    env["ORT_QMOE_HOST_CACHE_LAYERS"] = str(layers)
    return env, cache_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ram-budget-gib", type=float, required=True, help="total RAM target; 0 preserves unlimited mode"
    )
    parser.add_argument("--reserve-gib", type=float, default=8, help="RAM outside the expert cache (default: 8 GiB)")
    parser.add_argument("--layers", type=int, default=48, help="number of offload layers in this model (default: 48)")
    parser.add_argument("--enforce", action="store_true", help="enforce total RAM with systemd-run --user (no swap)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        env, cache_mib = plan_environment(args.ram_budget_gib, args.reserve_gib, args.layers, os.environ)
    except ValueError as exc:
        parser.error(str(exc))
        return 2  # parser.error exits; this keeps env and cache_mib provably assigned below
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a command after --")
    if args.enforce:
        if args.ram_budget_gib == 0:
            parser.error("--enforce requires a positive RAM budget")
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            f"--property=MemoryMax={math.floor(args.ram_budget_gib * (1 << 30))}",
            "--property=MemorySwapMax=0",
            "--",
            *command,
        ]
    print(
        json.dumps(
            {
                "ram_budget_gib": args.ram_budget_gib,
                "reserve_gib": args.reserve_gib,
                "expert_cache_mib": cache_mib,
                "layers": args.layers,
                "enforce_total_ram": args.enforce,
                "command": command,
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    if args.dry_run:
        return 0
    # A failed systemd invocation fails closed; never rerun without the requested limit.
    return subprocess.call(command, env=env)


if __name__ == "__main__":
    sys.exit(main())

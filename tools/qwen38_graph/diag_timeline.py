#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Per-token timeline diagnostic: prints the time between consecutive tokens (cycle boundaries show up as the 2nd token of an
accepted draft being ~0). Used to look for slow start-up cycles (graph capture, clock ramp)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import onnxruntime_genai as og

sys.path.insert(0, "/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01/mtp_runner")
import prompts  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--no-head-prefill", action="store_true")
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()
    og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(args.model)
    config.overlay(json.dumps({"model": {"qwen38_graph": {"mtp": {"prefill_head": not args.no_head_prefill}}}}))
    model = og.Model(config)
    for run in range(args.runs):
        prompt = prompts.make("text", args.context, 2048 + 7 * run)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=len(prompt) + 256, do_sample=False)
        generator = og.Generator(model, params)
        generator.append_tokens(prompt)
        stamps = []
        while not generator.is_done():
            generator.generate_next_token()
            stamps.append(time.perf_counter())
        deltas = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
        cycles = [d for d in deltas if d > 1.0]  # a cycle boundary = a token that needed the GPU
        thirds = [len(cycles) // 3, 2 * len(cycles) // 3]
        head, mid, tail = cycles[: thirds[0]], cycles[thirds[0] : thirds[1]], cycles[thirds[1] :]
        mean = lambda xs: sum(xs) / max(1, len(xs))  # noqa: E731
        print(f"run {run}: cycles {len(cycles)} mean ms {mean(cycles):.2f}; first third {mean(head):.2f} mid {mean(mid):.2f} "
              f"last {mean(tail):.2f}; first 12 cycles {[round(x, 1) for x in cycles[:12]]}", flush=True)
        del generator


if __name__ == "__main__":
    main()

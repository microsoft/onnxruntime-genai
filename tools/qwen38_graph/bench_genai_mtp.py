#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""vLLM-protocol decode benchmark of the GenAI qwen38_flash_graph model (modelled on CG/mtp_runner/bench_mtp.py).

Exact prompt lengths, 256 new tokens, EOS ignored (package built with build_package.py --stop-token-ids 248319), natural text from the vLLM source pool (frozen under runs/), 1 warmup +
3 measured runs on distinct prompts (seeds 1000+n for the warmup, i*7+n for run i), TPS = (n-1) generated tokens / (time of the
last token - time of the first token), through the public Generator API:
    generator.append_tokens(prompt); while not generator.is_done(): generator.generate_next_token(); generator.get_next_tokens()

usage: bench_genai_mtp.py <package> --out <json> [--contexts 2048,8192,32768] [--kind text] [--no-mtp] [--reps 3] [--new 256]
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

RUNNER = Path("/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01/mtp_runner")
sys.path.insert(0, str(RUNNER))
import prompts  # noqa: E402


def decode_once(model, prompt, new):
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + new, do_sample=False)
    generator = og.Generator(model, params)
    t0 = time.perf_counter()
    generator.append_tokens(prompt)
    t_append = time.perf_counter()
    count = 0
    tokens = []
    t_first = t_last = None
    while not generator.is_done():
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        now = time.perf_counter()
        tokens.append(int(token))
        if t_first is None:
            t_first = now
        t_last = now
        count += 1
    stats = generator.get_speculative_stats()
    del generator
    decode_s = t_last - t_first
    return {"tokens": count, "hash": hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest(), "prefill_append_s": t_append - t0, "first_token_s": t_first - t0, "decode_s": decode_s,
            "tps": (count - 1) / decode_s, "cycles": stats["rounds"], "accepted": stats["draft_tokens_accepted"],
            "accept_rate": stats["draft_tokens_accepted"] / stats["rounds"] if stats["rounds"] else None,
            "ms_per_cycle": decode_s / stats["rounds"] * 1e3 if stats["rounds"] else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", default="2048,8192,32768")
    parser.add_argument("--kind", default="text", choices=("text", "random", "prose"))
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--no-head-prefill", action="store_true")
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(args.model)
    graph = {"mtp": {"enabled": not args.no_mtp, "prefill_head": not args.no_head_prefill}}
    config.overlay(json.dumps({"model": {"qwen38_graph": graph}}))
    model = og.Model(config)

    result = {"args": {k: str(v) for k, v in vars(args).items()}, "contexts": {}}
    for n in [int(x) for x in args.contexts.split(",")]:
        seeds = [1000 + n] + [i * 7 + n for i in range(args.reps)]
        runs = []
        for idx, seed in enumerate(seeds):
            prompt = prompts.make(args.kind, n, seed)
            record = decode_once(model, prompt, args.new)
            record.update({"label": "warmup" if idx == 0 else f"run{idx}", "seed": seed})
            runs.append(record)
            print(json.dumps({"ctx": n, **record}), flush=True)
        measured = sorted((r for r in runs if r["label"] != "warmup"), key=lambda r: r["tps"])
        result["contexts"][n] = {"median": measured[len(measured) // 2], "runs": runs}
        median = result["contexts"][n]["median"]
        print(f"ctx {n}: decode {median['tps']:.2f} tps (median of {len(measured)}), accept {median['accept_rate']}", flush=True)
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

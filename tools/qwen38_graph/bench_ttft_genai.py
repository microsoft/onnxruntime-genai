#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Time to first token of the GenAI qwen38_flash_graph model (fast prefill) through the public API.

TTFT = time from before generator.append_tokens(prompt) to the return of the first generate_next_token() (dense prefill + hand-off +
head KV prefill + first draft + first verify enqueue); the append / first-token split is reported too. Natural text of the vLLM
source pool (as bench_mtp.py), median of --reps distinct prompts per length, plus MMLU-Pro prompts (--mmlu N, evenly spaced).

usage: bench_ttft_genai.py <package> --out <json> [--lengths 128,...,32768] [--reps 5] [--mmlu 80] [--prefill fast|steps]
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import onnxruntime_genai as og

RUNNER = Path("/home/tianlei/git/eval_runs/qwen38_cuda_graph_20261010_01/mtp_runner")
sys.path.insert(0, str(RUNNER))
import prompts  # noqa: E402

MMLU = Path("/home/tianlei/git/eval_runs/qwen38_mmlu_mtp_20261012_01/prompts.jsonl")


def ttft(model, prompt):
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + 8, do_sample=False)
    generator = og.Generator(model, params)
    t0 = time.perf_counter()
    generator.append_tokens(prompt)
    t1 = time.perf_counter()
    generator.generate_next_token()
    first = int(generator.get_next_tokens()[0])
    t2 = time.perf_counter()
    del generator
    return {"ttft_s": t2 - t0, "append_s": t1 - t0, "first_s": t2 - t1, "first_token": first}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", default="128,256,512,1024,2048,8192,32768")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--mmlu", type=int, default=0)
    parser.add_argument("--prefill", default="fast", choices=("fast", "steps"))
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(args.model)
    config.overlay(json.dumps({"model": {"qwen38_graph": {"prefill": {"enabled": args.prefill == "fast"}}}}))
    t = time.perf_counter()
    model = og.Model(config)
    result = {"load_s": time.perf_counter() - t, "lengths": {}, "args": {k: str(v) for k, v in vars(args).items()}}
    print(f"model load {result['load_s']:.1f} s", flush=True)
    for n in [int(x) for x in args.lengths.split(",") if x]:
        runs = [ttft(model, prompts.make("text", n, 2000 + n + 7 * i)) for i in range(args.reps + 1)]  # first = warm
        measured = runs[1:]
        result["lengths"][n] = {"median_ttft_s": statistics.median(r["ttft_s"] for r in measured),
                                "median_append_s": statistics.median(r["append_s"] for r in measured),
                                "median_first_s": statistics.median(r["first_s"] for r in measured), "runs": runs}
        print(f"text {n}: TTFT median {result['lengths'][n]['median_ttft_s']:.3f} s (append {result['lengths'][n]['median_append_s']:.3f}, "
              f"first token {result['lengths'][n]['median_first_s']:.3f}); runs {[round(r['ttft_s'], 3) for r in runs]}", flush=True)
        args.out.write_text(json.dumps(result, indent=1))
    if args.mmlu:
        rows = [json.loads(line) for line in open(MMLU)]
        rows = rows[:: max(1, len(rows) // args.mmlu)][: args.mmlu]
        ttft(model, rows[0]["prompt_ids"])  # warm
        runs = []
        for row in rows:
            r = ttft(model, row["prompt_ids"])
            r["tokens"] = len(row["prompt_ids"])
            r["index"] = row["index"]
            runs.append(r)
        values = sorted(r["ttft_s"] for r in runs)
        result["mmlu"] = {"n": len(runs), "mean_tokens": statistics.mean(r["tokens"] for r in runs),
                          "mean_ttft_s": statistics.mean(values), "median_ttft_s": statistics.median(values),
                          "p95_ttft_s": values[int(0.95 * (len(values) - 1))], "max_ttft_s": values[-1], "runs": runs}
        print("mmlu", {k: v for k, v in result["mmlu"].items() if k != "runs"}, flush=True)
        args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

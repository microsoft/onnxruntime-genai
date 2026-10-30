#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Time to first token of the Engine-hosted qwen38_flash_graph backend.

TTFT = time from before request.begin_turn(prompt) to the return of the Engine.run call that delivers the first token (all prompt
chunks, the dense -> packed hand-off, the batched head KV prefill, the first draft and the enqueue of the first verify cycle all
happen before that step returns). The turn then runs to its 2-token limit so that the next prompt starts on a clean slot. Also
reported: the time of the steps before the one that delivers the token (prefill chunks) and of that last step (last chunk + hand-off).
Natural text windows of --text-pool (as bench_engine_mtp.py), median of --reps distinct prompts per length, plus MMLU-Pro prompt
ids (--mmlu N evenly spaced rows of --mmlu-prompts, a jsonl with "prompt_ids" and "index").

usage: bench_ttft_engine.py <package> --out <json> [--lengths 128,...,32768] [--reps 5] [--mmlu 80 --mmlu-prompts FILE]
                            [--prefill fast|steps]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
from qwen38_engine_util import Prompts, add_prompt_arguments, load_model


def ttft(engine, buffer, prompt):
    request = engine.create_request()
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    turn.set_max_generated_tokens(2)
    t0 = time.perf_counter()
    request.begin_turn(np.asarray(prompt, dtype=np.int32), turn)
    steps = []
    first = None
    t_prev = t0
    while engine.has_pending_requests():
        events = engine.run(buffer)
        now = time.perf_counter()
        if first is None:
            steps.append(now - t_prev)
            t_prev = now
            for event in events:
                if event.flags & og.EngineEventFlags.TOKEN:
                    first = (now - t0, int(event.token))
                    break
        for event in events:
            if event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED):
                raise RuntimeError(f"Engine failure: flags={event.flags} error={event.error_code}")
    request.close()
    return {
        "ttft_s": first[0],
        "prefill_steps_s": sum(steps[:-1]),
        "last_step_s": steps[-1],
        "engine_steps": len(steps),
        "first_token": first[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lengths", default="128,256,512,1024,2048,8192,32768")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--mmlu", type=int, default=0)
    parser.add_argument("--mmlu-prompts", type=Path, default=os.environ.get("QWEN38_MMLU_PROMPTS"))
    parser.add_argument("--prefill", default=None, choices=("fast", "steps"))
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    add_prompt_arguments(parser)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    overlay = {"model": {"qwen38_graph": {"prefill": {"enabled": args.prefill == "fast"}}}} if args.prefill else None
    t = time.perf_counter()
    model = load_model(args.model, args.provider, overlay)
    prompts = Prompts.from_args(model, args)
    result = {"load_s": time.perf_counter() - t, "lengths": {}, "args": {k: str(v) for k, v in vars(args).items()}}
    print(f"model load {result['load_s']:.1f} s", flush=True)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)
    for n in [int(x) for x in args.lengths.split(",") if x]:
        runs = [
            ttft(engine, buffer, prompts.make("text", n, 2000 + n + 7 * i)) for i in range(args.reps + 1)
        ]  # first = warm
        measured = runs[1:]
        entry = {
            "median_ttft_s": statistics.median(r["ttft_s"] for r in measured),
            "median_prefill_steps_s": statistics.median(r["prefill_steps_s"] for r in measured),
            "median_last_step_s": statistics.median(r["last_step_s"] for r in measured),
            "runs": runs,
        }
        result["lengths"][n] = entry
        print(
            f"text {n}: TTFT median {entry['median_ttft_s']:.3f} s (chunk steps {entry['median_prefill_steps_s']:.3f}, "
            f"last step {entry['median_last_step_s']:.3f}); runs {[round(r['ttft_s'], 3) for r in runs]}",
            flush=True,
        )
        args.out.write_text(json.dumps(result, indent=1))
    if args.mmlu:
        with open(args.mmlu_prompts) as handle:
            rows = [json.loads(line) for line in handle]
        rows = rows[:: max(1, len(rows) // args.mmlu)][: args.mmlu]
        ttft(engine, buffer, rows[0]["prompt_ids"])  # warm
        runs = []
        for row in rows:
            r = ttft(engine, buffer, row["prompt_ids"])
            r["tokens"] = len(row["prompt_ids"])
            r["index"] = row["index"]
            runs.append(r)
        values = sorted(r["ttft_s"] for r in runs)
        result["mmlu"] = {
            "n": len(runs),
            "mean_tokens": statistics.mean(r["tokens"] for r in runs),
            "mean_ttft_s": statistics.mean(values),
            "median_ttft_s": statistics.median(values),
            "p95_ttft_s": values[int(0.95 * (len(values) - 1))],
            "max_ttft_s": values[-1],
            "runs": runs,
        }
        print("mmlu", {k: v for k, v in result["mmlu"].items() if k != "runs"}, flush=True)
        args.out.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

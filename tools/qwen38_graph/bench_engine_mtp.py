#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""vLLM-protocol decode benchmark of the Engine-hosted qwen38_flash_graph backend.

Exact prompt lengths, 256 new tokens, EOS ignored (package built with build_package.py --stop-token-ids 248319), natural
text from the vLLM source pool (frozen under the research runner's runs/), 1 warm-up + 3 measured runs on distinct prompts
(seed 1000+n for the warm-up, i*7+n for run i). TPS = (n-1) generated tokens / (time the last token was delivered - time
the first token was delivered), measured around Engine.run:

    request.begin_turn(prompt, turn); while engine.has_pending_requests(): events = engine.run(buffer)

usage: bench_engine_mtp.py <package> --out <json> [--contexts 2048,8192,32768] [--kind text] [--no-mtp] [--reps 3] [--new 256]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import onnxruntime_genai as og
from qwen38_engine_util import load_model, research_prompts, run_turn, token_hash


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

    prompts = research_prompts()
    overlay = {
        "model": {"qwen38_graph": {"mtp": {"enabled": not args.no_mtp, "prefill_head": not args.no_head_prefill}}}
    }
    model = load_model(args.model, args.provider, overlay)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)

    result = {"args": {k: str(v) for k, v in vars(args).items()}, "contexts": {}}
    for n in [int(x) for x in args.contexts.split(",")]:
        seeds = [1000 + n] + [i * 7 + n for i in range(args.reps)]
        runs = []
        for index, seed in enumerate(seeds):
            prompt = prompts.make(args.kind, n, seed)
            before = engine.get_speculative_stats()
            request = engine.create_request()
            turn, _ = run_turn(engine, buffer, request, prompt, args.new)
            request.close()
            after = engine.get_speculative_stats()
            count = len(turn.tokens)
            decode_s = turn.times[-1] - turn.times[0]
            rounds = after["rounds"] - before["rounds"]
            accepted = after["draft_tokens_accepted"] - before["draft_tokens_accepted"]
            record = {
                "label": "warmup" if index == 0 else f"run{index}",
                "seed": seed,
                "tokens": count,
                "finish_reason": str(turn.finish_reason),
                "hash": token_hash(turn.tokens),
                "first_token_s": turn.times[0] - turn.start,
                "decode_s": decode_s,
                "tps": (count - 1) / decode_s,
                "cycles": rounds,
                "accepted": accepted,
                "accept_rate": accepted / rounds if rounds else None,
                "ms_per_cycle": decode_s / rounds * 1e3 if rounds else None,
                "engine_runs": turn.runs,
            }
            runs.append(record)
            print(json.dumps({"ctx": n, **record}), flush=True)
        measured = sorted((r for r in runs if r["label"] != "warmup"), key=lambda r: r["tps"])
        median = measured[len(measured) // 2]
        result["contexts"][n] = {"median": median, "runs": runs}
        print(
            f"ctx {n}: decode {median['tps']:.2f} tps (median of {len(measured)}), accept {median['accept_rate']}",
            flush=True,
        )
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())

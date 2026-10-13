#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Prompt-length and chunk-boundary check of the Engine-hosted qwen38_flash_graph backend.

Runs greedy turns of --new tokens for prompts of many lengths (1 token, odd / even lengths, dense chunk boundaries 8191 / 8192 /
8193, two and three chunks) and writes the token streams. Run it twice on the same package, once with --prefill steps (the
bit-exact S=2-step prompt path) and once with --prefill fast (dense chunks + hand-off), then compare:

  engine_prompt_lengths.py compare steps.json fast.json [--min-prefix N]

The fast prefill is tolerance level: the first token must agree for every length and the streams must share a prefix of at
least --min-prefix tokens (default 8). Each turn also checks that the Engine finished with the token limit and usage matches.

usage: engine_prompt_lengths.py run <package> --prefill steps|fast --out <json> [--lengths 1,2,...] [--new 32]
"""

import argparse
import json
import sys
from pathlib import Path

import onnxruntime_genai as og
from qwen38_engine_util import load_model, research_prompts, run_turn, token_hash

LENGTHS = "1,2,3,4,5,7,8,31,127,128,129,1000,2047,2048,2049,4097,8191,8192,8193,12000,16384,16385,20000"


def run(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    prompts = research_prompts()
    model = load_model(
        args.model, args.provider, {"model": {"qwen38_graph": {"prefill": {"enabled": args.prefill == "fast"}}}}
    )
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)
    results = []
    for n in [int(x) for x in args.lengths.split(",")]:
        prompt = prompts.make("text", max(n, 64), 77 + n)[:n]
        request = engine.create_request()
        turn, _ = run_turn(engine, buffer, request, prompt, args.new)
        request.close()
        entry = {
            "length": n,
            "tokens": turn.tokens,
            "hash": token_hash(turn.tokens),
            "finish_reason": str(turn.finish_reason),
            "usage_ok": turn.usage.prompt_tokens == n and turn.usage.generated_tokens == len(turn.tokens),
            "engine_runs": turn.runs,
        }
        results.append(entry)
        print(json.dumps({k: entry[k] for k in ("length", "finish_reason", "usage_ok", "engine_runs")}), flush=True)
    args.out.write_text(json.dumps({"prefill": args.prefill, "new": args.new, "results": results}, indent=1))


def compare(args):
    a = {r["length"]: r for r in json.loads(Path(args.reference).read_text())["results"]}
    b = {r["length"]: r for r in json.loads(Path(args.candidate).read_text())["results"]}
    ok = True
    for n in sorted(a):
        x, y = a[n]["tokens"], b[n]["tokens"]
        common = 0
        while common < min(len(x), len(y)) and x[common] == y[common]:
            common += 1
        good = bool(x) and bool(y) and x[0] == y[0] and common >= min(args.min_prefix, len(x), len(y))
        good &= a[n]["usage_ok"] and b[n]["usage_ok"]
        ok &= good
        print(f"length {n:6d}: first token equal {x[:1] == y[:1]}, identical prefix {common}/{len(x)}, ok {good}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("model")
    r.add_argument("--prefill", choices=("steps", "fast"), required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--lengths", default=LENGTHS)
    r.add_argument("--new", type=int, default=32)
    r.add_argument("--provider")
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--min-prefix", type=int, default=8)
    args = parser.parse_args()
    return run(args) if args.mode == "run" else compare(args)


if __name__ == "__main__":
    sys.exit(main())

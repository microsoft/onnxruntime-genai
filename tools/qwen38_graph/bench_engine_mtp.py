#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""vLLM-protocol decode benchmark of the Engine-hosted qwen38_flash_graph backend.

Exact prompt lengths, 256 new tokens, EOS ignored (package built with build_package.py --stop-token-ids 248319), natural
text windows of --text-pool (any natural text of at least ~40k tokens), 1 warm-up + 3 measured runs on distinct prompts
(seed 1000+n for the warm-up, i*7+n for run i). TPS = (n-1) generated tokens / (time the last token was delivered - time
the first token was delivered), measured around Engine.run:

    request.begin_turn(prompt, turn); while engine.has_pending_requests(): events = engine.run(buffer)

usage: bench_engine_mtp.py <package> --out <json> [--contexts 2048,8192,32768] [--kind text] [--no-mtp] [--reps 3] [--new 256]
                           [--text-pool FILE] [--prose-pool FILE]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import onnxruntime_genai as og
from qwen38_engine_util import Prompts, add_prompt_arguments, load_model, run_turn, token_hash


def host_memory_stats():
    """Collect Linux host accounting outside timed decode; cgroup peak includes charged file cache."""
    stats = {"expert_cache_mib": os.environ.get("ORT_QMOE_HOST_CACHE_MIB", "0")}
    try:
        stats["process_status"] = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.startswith(("VmRSS:", "VmHWM:", "VmLck:", "RssAnon:", "RssFile:", "RssShmem:"))
        }
        stats["io"] = dict(line.split(": ", 1) for line in Path("/proc/self/io").read_text().splitlines())
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                group = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
                stats["cgroup"] = str(group)
                for name in ("memory.max", "memory.peak", "memory.current", "memory.events", "memory.swap.max"):
                    stats[name] = (group / name).read_text().strip()
    except OSError:
        pass
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contexts", default="2048,8192,32768")
    parser.add_argument("--kind", default="text", choices=("text", "random", "prose"))
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--stop-token-id", type=int, default=248319,
                        help="benchmark stop token (default: unused token 248319, for fixed-length runs)")
    parser.add_argument("--no-head-prefill", action="store_true")
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    add_prompt_arguments(parser)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    overlay = {
        "model": {"eos_token_id": args.stop_token_id, "qwen38_graph": {"mtp": {"enabled": not args.no_mtp, "prefill_head": not args.no_head_prefill}}}
    }
    model = load_model(args.model, args.provider, overlay)
    prompts = Prompts.from_args(model, args)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)

    result = {"args": {k: str(v) for k, v in vars(args).items()}, "contexts": {}}
    for n in [int(x) for x in args.contexts.split(",")]:
        seeds = ([] if args.no_warmup else [1000 + n]) + [i * 7 + n for i in range(args.reps)]
        runs = []
        for index, seed in enumerate(seeds):
            prompt = prompts.make(args.kind, n, seed)
            before = engine.get_speculative_stats()
            request = engine.create_request()
            turn, _ = run_turn(engine, buffer, request, prompt, args.new)
            request.close()
            after = engine.get_speculative_stats()
            count = len(turn.tokens)
            if count != args.new:
                raise RuntimeError(f"Benchmark expected {args.new} tokens, got {count}: {turn.finish_reason}")
            decode_s = turn.times[-1] - turn.times[0]
            rounds = after["rounds"] - before["rounds"]
            accepted = after["draft_tokens_accepted"] - before["draft_tokens_accepted"]
            proposed = after["draft_tokens_proposed"] - before["draft_tokens_proposed"]
            head_calls = after["draft_forward_passes"] - before["draft_forward_passes"]
            before_hist = before.get("acceptance_length_histogram", [])
            after_hist = after.get("acceptance_length_histogram", [])
            hist_len = min(8, len(after_hist), len(before_hist))
            accept_hist = [after_hist[i] - before_hist[i] for i in range(hist_len)]
            while len(accept_hist) < 4:
                accept_hist.append(0)
            ge1 = sum(accept_hist[1:])
            ge2 = sum(accept_hist[2:])
            ge3 = accept_hist[3]
            pos_accept_rates = [
                ge1 / rounds if rounds else 0.0,
                ge2 / rounds if rounds else 0.0,
                ge3 / rounds if rounds else 0.0,
            ]
            cond_accept_rates = [
                ge1 / rounds if rounds else 0.0,
                ge2 / ge1 if ge1 else 0.0,
                ge3 / ge2 if ge2 else 0.0,
            ]
            record = {
                "label": ("warmup" if index == 0 and not args.no_warmup else f"run{index if not args.no_warmup else index + 1}"),
                "seed": seed,
                "tokens": count,
                "finish_reason": str(turn.finish_reason),
                "hash": token_hash(turn.tokens),
                "first_token_s": turn.times[0] - turn.start,
                "decode_s": decode_s,
                "tps": (count - 1) / decode_s,
                "cycles": rounds,
                "proposed": proposed,
                "accepted": accepted,
                "accept_rate": accepted / rounds if rounds else None,
                "accepted_per_proposed": accepted / proposed if proposed else None,
                "head_calls": head_calls,
                "accept_hist": accept_hist,
                "pos_accept_rates": pos_accept_rates,
                "cond_accept_rates": cond_accept_rates,
                "tokens_per_cycle": (count - 1) / rounds if rounds else None,
                "ms_per_cycle": decode_s / rounds * 1e3 if rounds else None,
                "engine_runs": turn.runs,
            }
            runs.append(record)
            print(json.dumps({"ctx": n, **record}), flush=True)
        measured = sorted((r for r in runs if r["label"] != "warmup"), key=lambda r: r["tps"]) or runs
        median = measured[len(measured) // 2]
        result["contexts"][n] = {"median": median, "runs": runs}
        print(
            f"ctx {n}: decode {median['tps']:.2f} tps (median of {len(measured)}), accept {median['accept_rate']}",
            flush=True,
        )
        result["host_memory"] = host_memory_stats()
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    sys.exit(main())

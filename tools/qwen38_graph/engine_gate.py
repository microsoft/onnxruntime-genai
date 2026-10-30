#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Token-identity gate of the Engine-hosted qwen38_flash_graph backend.

Runs the gate prompts (text / prose / random 2k, three chat prompts) as greedy turns and compares the token streams with a
reference stream of the same prompts. The Engine never delivers the stop token itself (finish reason EOS), so a stream that
ended with EOS is compared as tokens + [the stop token of the reference].

References (--reference <json>), either of
  * a file written by this script (`results[].tokens`, `.ended_with_eos` and `.spec`), e.g. a run of the same package with --no-mtp, or
    with --force-reject (every draft replaced: the no-MTP greedy stream), which makes MTP == no-MTP the self-contained gate;
  * a file of reference streams from another runner: {"cases": [{"case": "<kind>:<context>:<seed>" or "chat:<index>",
    "prompt_tokens": N, "old": {"tokens": [...incl. the stop token], "cycles": C, "accepted": A}, "new": {...}}]}.
Without --reference the streams are only recorded (the result is "RECORDED", not PASS).

--prefill steps (prefill.enabled=false, the bit-exact muted-fold prompt path) must be token-identical to the reference (and to
its cycle / accept counts when it has them: only another runner's streams do); --prefill fast is reported as agreement
statistics (identical prefix, first token), because the fast prefill is tolerance level and not run-to-run reproducible.

usage: engine_gate.py <package> --out <json> [--no-mtp] [--ignore-eos] [--cases text,prose,random,chat]
                      [--prefill steps|fast] [--reference <json>] [--force-reject]
                      [--text-pool FILE] [--prose-pool FILE]
       --ignore-eos expects a package whose eos_token_id is an ordinary id never generated (config overlays append to the
       stop list, so such a package must be built with build_package.py --stop-token-ids 248319 or
       build_package_engine_v12b.py --stop-token-id 248319).
"""

import argparse
import json
import sys
from pathlib import Path

import onnxruntime_genai as og
from qwen38_engine_util import EOS_IDS, Prompts, add_prompt_arguments, load_model, run_turn, token_hash

CHAT_QUESTIONS = [
    "What is 2 + 3? Answer with only the number.",
    "Name the three primary colors in one short sentence.",
    "Write the word hello.",
]


def _split_reference_stream(entry: dict) -> tuple[list[int], bool]:
    """Return (delivered_tokens_without_eos, ended_with_eos) for a reference stream."""
    tokens = list(entry.get("tokens", []))
    if "ended_with_eos" in entry:
        ended_with_eos = bool(entry["ended_with_eos"])
        if ended_with_eos and tokens and tokens[-1] in EOS_IDS:
            tokens = tokens[:-1]
        return tokens, ended_with_eos
    if tokens and tokens[-1] in EOS_IDS:
        return tokens[:-1], True
    return tokens, False


def load_reference(path):
    """Reference streams by case spec: {spec: {"prompt_tokens": N|None, "old": {...}, "new": {...}}}."""
    data = json.loads(Path(path).read_text())
    if "cases" in data:
        return {c["case"]: c for c in data["cases"]}
    return {
        r["spec"]: {
            "prompt_tokens": r["prompt_len"],
            # no cycle counts: a recording of another configuration (--no-mtp, --force-reject) has different ones by design
            "old": {
                "tokens": list(r["tokens"]),
                "ended_with_eos": bool(r["ended_with_eos"]),
            },
        }
        for r in data["results"]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", help="record only: the package must not stop at an EOS id")
    parser.add_argument("--cases", default="text,prose,random,chat")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument(
        "--prefill", choices=("steps", "fast"), default=None, help="overlay prefill.enabled (default: package)"
    )
    parser.add_argument("--reference", type=Path, help="reference streams (see the module documentation)")
    parser.add_argument(
        "--min-fast-prefix",
        type=int,
        default=1,
        help="minimum identical-prefix length required per case in --prefill fast mode (default: 1, i.e. first token equal)",
    )
    parser.add_argument(
        "--force-reject", action="store_true", help="test hook fold.force_reject: every draft is replaced"
    )
    parser.add_argument("--provider")
    add_prompt_arguments(parser)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    reference = load_reference(args.reference) if args.reference else {}
    graph_overlay = {}
    if args.no_mtp:
        graph_overlay["mtp"] = {"enabled": False}
    if args.prefill:
        graph_overlay["prefill"] = {"enabled": args.prefill == "fast"}
    if args.force_reject:
        graph_overlay["fold"] = {"force_reject": True}
    overlay = {"model": {"qwen38_graph": graph_overlay}} if graph_overlay else None
    model = load_model(args.model, args.provider, overlay)
    prompts = Prompts.from_args(model, args)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)

    cases = []
    for case in args.cases.split(","):
        if case == "chat":
            for index, question in enumerate(CHAT_QUESTIONS):
                cases.append((f"chat:{question[:24]}", prompts.chat_prompt(question), f"chat:{index}"))
        else:
            prompt = prompts.make(case, args.context, 1000 + args.context)
            cases.append((f"{case}{args.context}", prompt, f"{case}:{args.context}:{1000 + args.context}"))

    results, all_ok = [], True
    for name, prompt, spec in cases:
        request = engine.create_request()
        before = engine.get_speculative_stats()
        result, _ = run_turn(engine, buffer, request, prompt, args.new)
        request.close()
        after = engine.get_speculative_stats()
        tokens = result.tokens
        ended_with_eos = result.finish_reason == og.FinishReason.EOS
        entry = {
            "case": name,
            "spec": spec,
            "prompt_len": len(prompt),
            "prompt_hash": token_hash(prompt),
            "n_events": len(tokens),
            "finish_reason": str(result.finish_reason),
            "usage_generated": result.usage.generated_tokens,
            "tokens": tokens,  # without the stop token, which the Engine does not deliver
            "ended_with_eos": ended_with_eos,
            "hash": token_hash(tokens),
            "runs": result.runs,
            "tps": (len(tokens) - 1) / (result.times[-1] - result.times[0]) if len(tokens) > 1 else None,
            "cycles": after["rounds"] - before["rounds"],
            "accepted": after["draft_tokens_accepted"] - before["draft_tokens_accepted"],
            "first": tokens[0] if tokens else None,
        }
        check = True
        if spec in reference:
            key = "new" if args.prefill == "fast" and "new" in reference[spec] else "old"
            ref = reference[spec][key]
            ref_tokens, ref_ended_with_eos = _split_reference_stream(ref)
            common = 0
            while common < min(len(tokens), len(ref_tokens)) and tokens[common] == ref_tokens[common]:
                common += 1
            first_equal = (
                (tokens[0] == ref_tokens[0])
                if (tokens and ref_tokens)
                else (not tokens and not ref_tokens and ended_with_eos == ref_ended_with_eos)
            )
            entry["reference"] = {
                "key": key,
                "prompt_tokens_equal": reference[spec].get("prompt_tokens") in (None, len(prompt)),
                "identical": tokens == ref_tokens and ended_with_eos == ref_ended_with_eos,
                "ended_with_eos_equal": ended_with_eos == ref_ended_with_eos,
                "identical_prefix": common,
                "n": len(ref_tokens),
                "first_equal": first_equal,
                "cycles": ref.get("cycles"),
                "accepted": ref.get("accepted"),
                "cycles_equal": ref.get("cycles") in (None, entry["cycles"])
                and ref.get("accepted") in (None, entry["accepted"]),
            }
            if args.prefill != "fast":
                check = (
                    entry["reference"]["identical"]
                    and entry["reference"]["cycles_equal"]
                    and entry["reference"]["prompt_tokens_equal"]
                )
            else:
                required_prefix = min(max(1, args.min_fast_prefix), len(ref_tokens)) if ref_tokens else 0
                check = (
                    entry["reference"]["prompt_tokens_equal"]
                    and entry["reference"]["first_equal"]
                    and entry["reference"]["identical_prefix"] >= required_prefix
                )
        elif args.reference:
            check = False  # a reference was given but has no stream for this case
        all_ok &= check
        results.append(entry)
        print(
            json.dumps(
                {k: entry[k] for k in ("case", "n_events", "finish_reason", "tps", "cycles")}
                | ({"reference": entry["reference"]} if "reference" in entry else {})
            ),
            flush=True,
        )
    stats = engine.get_speculative_stats()
    out = {
        "all_ok": all_ok if args.reference else None,
        "mtp": not args.no_mtp,
        "prefill": args.prefill,
        "force_reject": args.force_reject,
        "ignore_eos": args.ignore_eos,
        "context": args.context,
        "new": args.new,
        "speculative_stats": {
            k: stats[k] for k in ("rounds", "draft_tokens_proposed", "draft_tokens_accepted", "tokens_emitted")
        },
        "results": results,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print("GATE", ("PASS" if all_ok else "FAIL") if args.reference else "RECORDED", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

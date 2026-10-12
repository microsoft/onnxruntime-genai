#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Token-identity gate of the Engine-hosted qwen38_flash_graph backend.

Greedy MTP(1) decoding verifies every draft with the target, so it must produce exactly the tokens of plain greedy decoding
(mtp.enabled=false) of the same package. The gate therefore runs the same prompts twice through the Engine, once per mode,
and compares the token streams. Tokens of any other greedy stack can be compared the same way (see `compare`).

  engine_gate.py collect <package> --text-file TEXT --out mtp.json
  engine_gate.py collect <package> --text-file TEXT --out plain.json --no-mtp
  engine_gate.py compare mtp.json plain.json

`collect` runs a fixed set of prompts (`--contexts` lengths cut from the text file, plus a few short chat prompts) for `--new`
tokens each and writes the Engine's tokens, finish reasons and hashes. Use a package whose stop tokens (eos_token_id) the
model does not generate to compare fixed-length streams; otherwise a stream ends at the model's own stop token, which the
Engine never delivers (finish reason EOS).

`compare` checks every case of two files for identical prompts and tokens. A reference that was produced elsewhere may be
given as the second file in the same format ({"cases": [{"case", "prompt_hash", "tokens", "finish_reason"}]}); a single
trailing stop token (one of the "eos_token_ids" of the first file) that only one side lists is ignored.
"""

import argparse
import json
import sys
from pathlib import Path

import onnxruntime_genai as og
from qwen38_engine_util import (
    add_common_arguments,
    chat_prompt,
    load_model,
    load_text_tokens,
    mtp_overlay,
    package_info,
    run_turn,
    text_prompt,
    token_hash,
)

CHAT_QUESTIONS = [
    "What is 2 + 3? Answer with only the number.",
    "Name the three primary colors in one short sentence.",
    "Write the word hello.",
]


def collect(args):
    if args.out.exists():
        raise FileExistsError(args.out)
    info = package_info(args.model)
    model = load_model(args.model, args.provider, mtp_overlay(not args.no_mtp))
    tokenizer = og.Tokenizer(model)
    text_tokens = load_text_tokens(tokenizer, args.text_file)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)

    cases = []
    for context in (int(x) for x in args.contexts.split(",") if x):
        cases.append((f"text{context}", text_prompt(text_tokens, context, 1000 + context)))
    if not args.no_chat:
        cases += [(f"chat:{question[:24]}", chat_prompt(tokenizer, question)) for question in CHAT_QUESTIONS]

    results = []
    for name, prompt in cases:
        request = engine.create_request()
        result, _ = run_turn(engine, buffer, request, prompt, args.new)
        request.close()
        tps = (len(result.tokens) - 1) / (result.times[-1] - result.times[0]) if len(result.tokens) > 1 else None
        results.append(
            {
                "case": name,
                "prompt_len": len(prompt),
                "prompt_hash": token_hash(prompt),
                "finish_reason": str(result.finish_reason),
                "tokens": result.tokens,
                "hash": token_hash(result.tokens),
                "tps": tps,
            }
        )
        print(json.dumps({k: v for k, v in results[-1].items() if k != "tokens"}), flush=True)
    stats = engine.get_speculative_stats()
    args.out.write_text(
        json.dumps(
            {
                "mtp": not args.no_mtp,
                "new": args.new,
                "eos_token_ids": info["eos_token_ids"],
                "speculative_stats": {
                    k: stats[k] for k in ("rounds", "draft_tokens_proposed", "draft_tokens_accepted", "tokens_emitted")
                },
                "cases": results,
            },
            indent=1,
        )
    )
    return 0


def same_stream(a, b, eos_ids):
    """True when the token lists are equal, ignoring one trailing stop token that only one of them lists."""
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    return len(longer) == len(shorter) + 1 and longer[:-1] == shorter and longer[-1] in eos_ids


def compare(args):
    first = json.loads(args.first.read_text())
    second = json.loads(args.second.read_text())
    eos_ids = set(first.get("eos_token_ids", []))
    other = {c["case"]: c for c in second["cases"]}
    all_ok = True
    for case in first["cases"]:
        match = other.get(case["case"])
        if match is None:
            print(f"MISSING {case['case']} in {args.second}")
            all_ok = False
            continue
        same_prompt = case["prompt_hash"] == match["prompt_hash"]
        identical = same_prompt and same_stream(case["tokens"], match["tokens"], eos_ids)
        all_ok &= identical
        print(
            f"{'OK  ' if identical else 'FAIL'} {case['case']}: {len(case['tokens'])} vs {len(match['tokens'])} tokens, "
            f"prompts {'equal' if same_prompt else 'DIFFER'}"
        )
    print("GATE", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("collect", help="run the prompts through the Engine and write the tokens")
    add_common_arguments(run)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--text-file", type=Path, required=True, help="natural text the long prompts are cut from")
    run.add_argument("--no-mtp", action="store_true", help="plain S=1 graph decode (mtp.enabled=false)")
    run.add_argument("--contexts", default="2048", help="comma separated prompt lengths in tokens (empty: none)")
    run.add_argument("--new", type=int, default=256, help="generated tokens per prompt")
    run.add_argument("--no-chat", action="store_true", help="skip the short chat prompts")
    run.set_defaults(func=collect)
    cmp_ = sub.add_parser("compare", help="compare two collected token files")
    cmp_.add_argument("first", type=Path)
    cmp_.add_argument("second", type=Path)
    cmp_.set_defaults(func=compare)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

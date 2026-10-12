#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Token-identity gate of the Engine-hosted qwen38_flash_graph backend against the research runner.

The prompts are those of CG/mtp_runner/gate_mtp.py (text/prose/random 2k, three chat prompts); the reference hashes are the
no-MTP v10all/v11 S=1 greedy hashes of the research gate (runs/s2dev_gate_c2048_n256.json). Tokens come from the Engine's
token events. The Engine never delivers the stop token itself (finish reason EOS), so the reference token list is compared as
tokens + [the model's stop token]; both stop tokens of the model are tried and the one that matches is reported.

usage: engine_gate.py <package> --out <json> [--no-mtp] [--ignore-eos] [--cases text,prose,random,chat]
       --ignore-eos selects the "fixed length" reference hashes and expects a package whose eos_token_id is an ordinary id
       never generated (config overlays append to the stop list, so such a package must be built with
       build_package.py --stop-token-ids 248319).
"""

import argparse
import json
import sys
from pathlib import Path

import onnxruntime_genai as og
from qwen38_engine_util import EOS_IDS, RUNNER_DIR, load_model, research_prompts, run_turn, token_hash

CHAT_QUESTIONS = [
    "What is 2 + 3? Answer with only the number.",
    "Name the three primary colors in one short sentence.",
    "Write the word hello.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-mtp", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--cases", default="text,prose,random,chat")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--new", type=int, default=256)
    parser.add_argument("--provider")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    prompts = research_prompts()
    reference = {
        r["case"]: r for r in json.loads((RUNNER_DIR / "runs/s2dev_gate_c2048_n256.json").read_text())["results"]
    }
    overlay = {"model": {"qwen38_graph": {"mtp": {"enabled": False}}}} if args.no_mtp else None
    model = load_model(args.model, args.provider, overlay)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)

    cases = []
    for case in args.cases.split(","):
        if case == "chat":
            for question in CHAT_QUESTIONS:
                messages = json.dumps([{"role": "user", "content": question}])
                template = (Path(args.model) / "chat_template.jinja").read_text()
                template = template.replace("enable_thinking is undefined or enable_thinking is true", "false").replace(
                    "enable_thinking is defined and enable_thinking is false", "true"
                )
                text = tokenizer.apply_chat_template(messages, template_str=template, add_generation_prompt=True)
                cases.append(
                    (f"chat:{question[:24]}", [int(x) for x in tokenizer.encode(text)], prompts.chat_prompt(question))
                )
        else:
            prompt = prompts.make(case, args.context, 1000 + args.context)
            cases.append((f"{case}{args.context}", prompt, prompt))

    results, all_ok = [], True
    for name, prompt, research_prompt in cases:
        ref = reference[name]["plain"][1 if args.ignore_eos else 0]
        request = engine.create_request()
        result, _ = run_turn(engine, buffer, request, prompt, args.new)
        request.close()
        tokens = result.tokens
        matched = None
        if result.finish_reason == og.FinishReason.EOS:
            for eos in EOS_IDS:
                if token_hash([*tokens, eos]) == ref["hash"]:
                    matched = [*tokens, eos]
        else:
            if token_hash(tokens) == ref["hash"]:
                matched = tokens
        entry = {
            "case": name,
            "prompt_matches_research": prompt == research_prompt,
            "prompt_len": len(prompt),
            "ref_n": ref["n"],
            "n_events": len(tokens),
            "finish_reason": str(result.finish_reason),
            "usage_generated": result.usage.generated_tokens,
            "tokens_identical": matched is not None,
            "ref_hash": ref["hash"],
            "hash": token_hash(tokens),
            "runs": result.runs,
            "tps": (len(tokens) - 1) / (result.times[-1] - result.times[0]) if len(tokens) > 1 else None,
        }
        all_ok &= entry["tokens_identical"] and entry["prompt_matches_research"]
        results.append(entry)
        print(
            json.dumps(
                {k: entry[k] for k in ("case", "n_events", "ref_n", "finish_reason", "tokens_identical", "tps")}
            ),
            flush=True,
        )
    stats = engine.get_speculative_stats()
    out = {
        "all_ok": all_ok,
        "mtp": not args.no_mtp,
        "ignore_eos": args.ignore_eos,
        "context": args.context,
        "new": args.new,
        "speculative_stats": {
            k: stats[k] for k in ("rounds", "draft_tokens_proposed", "draft_tokens_accepted", "tokens_emitted")
        },
        "results": results,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print("GATE", "PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

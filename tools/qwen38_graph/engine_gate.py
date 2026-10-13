#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Token-identity gate of the Engine-hosted qwen38_flash_graph backend against the research runner.

The prompts are those of CG/mtp_runner/gate_mtp.py (text/prose/random 2k, three chat prompts); the reference hashes are the
no-MTP v10all/v11 S=1 greedy hashes of the research gate (runs/s2dev_gate_c2048_n256.json). Tokens come from the Engine's
token events. The Engine never delivers the stop token itself (finish reason EOS), so the reference token list is compared as
tokens + [the model's stop token]; both stop tokens of the model are tried and the one that matches is reported.

v12b stack: the stored research hashes are obsolete (accurate-sigmoid numerics); pass --python-ref <json> (the python runner's
streams of the same prompts, one entry per case spec with "old" = muted S=2-step prefill, optionally "new" = fast prefill; the fast prefill is compared with "old" when there is no "new") and
--no-research-hash. --prefill steps (prefill.enabled=false, the bit-exact muted-fold prompt path) must then be token-identical to
the "old" stream and to its cycle / accept counts; --prefill fast is reported as agreement statistics (identical prefix, first
token), because the fast prefill is tolerance level and not run-to-run reproducible.

usage: engine_gate.py <package> --out <json> [--no-mtp] [--ignore-eos] [--cases text,prose,random,chat]
                      [--prefill steps|fast] [--python-ref <json>] [--no-research-hash] [--force-reject]
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
    parser.add_argument(
        "--prefill", choices=("steps", "fast"), default=None, help="overlay prefill.enabled (default: package)"
    )
    parser.add_argument("--python-ref", type=Path, help="python runner streams (ref_gate_{eos,fixed}.json)")
    parser.add_argument("--no-research-hash", action="store_true", help="do not require the stored research hashes")
    parser.add_argument(
        "--force-reject", action="store_true", help="test hook fold.force_reject: every draft is replaced"
    )
    parser.add_argument("--provider")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    prompts = research_prompts()
    reference = {
        r["case"]: r for r in json.loads((RUNNER_DIR / "runs/s2dev_gate_c2048_n256.json").read_text())["results"]
    }
    python_ref = {}
    if args.python_ref:
        python_ref = {c["case"]: c for c in json.loads(args.python_ref.read_text())["cases"]}
    graph_overlay = {}
    if args.no_mtp:
        graph_overlay["mtp"] = {"enabled": False}
    if args.prefill:
        graph_overlay["prefill"] = {"enabled": args.prefill == "fast"}
    if args.force_reject:
        graph_overlay["fold"] = {"force_reject": True}
    overlay = {"model": {"qwen38_graph": graph_overlay}} if graph_overlay else None
    model = load_model(args.model, args.provider, overlay)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    buffer = engine.create_event_buffer(16)

    cases = []
    for case in args.cases.split(","):
        if case == "chat":
            for index, question in enumerate(CHAT_QUESTIONS):
                messages = json.dumps([{"role": "user", "content": question}])
                template = (Path(args.model) / "chat_template.jinja").read_text()
                template = template.replace("enable_thinking is undefined or enable_thinking is true", "false").replace(
                    "enable_thinking is defined and enable_thinking is false", "true"
                )
                text = tokenizer.apply_chat_template(messages, template_str=template, add_generation_prompt=True)
                cases.append(
                    (
                        f"chat:{question[:24]}",
                        [int(x) for x in tokenizer.encode(text)],
                        prompts.chat_prompt(question),
                        f"chat:{index}",
                    )
                )
        else:
            prompt = prompts.make(case, args.context, 1000 + args.context)
            cases.append((f"{case}{args.context}", prompt, prompt, f"{case}:{args.context}:{1000 + args.context}"))

    results, all_ok = [], True
    for name, prompt, research_prompt, spec in cases:
        ref = reference[name]["plain"][1 if args.ignore_eos else 0]
        request = engine.create_request()
        before = engine.get_speculative_stats()
        result, _ = run_turn(engine, buffer, request, prompt, args.new)
        request.close()
        after = engine.get_speculative_stats()
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
        entry["cycles"] = after["rounds"] - before["rounds"]
        entry["accepted"] = after["draft_tokens_accepted"] - before["draft_tokens_accepted"]
        entry["first"] = tokens[0] if tokens else None
        check = entry["tokens_identical"] or args.no_research_hash
        if spec in python_ref:
            key = "new" if args.prefill == "fast" and "new" in python_ref[spec] else "old"
            py = python_ref[spec][key]
            full = [*tokens, py["tokens"][-1]] if result.finish_reason == og.FinishReason.EOS else tokens
            common = 0
            while common < min(len(full), len(py["tokens"])) and full[common] == py["tokens"][common]:
                common += 1
            entry["python"] = {
                "key": key,
                "prompt_tokens_equal": python_ref[spec].get("prompt_tokens") == len(prompt),
                "identical": full == py["tokens"],
                "identical_prefix": common,
                "n": len(py["tokens"]),
                "first_equal": py["tokens"][0] == entry["first"],
                "cycles": py.get("cycles"),
                "accepted": py.get("accepted"),
                "cycles_equal": py.get("cycles") == entry["cycles"] and py.get("accepted") == entry["accepted"],
            }
            if args.prefill != "fast":
                check = (
                    check
                    and entry["python"]["identical"]
                    and entry["python"]["cycles_equal"]
                    and entry["python"]["prompt_tokens_equal"]
                )
        elif args.python_ref and args.prefill != "fast":
            check = False
        if args.prefill == "fast" and args.python_ref:
            check = True  # agreement statistics only
        all_ok &= check and entry["prompt_matches_research"]
        results.append(entry)
        print(
            json.dumps(
                {
                    k: entry[k]
                    for k in ("case", "n_events", "ref_n", "finish_reason", "tokens_identical", "tps", "cycles")
                }
                | ({"python": entry["python"]} if "python" in entry else {})
            ),
            flush=True,
        )
    stats = engine.get_speculative_stats()
    out = {
        "all_ok": all_ok,
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
    print("GATE", "PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

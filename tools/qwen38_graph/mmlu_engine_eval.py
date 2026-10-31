#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""MMLU-Pro (800 items) through the GenAI Engine API (og.Engine / og.Request) on a qwen38_flash_graph package of the v12b stack:
folded MTP(1) decode loop + fast prefill behind the Engine's static single-slot path.

Same protocol as the Generator-API driver the stack was first qualified with: prompt token ids from a prompts.jsonl (rows with
index, prompt_ids, prompt_text, prompt_sha256, ideal), greedy, the package's stop ids, cap --max-new generated tokens (the stop
token counts, as in the Generator protocol: the Engine does not deliver it, so 248046 is appended to the record when the turn ended
with the stop token), text = HF tokenizer decode (skip_special_tokens=False), choice extractor of the research eval library.
Resumable (one json record per item).

The research evaluation helpers (eval_lib: extract_choice_letter, extract_final_channel, load_samples, messages_of;
eval_mcq_genai: non_thinking_template) are imported from the directories listed in $QWEN38_EVAL_PATHS (os.pathsep separated).

mode run   : --shard K --shards N (round-robin over the 800 items), --out <jsonl> --prompts <prompts.jsonl> --checkpoint <dir>
             Also verifies, with the GenAI tokenizer of the model, that the prompt token ids and text are the canonical ones of
             prompts.jsonl; any difference is written to <out>.promptcheck.json and stops the run (exit status 1) before the
             warm-up and generation.
mode check : --check-items i,j,... --check-new M --out <json> [--force-reject]: token streams of the given items with
             prefill.enabled=false (bit-exact S=2-step prompt path, so two runs start from identical state); with --force-reject
             every draft is replaced by a different token (config fold.force_reject: the no-MTP greedy stream).
             Compare the MTP run with the --force-reject run via --compare mtp.json forced.json: exit status 0 only if both
             cover the same non-empty set of items and every stream is identical, 1 (with the problems listed) otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
from qwen38_engine_util import load_model, run_turn

for _path in os.environ.get("QWEN38_EVAL_PATHS", "").split(os.pathsep):
    if _path:
        sys.path.insert(0, _path)

EOS = {248046, 248044}
EOS_APPENDED = 248046


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def load_prompts(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def open_engine(args, prefill: bool):
    overlay = {"prefill": {"enabled": prefill}}
    if args.force_reject:
        overlay["fold"] = {"force_reject": True}
    t = time.perf_counter()
    model = load_model(args.model, args.provider, {"model": {"qwen38_graph": overlay}})
    load_s = time.perf_counter() - t
    engine = og.Engine(model)
    return model, engine, engine.create_event_buffer(16), load_s


def generate(engine, buffer, prompt, max_new):
    before = engine.get_speculative_stats()
    request = engine.create_request()
    result, _ = run_turn(engine, buffer, request, prompt, max_new)
    request.close()
    after = engine.get_speculative_stats()
    tokens = list(result.tokens)
    ended_with_stop = result.finish_reason == og.FinishReason.EOS
    if ended_with_stop:
        tokens.append(EOS_APPENDED)
    decode_s = result.times[-1] - result.times[0] if result.times else 0.0
    return {
        "tokens": tokens,
        "stop_token_appended": ended_with_stop,
        "usage_ok": result.usage.prompt_tokens == len(prompt) and result.usage.generated_tokens == len(result.tokens),
        "ttft_s": result.times[0] - result.start if result.times else None,
        "decode_s": decode_s,
        "wall_s": (result.times[-1] - result.start) if result.times else 0.0,
        "cycles": int(after["rounds"] - before["rounds"]),
        "accepted": int(after["draft_tokens_accepted"] - before["draft_tokens_accepted"]),
    }


def run(args):
    from eval_lib import (  # noqa: PLC0415 - research environment only needed in this mode
        extract_choice_letter,
        extract_final_channel,
        load_samples,
        messages_of,
    )
    from eval_mcq_genai import non_thinking_template  # noqa: PLC0415 - research environment only needed in this mode
    from transformers import AutoTokenizer  # noqa: PLC0415 - research environment only needed in this mode

    rows = load_prompts(args.prompts)
    assert len(rows) == 800
    mine = rows[args.shard :: args.shards]
    done = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done[r["index"]] = r
    meta = {
        "max_new": args.max_new,
        "shard": args.shard,
        "shards": args.shards,
        "package": str(args.model),
        "provider": args.provider,
        "prompts": str(args.prompts),
        "checkpoint": str(args.checkpoint),
        "stack": "GenAI Engine qwen38_flash_graph v12b: fold MTP(1) + fast prefill",
    }
    proto = args.out.with_suffix(".protocol.json")
    if proto.exists() and json.loads(proto.read_text()) != meta:
        raise SystemExit("protocol mismatch; use a fresh --out")
    proto.write_text(json.dumps(meta, indent=1))

    hf = AutoTokenizer.from_pretrained(args.checkpoint)
    model, engine, buffer, load_s = open_engine(args, prefill=True)
    tokenizer = og.Tokenizer(model)

    # prompt identity through the GenAI tokenizer / chat template (non-thinking rewrite of the template, the established path)
    samples = dict(load_samples("mmlu_pro", 800, 0, 1))
    template = non_thinking_template(args.checkpoint)
    check = {
        "items": 0,
        "encode_equal": 0,
        "chat_template_text_equal": 0,
        "chat_template_ids_equal": 0,
        "default_template_text_equal": 0,
        "default_template_ids_equal": 0,
        "mismatch_items": [],
    }
    for row in mine:
        messages = json.dumps(messages_of(samples[row["index"]]))
        text = tokenizer.apply_chat_template(messages, template_str=template, add_generation_prompt=True)
        default = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        ids = [int(x) for x in tokenizer.encode(row["prompt_text"])]
        check["items"] += 1
        check["encode_equal"] += ids == row["prompt_ids"]
        check["chat_template_text_equal"] += text == row["prompt_text"]
        check["chat_template_ids_equal"] += [int(x) for x in tokenizer.encode(text)] == row["prompt_ids"]
        check["default_template_text_equal"] += default == row["prompt_text"]
        check["default_template_ids_equal"] += [int(x) for x in tokenizer.encode(default)] == row["prompt_ids"]
        if ids != row["prompt_ids"] or text != row["prompt_text"]:
            check["mismatch_items"].append(row["index"])
    print("prompt check", json.dumps(check), flush=True)
    check_path = args.out.with_suffix(".promptcheck.json")
    check_path.write_text(json.dumps(check, indent=1))
    if check["mismatch_items"]:
        # A stale or wrong prompts file must not yield an accuracy summary: nothing is generated or scored.
        shown = ", ".join(str(index) for index in check["mismatch_items"][:20])
        more = ", ..." if len(check["mismatch_items"]) > 20 else ""
        raise SystemExit(
            f"prompt check failed: {len(check['mismatch_items'])} of {check['items']} items differ from the canonical "
            f"prompt ids/text (indices {shown}{more}); see {check_path}"
        )

    # unrecorded warm-up item (first use of the decode path with a real prompt)
    generate(engine, buffer, mine[0]["prompt_ids"], 64)
    print(f"ready (load {load_s:.1f} s), {len(mine) - len(done)} items to do", flush=True)

    t_start = time.perf_counter()
    text_mismatch = 0
    with args.out.open("a", encoding="utf-8") as sink:
        for row in mine:
            index = row["index"]
            if index in done:
                continue
            prompt = row["prompt_ids"]
            r = generate(engine, buffer, prompt, args.max_new)
            toks = r["tokens"]
            text = hf.decode(toks, skip_special_tokens=False)
            og_text = tokenizer.decode(toks)
            text_mismatch += og_text != text
            picked = extract_choice_letter(extract_final_channel(text), "ABCDEFGHIJ")
            eos_end = toks[-1] in EOS
            rec = {
                "index": index,
                "ideal": row["ideal"],
                "picked": picked,
                "correct": picked == row["ideal"],
                "prompt_tokens": len(prompt),
                "prompt_sha256": row["prompt_sha256"],
                "generated_tokens": len(toks),
                "truncated": len(toks) == args.max_new and not eos_end,
                "eos_end": eos_end,
                "generated_sha256": token_hash(toks),
                "tokens": toks,
                "text": text,
                "og_decode_equal": og_text == text,
                "usage_ok": r["usage_ok"],
                "prefill_s": r["ttft_s"],
                "decode_s": r["decode_s"],
                "wall_s": r["wall_s"],
                "tps": (len(toks) - 1) / r["decode_s"] if len(toks) > 1 and r["decode_s"] > 0 else None,
                "cycles": r["cycles"],
                "accepted": r["accepted"],
                "accept_rate": r["accepted"] / r["cycles"] if r["cycles"] else None,
            }
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            done[index] = rec
            print(
                json.dumps(
                    {
                        k: rec[k]
                        for k in (
                            "index",
                            "picked",
                            "ideal",
                            "correct",
                            "generated_tokens",
                            "truncated",
                            "prefill_s",
                            "tps",
                            "accept_rate",
                            "wall_s",
                        )
                    }
                ),
                flush=True,
            )
    recs = [done[r["index"]] for r in mine]
    summary = {
        "n": len(recs),
        "correct": sum(r["correct"] for r in recs),
        "truncated": sum(r["truncated"] for r in recs),
        "wall_s_this_run": time.perf_counter() - t_start,
        "model_load_s": load_s,
        "og_text_mismatch_this_run": text_mismatch,
        "prompt_check": {k: v for k, v in check.items() if k != "mismatch_items"},
        **meta,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def check(args):
    rows = {r["index"]: r for r in load_prompts(args.prompts)}
    model, engine, buffer, load_s = open_engine(args, prefill=False)
    result = []
    for index in [int(x) for x in args.check_items.split(",")]:
        r = generate(engine, buffer, rows[index]["prompt_ids"], args.check_new)
        rec = {
            "index": index,
            "prompt_tokens": len(rows[index]["prompt_ids"]),
            "tokens": len(r["tokens"]),
            "sha256": token_hash(r["tokens"]),
            "cycles": r["cycles"],
            "accepted": r["accepted"],
            "accept_rate": r["accepted"] / max(1, r["cycles"]),
            "decode_s": r["decode_s"],
            "ttft_s": r["ttft_s"],
            "ended_with_eos": r["tokens"][-1] in EOS,
            "stream": r["tokens"],
        }
        result.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "stream"}), flush=True)
        args.out.write_text(
            json.dumps({"force_reject": args.force_reject, "check_new": args.check_new, "items": result})
        )


def _items_by_index(run, label, problems):
    items = run.get("items") or []
    by_index = {r["index"]: r for r in items}
    if not items:
        problems.append(f"{label} has no items")
    elif len(by_index) != len(items):
        problems.append(f"{label} lists an item index more than once")
    return by_index


def compare(a_path, b_path):
    """Compares the --check-items token streams of an MTP run (a) and a --force-reject, i.e. no-MTP greedy, run (b).

    "all_identical" is true only if both runs cover the same non-empty set of items with the same --check-new and prompt
    lengths and every stream is non-empty and identical. A missing item or any other discrepancy is listed in "problems" and
    makes the comparison fail; the report covers every item of either run.
    """
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    problems = []
    if a.get("force_reject") is not False or b.get("force_reject") is not True:
        problems.append(
            "expected an MTP run (force_reject false) as a and a --force-reject run (force_reject true) as b, got "
            f"{a.get('force_reject')} and {b.get('force_reject')}"
        )
    if a.get("check_new") != b.get("check_new"):
        problems.append(f"check_new differs: {a.get('check_new')} vs {b.get('check_new')}")
    ia, ib = _items_by_index(a, "a", problems), _items_by_index(b, "b", problems)
    out = []
    for k in sorted(set(ia) | set(ib)):
        if k not in ia or k not in ib:
            problems.append(f"item {k} is only in {'a' if k in ia else 'b'}")
            out.append({"index": k, "identical": False, "missing_from": "b" if k in ia else "a"})
            continue
        x, y = ia[k]["stream"], ib[k]["stream"]
        n = min(len(x), len(y))
        first = next((i for i in range(n) if x[i] != y[i]), n if len(x) != len(y) else None)
        if ia[k]["prompt_tokens"] != ib[k]["prompt_tokens"]:
            problems.append(f"item {k}: prompt lengths differ ({ia[k]['prompt_tokens']} vs {ib[k]['prompt_tokens']})")
        if not x or not y:
            problems.append(f"item {k}: empty token stream ({len(x)} vs {len(y)} tokens)")
        elif x != y:
            problems.append(f"item {k}: streams differ at token {first} ({len(x)} vs {len(y)} tokens)")
        out.append(
            {
                "index": k,
                "prompt_tokens": ia[k]["prompt_tokens"],
                "mtp_tokens": len(x),
                "forced_reject_tokens": len(y),
                "identical": x == y and bool(x),
                "first_diff": first,
                "mtp_accept_rate": ia[k]["accept_rate"],
                "forced_reject_accepted": ib[k]["accepted"],
                "forced_reject_cycles": ib[k]["cycles"],
                "mtp_cycles": ia[k]["cycles"],
                "mtp_decode_s": ia[k]["decode_s"],
                "forced_decode_s": ib[k]["decode_s"],
            }
        )
    report = {"all_identical": not problems, "problems": problems, "items": out}
    print(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path, nargs="?", help="package dir")
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--checkpoint", type=Path, help="HF checkpoint folder (tokenizer, chat template)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--check-items", default="")
    ap.add_argument("--check-new", type=int, default=1024)
    ap.add_argument("--force-reject", action="store_true")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    args = ap.parse_args()
    if args.compare:
        report = compare(*args.compare)
        if not report["all_identical"]:
            print(
                f"comparison failed with {len(report['problems'])} problem(s), first: {report['problems'][0]}",
                file=sys.stderr,
            )
        return 0 if report["all_identical"] else 1
    assert args.model and args.out and args.prompts
    if args.check_items:
        if args.out.exists():
            raise FileExistsError(args.out)
        check(args)
    else:
        assert args.checkpoint
        run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

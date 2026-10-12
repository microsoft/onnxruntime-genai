"""MMLU-Pro 800 through the ONNX Runtime GenAI API (og.Model / og.Generator, model type qwen38_flash_graph) on the v12b package:
MTP(1) fold decode loop + fast prefill, everything native behind append_tokens / generate_next_token.

Same protocol as ort_v12_eval.py / ort_mtp_eval.py: prompts.jsonl (token ids), greedy, EOS 248046 / 248044 (package stop ids), cap 8192 new tokens
(EOS counted), text = HF tokenizer decode (skip_special_tokens=False), extractor of eval_lib. Resumable (one json record per item).

mode run   : --shard K --shards N (round-robin over the 800 items), --out <jsonl>
             Also verifies, with the GenAI tokenizer of the model, that the prompt token ids are those of prompts.jsonl:
             og.Tokenizer.encode(prompt_text) == prompt_ids and og.Tokenizer.apply_chat_template(messages, non-thinking template_str) == prompt_text.
mode check : --check-items i,j,... --check-new M --out <json> [--force-reject]: token streams of the given items with prefill.enabled=false
             (bit-exact S=2-step prompt path, so the two runs start from identical state); with --force-reject every draft is replaced by a
             different token (config fold.force_reject, test hook of the fold loop = no-MTP greedy). Compare the two json files with --compare a.json b.json.
Launch through run_v12b.sh (cwd /tmp, GPU=<n>), see REPORT_V12B.md.
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

EVAL = Path(__file__).resolve().parent
sys.path.insert(0, "/home/tianlei/git/dev/scripts/h200_18")
sys.path.insert(0, "/home/tianlei/git/dev/scripts/qwen38_flash_next")

CKPT = "/home/tianlei/git/Qwen3.8-Flash-Next-NVFP4"
EOS = {248046, 248044}


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def load_prompts(path):
    return [json.loads(line) for line in open(path)]


def open_model(args, prefill: bool):
    og.register_execution_provider_library("CUDAExecutionProvider", args.provider)
    config = og.Config(str(args.model))
    overlay = {"prefill": {"enabled": prefill}}
    if args.force_reject:
        overlay["fold"] = {"force_reject": True}
    config.overlay(json.dumps({"model": {"qwen38_graph": overlay}}))
    t = time.perf_counter()
    model = og.Model(config)
    return model, time.perf_counter() - t


def generate(model, prompt, max_new):
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(prompt) + max_new, do_sample=False)
    t_create = time.perf_counter()
    generator = og.Generator(model, params)
    t0 = time.perf_counter()
    generator.append_tokens(prompt)
    t_append = time.perf_counter()
    tokens = []
    t_first = t_last = None
    while not generator.is_done():
        generator.generate_next_token()
        now = time.perf_counter()
        tokens.append(int(generator.get_next_tokens()[0]))
        if t_first is None:
            t_first = now
        t_last = now
    sequence = [int(x) for x in generator.get_sequence(0)][len(prompt):]
    stats = generator.get_speculative_stats()
    del generator
    decode_s = t_last - t_first if tokens else 0.0
    return {"tokens": tokens, "sequence_ok": sequence == tokens[: len(sequence)] and len(tokens) - len(sequence) <= 1,
            "create_s": t0 - t_create, "append_s": t_append - t0, "ttft_s": t_first - t0, "decode_s": decode_s,
            "wall_s": t_last - t0, "cycles": int(stats["rounds"]), "accepted": int(stats["draft_tokens_accepted"])}


def run(args):
    from eval_lib import extract_choice_letter, extract_final_channel, load_samples, messages_of
    from eval_mcq_genai import non_thinking_template
    from transformers import AutoTokenizer

    rows = load_prompts(args.prompts)
    assert len(rows) == 800
    mine = rows[args.shard::args.shards]
    done = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done[r["index"]] = r
    meta = {"max_new": args.max_new, "shard": args.shard, "shards": args.shards, "package": str(args.model),
            "provider": args.provider, "prompts": str(args.prompts), "stack": "GenAI qwen38_flash_graph v12b: fold MTP(1) + fast prefill"}
    proto = args.out.with_suffix(".protocol.json")
    if proto.exists() and json.loads(proto.read_text()) != meta:
        raise SystemExit("protocol mismatch; use a fresh --out")
    proto.write_text(json.dumps(meta, indent=1))

    hf = AutoTokenizer.from_pretrained(CKPT)
    model, load_s = open_model(args, prefill=True)
    tokenizer = og.Tokenizer(model)

    # prompt identity through the GenAI tokenizer / chat template (non-thinking rewrite of the template, the established path)
    samples = dict(load_samples("mmlu_pro", 800, 0, 1))
    template = non_thinking_template(CKPT)
    check = {"items": 0, "encode_equal": 0, "chat_template_text_equal": 0, "chat_template_ids_equal": 0, "default_template_text_equal": 0,
             "default_template_ids_equal": 0, "mismatch_items": []}
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
    args.out.with_suffix(".promptcheck.json").write_text(json.dumps(check, indent=1))

    # unrecorded warm-up item (first use of the decode path with a real prompt)
    generate(model, mine[0]["prompt_ids"], 64)
    print(f"ready (load {load_s:.1f} s), {len(mine) - len(done)} items to do", flush=True)

    t_start = time.perf_counter()
    text_mismatch = 0
    with args.out.open("a", encoding="utf-8") as sink:
        for row in mine:
            index = row["index"]
            if index in done:
                continue
            prompt = row["prompt_ids"]
            r = generate(model, prompt, args.max_new)
            toks = r["tokens"]
            text = hf.decode(toks, skip_special_tokens=False)
            og_text = tokenizer.decode(toks)
            text_mismatch += og_text != text
            picked = extract_choice_letter(extract_final_channel(text), "ABCDEFGHIJ")
            eos_end = toks[-1] in EOS
            rec = {
                "index": index, "ideal": row["ideal"], "picked": picked, "correct": picked == row["ideal"],
                "prompt_tokens": len(prompt), "prompt_sha256": row["prompt_sha256"],
                "generated_tokens": len(toks), "truncated": len(toks) == args.max_new and not eos_end,
                "eos_end": eos_end, "generated_sha256": token_hash(toks), "tokens": toks, "text": text,
                "og_decode_equal": og_text == text, "sequence_ok": r["sequence_ok"],
                "prefill_s": r["ttft_s"], "append_s": r["append_s"], "create_s": r["create_s"], "decode_s": r["decode_s"],
                "wall_s": r["wall_s"], "tps": (len(toks) - 1) / r["decode_s"] if len(toks) > 1 and r["decode_s"] > 0 else None,
                "cycles": r["cycles"], "accepted": r["accepted"],
                "accept_rate": r["accepted"] / r["cycles"] if r["cycles"] else None,
            }
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            done[index] = rec
            print(json.dumps({k: rec[k] for k in ("index", "picked", "ideal", "correct", "generated_tokens", "truncated",
                                                   "prefill_s", "tps", "accept_rate", "wall_s")}), flush=True)
    recs = [done[r["index"]] for r in mine]
    summary = {"n": len(recs), "correct": sum(r["correct"] for r in recs), "truncated": sum(r["truncated"] for r in recs),
               "wall_s_this_run": time.perf_counter() - t_start, "model_load_s": load_s, "og_text_mismatch_this_run": text_mismatch,
               "prompt_check": {k: v for k, v in check.items() if k != "mismatch_items"}, **meta}
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


def check(args):
    rows = {r["index"]: r for r in load_prompts(args.prompts)}
    model, load_s = open_model(args, prefill=False)
    result = []
    for index in [int(x) for x in args.check_items.split(",")]:
        r = generate(model, rows[index]["prompt_ids"], args.check_new)
        rec = {"index": index, "prompt_tokens": len(rows[index]["prompt_ids"]), "tokens": len(r["tokens"]),
               "sha256": token_hash(r["tokens"]), "cycles": r["cycles"], "accepted": r["accepted"],
               "accept_rate": r["accepted"] / max(1, r["cycles"]), "decode_s": r["decode_s"], "ttft_s": r["ttft_s"],
               "ended_with_eos": r["tokens"][-1] in EOS, "stream": r["tokens"]}
        result.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "stream"}), flush=True)
        args.out.write_text(json.dumps({"force_reject": args.force_reject, "check_new": args.check_new, "items": result}))


def compare(a_path, b_path):
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    assert a["check_new"] == b["check_new"]
    ia, ib = {r["index"]: r for r in a["items"]}, {r["index"]: r for r in b["items"]}
    out = []
    for k in sorted(set(ia) & set(ib)):
        x, y = ia[k]["stream"], ib[k]["stream"]
        n = min(len(x), len(y))
        first = next((i for i in range(n) if x[i] != y[i]), None)
        out.append({"index": k, "prompt_tokens": ia[k]["prompt_tokens"], "mtp_tokens": len(x), "forced_reject_tokens": len(y),
                    "identical": x == y, "first_diff": first, "mtp_accept_rate": ia[k]["accept_rate"],
                    "forced_reject_accepted": ib[k]["accepted"], "forced_reject_cycles": ib[k]["cycles"],
                    "mtp_cycles": ia[k]["cycles"], "mtp_decode_s": ia[k]["decode_s"], "forced_decode_s": ib[k]["decode_s"]})
    print(json.dumps({"all_identical": all(r["identical"] for r in out), "items": out}, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path, nargs="?", help="package dir (pkg_v12b)")
    ap.add_argument("--prompts", type=Path, default=EVAL / "prompts.jsonl")
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
        compare(*args.compare)
        return
    assert args.model and args.out
    if args.check_items:
        if args.out.exists():
            raise FileExistsError(args.out)
        check(args)
    else:
        run(args)


if __name__ == "__main__":
    main()

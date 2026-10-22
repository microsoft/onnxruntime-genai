#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""LiveCodeBench v6 (175 problems) through the GenAI Engine API (og.Engine / og.Request) on a qwen38_flash_graph package.

Uses the established Qwen3.8 LiveCodeBench protocol (prompts/lcb.jsonl rendered with eval_lib.qwen38_prompt(messages, thinking=True),
greedy decoding, package stop token ids, cap min(max_new, max_context - len(prompt_ids)), official LiveCodeBench v6 sandbox grading in
lcb-grader:latest at both 16K and 32K caps, plus paired exact McNemar comparison).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
from qwen38_engine_util import load_model, run_turn

sys.path.insert(0, "/home/tianlei/git/dev/scripts/h200_18")

EOS_APPENDED = 248046
MAX_CONTEXT = 33024
DEFAULT_PROMPTS = Path("/home/tianlei/git/eval_runs/qwen38_llamacpp_vs_ort_20260927/prompts/lcb.jsonl")
DEFAULT_SAMPLES = Path("/home/tianlei/git/eval_runs/nemotron_bolt_20261015/lcb/lcb_v6/samples.jsonl")
DEFAULT_TESTS = Path("/home/tianlei/git/eval_runs/nemotron_bolt_20261015/lcb/lcb_v6/tests")
DEFAULT_LCB_REPO = Path("/home/tianlei/git/eval_runs/nemotron_bolt_20261015/lcb/LiveCodeBench")
DEFAULT_GRADER = Path("/home/tianlei/git/dev/scripts/h200_18_b/container_grade_lcb.py")
DEFAULT_CKPT = Path("/home/tianlei/git/Qwen3.8-Flash-Next-NVFP4")


def token_hash(tokens) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype="<i4").tobytes()).hexdigest()


def extract_code(raw: str) -> str:
    """LiveCodeBench takes the first fenced block after stripping thinking."""
    text = raw.rsplit("</think>", 1)[-1]
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[0] + 1 : fences[1]])


def open_engine(model_path: Path, provider: str):
    t = time.perf_counter()
    model = load_model(model_path, provider, {"model": {"qwen38_graph": {"prefill": {"enabled": True}}}})
    load_s = time.perf_counter() - t
    engine = og.Engine(model)
    return model, engine, engine.create_event_buffer(16), load_s


def generate(engine, buffer, prompt: list[int], max_new: int):
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


def run_shard(args) -> None:
    from eval_lib import messages_of, qwen38_prompt  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    prompts = [json.loads(line) for line in args.prompts.open()]
    samples = {json.loads(line)["index"]: json.loads(line) for line in args.samples.open()}
    assert len(prompts) == 175
    mine = prompts[args.shard :: args.shards]

    done = {}
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["index"]] = r

    meta = {
        "max_new": args.max_new,
        "max_context": args.max_context,
        "shard": args.shard,
        "shards": args.shards,
        "package": str(args.model),
        "provider": args.provider,
        "prompts": str(args.prompts),
        "samples": str(args.samples),
        "checkpoint": str(args.checkpoint),
        "stack": "GenAI Engine qwen38_flash_graph: fold MTP(1) + fast prefill (LiveCodeBench v6)",
    }
    proto = args.out.with_suffix(".protocol.json")
    if proto.exists() and json.loads(proto.read_text()) != meta:
        raise SystemExit("protocol mismatch; use a fresh --out")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    proto.write_text(json.dumps(meta, indent=1))

    hf = AutoTokenizer.from_pretrained(args.checkpoint)
    model, engine, buffer, load_s = open_engine(args.model, args.provider)
    tokenizer = og.Tokenizer(model)

    check = {"items": 0, "prompt_ids_equal": 0, "mismatch_items": []}
    for row in mine:
        sample = samples[row["index"]]
        text = qwen38_prompt(messages_of(sample), True)
        ids = [int(x) for x in tokenizer.encode(text)]
        check["items"] += 1
        eq = ids == row["ids"] and token_hash(ids) == row["prompt_sha256"]
        check["prompt_ids_equal"] += int(eq)
        if not eq:
            check["mismatch_items"].append(row["index"])
    args.out.with_suffix(".promptcheck.json").write_text(json.dumps(check, indent=1))
    if check["mismatch_items"]:
        raise RuntimeError(f"prompt check failed: {check}")

    # Warm-up turn
    w_req = engine.create_request()
    run_turn(engine, buffer, w_req, mine[0]["ids"], 16)
    w_req.close()

    t_wall = time.perf_counter()
    with args.out.open("a") as sink:
        for k, row in enumerate(mine, 1):
            idx = row["index"]
            prompt_ids = row["ids"]
            if idx in done:
                if done[idx]["prompt_sha256"] != row["prompt_sha256"]:
                    raise RuntimeError(f"resume mismatch at {idx}")
                continue
            budget = min(args.max_new, args.max_context - len(prompt_ids))
            gen = generate(engine, buffer, prompt_ids, budget)
            tokens = gen["tokens"]
            text = hf.decode(tokens, skip_special_tokens=False)
            og_text = tokenizer.decode(tokens)
            code = extract_code(text)
            sample = samples[idx]
            rec = {
                "index": idx,
                "question_id": row["question_id"],
                "test_file": row["test_file"],
                "func_name": row["func_name"],
                "difficulty": sample["difficulty"],
                "prompt_tokens": len(prompt_ids),
                "prompt_sha256": row["prompt_sha256"],
                "budget_tokens": budget,
                "generated_tokens": len(tokens),
                "truncated": not gen["stop_token_appended"],
                "eos_end": gen["stop_token_appended"],
                "has_code": bool(code),
                "generated_sha256": token_hash(tokens),
                "tokens": tokens,
                "text": text,
                "og_decode_equal": og_text == text,
                "usage_ok": gen["usage_ok"],
                "prefill_s": gen["ttft_s"],
                "decode_s": gen["decode_s"],
                "wall_s": gen["wall_s"],
                "tps": (len(tokens) - 1) / gen["decode_s"] if gen["decode_s"] > 0 and len(tokens) > 1 else 0.0,
                "cycles": gen["cycles"],
                "accepted": gen["accepted"],
                "accept_rate": gen["accepted"] / gen["cycles"] if gen["cycles"] else 0.0,
            }
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            done[idx] = rec
            print(
                f"[{k}/{len(mine)}] idx={idx} diff={sample['difficulty']} prompt={len(prompt_ids)} "
                f"gen={len(tokens)}/{budget} trunc={rec['truncated']} code={rec['has_code']} "
                f"ttft={rec['prefill_s']:.3f}s tps={rec['tps']:.1f} acc={rec['accept_rate']:.3f}",
                flush=True,
            )

    recs = [done[r["index"]] for r in mine]
    summary = {
        "n": len(recs),
        "truncated": sum(r["truncated"] for r in recs),
        "has_code": sum(r["has_code"] for r in recs),
        "wall_s_this_run": time.perf_counter() - t_wall,
        "model_load_s": load_s,
        "prompt_check": check,
        **meta,
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print("DONE", json.dumps(summary), flush=True)


def _quote(v: str) -> str:
    return "'" + str(v).replace("'", "'\\''") + "'"


def grade_records_in_docker(
    items: list[dict],
    tests_dir: Path,
    lcb_repo: Path,
    grader_py: Path,
    workers: int = 16,
    timeout: int = 6,
) -> dict[tuple[str, int], dict]:
    with tempfile.TemporaryDirectory(prefix="lcb_grade_") as tmp:
        work = Path(tmp)
        shutil.copytree(tests_dir, work / "tests")
        (work / "payload.json").write_text(json.dumps({"items": items, "workers": workers, "timeout": timeout}))
        work.chmod(0o777)
        for p in work.rglob("*"):
            p.chmod(stat.S_IMODE(p.stat().st_mode) | (0o755 if p.is_dir() else 0o444))
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "32g",
            "--pids-limit",
            "4096",
            "-v",
            f"{lcb_repo.resolve()}:/lcb:ro",
            "-v",
            f"{grader_py.resolve()}:/grade.py:ro",
            "-v",
            f"{work}:/work",
            "lcb-grader:latest",
            "python",
            "/grade.py",
        ]
        completed = subprocess.run(["sg", "docker", "-c", " ".join(map(_quote, cmd))], capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Grading container failed:\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")
        results = json.loads((work / "results.json").read_text())
    out = {}
    for v in results["verdicts"]:
        out[(v["arm"], int(v["key"]))] = v
    return out


def grade_suite(args) -> None:
    from scipy.stats import binomtest  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    hf = AutoTokenizer.from_pretrained(args.checkpoint)
    samples = {json.loads(line)["index"]: json.loads(line) for line in args.samples.open()}
    caps = sorted(args.caps)

    arms_records: dict[str, dict[int, dict]] = {}
    for spec in args.arms:
        name, pattern = spec.split("=", 1)
        recs = {}
        for p in sorted(Path("/").glob(pattern.lstrip("/")) if pattern.startswith("/") else Path(".").glob(pattern)):
            for line in p.open():
                if line.strip():
                    r = json.loads(line)
                    recs[r["index"]] = r
        if len(recs) != 175:
            raise RuntimeError(f"Arm {name} has {len(recs)} records (expected 175) from {pattern}")
        arms_records[name] = recs

    items = []
    trunc_flags: dict[tuple[str, int, int], bool] = {}
    for name, recs in arms_records.items():
        for cap in caps:
            arm_cap = f"{name}@{cap}"
            for idx, r in sorted(recs.items()):
                if r["generated_tokens"] <= cap:
                    raw = r["text"]
                    is_trunc = r["truncated"]
                else:
                    raw = hf.decode(r["tokens"][:cap], skip_special_tokens=False)
                    is_trunc = True
                trunc_flags[(name, cap, idx)] = is_trunc
                items.append(
                    {
                        "key": str(idx),
                        "arm": arm_cap,
                        "question_id": r["question_id"],
                        "test_file": r["test_file"],
                        "code": extract_code(raw),
                    }
                )

    verdicts = grade_records_in_docker(items, args.tests, args.lcb_repo, args.grader, args.workers, args.timeout)

    report = {"caps": {}, "performance": {}, "paired": {}}
    for name, recs in arms_records.items():
        tps_list = [r["tps"] for r in recs.values() if r["tps"] > 0]
        acc_list = [r["accept_rate"] for r in recs.values() if r["cycles"] > 0]
        ttft_list = [r["prefill_s"] for r in recs.values() if r["prefill_s"] is not None]
        gen_list = [r["generated_tokens"] for r in recs.values()]
        report["performance"][name] = {
            "total_generated_tokens": sum(gen_list),
            "generated_tokens_median": statistics.median(gen_list),
            "generated_tokens_mean": statistics.mean(gen_list),
            "decode_tps_median": statistics.median(tps_list),
            "decode_tps_mean": statistics.mean(tps_list),
            "acceptance_rate_mean": statistics.mean(acc_list),
            "prefill_s_median": statistics.median(ttft_list),
            "prefill_s_mean": statistics.mean(ttft_list),
            "usage_ok": sum(bool(r["usage_ok"]) for r in recs.values()),
        }

    for cap in caps:
        cap_entry = {}
        for name, recs in arms_records.items():
            arm_cap = f"{name}@{cap}"
            by_diff = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}
            correct = 0
            no_code = 0
            truncated = 0
            for idx in sorted(recs):
                v = verdicts[(arm_cap, idx)]
                d = samples[idx]["difficulty"]
                by_diff[d][1] += 1
                by_diff[d][0] += int(v["correct"])
                correct += int(v["correct"])
                no_code += int(not v["extracted_code"])
                truncated += int(trunc_flags[(name, cap, idx)])
            cap_entry[name] = {
                "n": len(recs),
                "correct": correct,
                "pass_at_1_pct": 100.0 * correct / len(recs),
                "no_code": no_code,
                "truncated": truncated,
                "by_difficulty": {k: {"solved": v[0], "total": v[1], "pct": 100.0 * v[0] / v[1]} for k, v in by_diff.items()},
            }
        report["caps"][str(cap)] = cap_entry

        arm_names = list(arms_records.keys())
        if len(arm_names) >= 2:
            ref_name = arm_names[0]
            for alt_name in arm_names[1:]:
                wins = sum(
                    (not verdicts[(f"{ref_name}@{cap}", idx)]["correct"])
                    and verdicts[(f"{alt_name}@{cap}", idx)]["correct"]
                    for idx in range(175)
                )
                losses = sum(
                    verdicts[(f"{ref_name}@{cap}", idx)]["correct"]
                    and (not verdicts[(f"{alt_name}@{cap}", idx)]["correct"])
                    for idx in range(175)
                )
                disc = wins + losses
                report["paired"][f"{alt_name}_vs_{ref_name}@{cap}"] = {
                    "wins": wins,
                    "losses": losses,
                    "delta_pp": 100.0 * (wins - losses) / 175.0,
                    "mcnemar_p": float(binomtest(wins, disc).pvalue) if disc else 1.0,
                }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grade", action="store_true", help="grade completed JSONL shards in Docker sandbox")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--provider", default=os.environ.get("QWEN_PROVIDER"))
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--tests", type=Path, default=DEFAULT_TESTS)
    parser.add_argument("--lcb-repo", type=Path, default=DEFAULT_LCB_REPO)
    parser.add_argument("--grader", type=Path, default=DEFAULT_GRADER)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=32768)
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", help="name=glob pairs for --grade")
    parser.add_argument("--caps", type=int, nargs="+", default=[16384, 32768])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()
    if args.grade:
        grade_suite(args)
    else:
        if not args.model:
            parser.error("--model is required when not using --grade")
        run_shard(args)


if __name__ == "__main__":
    main()

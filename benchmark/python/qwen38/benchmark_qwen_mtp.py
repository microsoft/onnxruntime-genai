# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_genai as og


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "mtp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--capture-decode-tokens", type=int, default=0)
    parser.add_argument("--capture-prefill", action="store_true")
    parser.add_argument("--capture-each-decode", action="store_true")
    parser.add_argument("--step-timings", action="store_true")
    parser.add_argument("--memory-markers", action="store_true")
    parser.add_argument("--cuda-runtime", default="libcudart.so")
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 1:
        parser.error("--warmup and --repetitions must be positive")
    if not args.model.is_dir():
        parser.error("--model must name an existing model directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.capture_each_decode and (not args.capture_decode_tokens or args.capture_prefill):
        parser.error("--capture-each-decode requires --capture-decode-tokens and excludes --capture-prefill")
    model_path = args.model
    config = og.Config(str(model_path))
    mode_overlay = {"model": {"mtp": {"enabled": args.mode == "mtp"}}}
    config.overlay(json.dumps(mode_overlay))
    overlay = json.loads(args.overlay.read_text()) if args.overlay else {}
    if overlay:
        config.overlay(json.dumps(overlay))
    cudart = None
    if args.capture_decode_tokens or args.capture_prefill:
        cudart = ctypes.CDLL(args.cuda_runtime)
    start = time.perf_counter()
    if args.memory_markers:
        print("MEMORY_PHASE loading", flush=True)
    print(f"Loading {args.mode} model...", flush=True)
    model = og.Model(config)
    load_seconds = time.perf_counter() - start
    tokenizer = og.Tokenizer(model)
    text = (
        "You are reviewing a distributed inference service. The service receives requests, "
        "tokenizes their input, schedules GPU work, and streams generated responses. "
        "Explain how batching, memory allocation, caching, and fault recovery interact. "
        "Consider both throughput and latency. Each worker records queue depth, request "
        "duration, and memory consumption. Backpressure prevents overload, while retries "
        "must preserve correctness and avoid duplicate work. Discuss practical tradeoffs "
        "and give concrete implementation examples.\n"
    )
    encoded = tokenizer.encode(text * 256)
    if len(encoded) < 8192:
        raise RuntimeError(f"Insufficient prompt tokens: {len(encoded)}")
    tokens = np.asarray(encoded[:8192], dtype=np.int32)
    print(f"Loaded in {load_seconds:.2f}s; input_tokens={len(tokens)}", flush=True)
    if args.memory_markers:
        print("MEMORY_PHASE loaded", flush=True)
    results = {
        "mode": args.mode,
        "attention_tensor_core_flags": {
            name: os.environ.get(name, "0")
            for name in ("ORT_SPARSE_PREFILL_TENSOR_CORE_QK", "ORT_SPARSE_PREFILL_TENSOR_CORE_PV")
        },
        "model": str(model_path),
        "ort_version": ort.__version__,
        "genai_distributions": {
            dist.metadata["Name"]: dist.version
            for dist in importlib.metadata.distributions()
            if "onnxruntime" in dist.metadata.get("Name", "").lower()
        },
        "prompt_tokens": len(tokens),
        "prompt_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "output_tokens": 512,
        "batch_size": 1,
        "do_sample": False,
        "overlay": overlay,
        "mode_overlay": mode_overlay,
        "max_draft_tokens": overlay.get("speculative", {}).get("max_draft_tokens", 7)
        if args.mode == "mtp" else 0,
        "model_load_seconds": load_seconds,
        "warmup_runs": args.warmup,
        "measured_runs": args.repetitions,
        "runs": [],
    }
    for run in range(args.warmup + args.repetitions):
        if args.memory_markers:
            print("MEMORY_PHASE " + ("warmup" if run < args.warmup else "measured") + "_prefill", flush=True)
        engine = og.Engine(model)
        buffer = engine.create_event_buffer(16)
        request_options = og.RequestOptions()
        request_options.set_max_session_tokens(8192 + 512)
        request = engine.create_request(options=request_options)
        turn = og.TurnOptions(request)
        turn.set_do_sample(False)
        turn.set_min_generated_tokens(512)
        turn.set_max_generated_tokens(512)
        output_tokens = []
        first_time = None
        last_time = None
        start = time.perf_counter()
        request.begin_turn(tokens, turn)
        finished = False
        steps = 0
        capture_started = False
        capture_stopped = False
        step_timings = []
        run_start_ns = time.perf_counter_ns()
        run_start_unix_ns = time.time_ns()
        if cudart is not None and args.capture_prefill and run == args.warmup:
            if cudart.cudaProfilerStart() != 0:
                raise RuntimeError("cudaProfilerStart failed")
            capture_started = True
        while engine.has_pending_requests():
            step_start_ns = time.perf_counter_ns() if args.step_timings else 0
            previous_tokens = len(output_tokens)
            events = engine.run(buffer)
            now = time.perf_counter()
            for event in events:
                if event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED):
                    raise RuntimeError(f"Engine failure: flags={event.flags}, error={event.error_code}")
                if event.flags & og.EngineEventFlags.TOKEN:
                    output_tokens.append(event.token)
                    if first_time is None:
                        first_time = now
                        if args.memory_markers:
                            print("MEMORY_PHASE " + ("warmup" if run < args.warmup else "measured") + "_decode", flush=True)
                    last_time = now
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    finished = True
            steps += 1
            if args.step_timings:
                stats_now = engine.get_speculative_stats()
                step_timings.append({
                    "step": steps,
                    "start_ns": step_start_ns,
                    "engine_end_ns": int(now * 1e9),
                    "observed_end_ns": time.perf_counter_ns(),
                    "tokens_before": previous_tokens,
                    "tokens_after": len(output_tokens),
                    "stats": {key: stats_now[key] for key in (
                        "rounds", "draft_tokens_proposed", "draft_tokens_evaluated", "draft_tokens_accepted",
                        "draft_forward_passes", "target_forward_passes", "standard_fallback_steps", "mtp_failures")},
                })
            capture_run = run >= args.warmup if args.capture_each_decode else run == args.warmup
            if cudart is not None and capture_run and first_time is not None:
                if args.capture_prefill and not capture_stopped:
                    if cudart.cudaProfilerStop() != 0:
                        raise RuntimeError("cudaProfilerStop failed")
                    capture_stopped = True
                elif not capture_started:
                    if cudart.cudaProfilerStart() != 0:
                        raise RuntimeError("cudaProfilerStart failed")
                    capture_started = True
                elif not capture_stopped and len(output_tokens) >= args.capture_decode_tokens:
                    if cudart.cudaProfilerStop() != 0:
                        raise RuntimeError("cudaProfilerStop failed")
                    capture_stopped = True
            if steps > 20000 or now - start > 600:
                raise RuntimeError("Benchmark exceeded iteration or time limit")
        end = time.perf_counter()
        if not finished or len(output_tokens) != 512 or first_time is None or last_time is None:
            raise RuntimeError(f"Expected completed 512-token turn; got {len(output_tokens)}, finished={finished}")
        stats = engine.get_speculative_stats()
        if args.mode == "mtp":
            if stats.get("mtp_failures", 0) != 0 or stats.get("draft_tokens_proposed", 0) == 0:
                raise RuntimeError(f"MTP did not execute successfully: {stats}")
        row = {
            "warmup": run < args.warmup,
            "output_tokens": len(output_tokens),
            "ttft_seconds": first_time - start,
            "decode_seconds": last_time - first_time,
            "decode_tps": 511 / (last_time - first_time),
            "end_to_end_seconds": end - start,
            "end_to_end_tps": 512 / (end - start),
            "engine_steps": steps,
            "speculative_stats": stats,
            "output_sha256": hashlib.sha256(np.asarray(output_tokens, dtype=np.int32).tobytes()).hexdigest(),
            "token_ids": output_tokens,
            "output_preview": tokenizer.decode(output_tokens[:64]),
        }
        if args.step_timings:
            row["run_start_ns"] = run_start_ns
            row["run_start_unix_ns"] = run_start_unix_ns
            row["step_timings"] = step_timings
        results["runs"].append(row)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({key: value for key, value in row.items()
                          if key not in {"speculative_stats", "token_ids", "step_timings"}}), flush=True)
        request.close()
        del event, events, turn, request, request_options, buffer, engine
        if args.memory_markers:
            print("MEMORY_PHASE between_requests", flush=True)
    measured = results["runs"][args.warmup:]
    results["summary"] = {
        "decode_tps": sum(row["output_tokens"] - 1 for row in measured)
        / sum(row["decode_seconds"] for row in measured),
        "end_to_end_tps": sum(row["output_tokens"] for row in measured)
        / sum(row["end_to_end_seconds"] for row in measured),
        "mean_ttft_seconds": float(np.mean([row["ttft_seconds"] for row in measured])),
        "mean_end_to_end_seconds": float(np.mean([row["end_to_end_seconds"] for row in measured])),
        "decode_tps_min": min(row["decode_tps"] for row in measured),
        "decode_tps_max": max(row["decode_tps"] for row in measured),
    }
    results["loaded_runtime_libraries"] = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libonnxruntime" in line and "/" in line
        }
    )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print("SUMMARY " + json.dumps(results["summary"]), flush=True)


if __name__ == "__main__":
    main()

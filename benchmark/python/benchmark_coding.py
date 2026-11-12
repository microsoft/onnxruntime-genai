# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Reproducible, single-request Engine coding qualification from pinned token IDs.

No generated code is executed. Imports of the native runtime occur only in main,
so the protocol and validation tests need no runtime, model, or GPU.
"""

import argparse
import hashlib
import importlib
import json
import platform
import time
from pathlib import Path


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_prompt(path):
    ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not ids:
        raise ValueError("Prompt must be a nonempty JSON array of token IDs")
    if any(type(token) is not int or not 0 <= token <= 2147483647 for token in ids):
        raise ValueError("Prompt token IDs must be nonnegative int32 integers")
    return ids


def stats_delta(before, after):
    # effective_k and cooldown gauges are not additive counters. Keep the complete
    # snapshots too; do not subtract ratios, means, or throughput estimates.
    gauges = {"effective_k", "cooldown_remaining", "active_rounds", "tokens_buffered"}
    result = {
        key: value - before[key]
        for key, value in after.items()
        if type(value) is int and key not in gauges and type(before.get(key)) is int
    }
    for key, value in after.items():
        if key.startswith("total_") and key.endswith("_ms") and key in before:
            result[key] = value - before[key]
    proposed = result.get("draft_tokens_evaluated", 0)
    result["acceptance_rate"] = result.get("draft_tokens_accepted", 0) / proposed if proposed else None
    return result


def run_request(og, engine, prompt, output_cap, clock=time.perf_counter):
    options = og.RequestOptions()
    options.set_max_session_tokens(len(prompt) + output_cap)
    before = engine.get_speculative_stats()
    request = engine.create_request(options=options)
    tokens, timestamps = [], []
    finish_reason = None
    calls = 0
    try:
        turn = og.TurnOptions(request)
        turn.set_do_sample(False)
        turn.set_max_generated_tokens(output_cap)
        # Capacity one preserves per-token host event pacing. This is not a GPU
        # kernel timestamp; speculative tokens may already be buffered together.
        events = engine.create_event_buffer(1)
        started = clock()
        request.begin_turn(prompt, turn)
        while engine.has_pending_requests():
            calls += 1
            for event in engine.run(events):
                # Shared-batch outcomes may have no request. Preserve the native
                # failure category and error code before checking token ownership.
                if event.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(f"Engine failed with error_code={event.error_code}")
                if event.flags & og.EngineEventFlags.CAPACITY_BLOCKED:
                    raise RuntimeError(f"Engine capacity-blocked with error_code={event.error_code}")
                if event.flags & og.EngineEventFlags.RETRYABLE:
                    raise RuntimeError(f"Engine retryable abort with error_code={event.error_code}")
                if event.request is not request:
                    raise RuntimeError("Engine returned an event for another request")
                if event.flags & og.EngineEventFlags.TOKEN:
                    tokens.append(int(event.token))
                    timestamps.append(clock() - started)
                    if len(tokens) > output_cap:
                        raise RuntimeError("Engine exceeded the output token cap")
                if event.flags & og.EngineEventFlags.TURN_FINISHED:
                    finish_reason = str(event.finish_reason)
        elapsed = clock() - started
        after = engine.get_speculative_stats()
    finally:
        request.close()
    if not tokens or finish_reason is None:
        raise RuntimeError("Engine ended without tokens and a turn-finished event")
    decode_seconds = timestamps[-1] - timestamps[0]
    return {
        "tokens": tokens,
        "generated_tokens": len(tokens),
        "finish_reason": finish_reason,
        "output_token_cap": output_cap,
        "host_token_seconds": timestamps,
        "ttft_seconds": timestamps[0],
        "request_seconds": elapsed,
        "decode_tokens_per_second": (len(tokens) - 1) / decode_seconds if decode_seconds > 0 else None,
        "engine_calls": calls,
        "stats_before": before,
        "stats_after": after,
        "stats_delta": stats_delta(before, after),
    }


def qualify(result, reference):
    for key in ("schema_version", "prompt_sha256", "prompt_tokens", "output_token_cap"):
        if result[key] != reference[key]:
            raise ValueError(f"Reference differs in {key}")
    expected = reference["first_request"]["tokens"]
    reference_runs = [reference["first_request"], *reference["warm_process_requests"]]
    actual_runs = [result["first_request"], *result["warm_process_requests"]]
    matches = all(run["tokens"] == expected for run in [*reference_runs, *actual_runs])
    return {"token_ids_equal": matches, "reference_run_count": len(reference_runs)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument(
        "--provider", default="cpu", help="GenAI provider name, e.g. cpu, cuda, webgpu, MLXExecutionProvider"
    )
    parser.add_argument("--plugin-library", type=Path, help="Explicit plugin to register for --provider")
    parser.add_argument(
        "--runtime-artifact",
        type=Path,
        action="append",
        required=True,
        help="Loaded GenAI/core native library; repeat for each artifact to fingerprint",
    )
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1, help="Extra unmeasured requests after the first request")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reference", type=Path, help="Compare deterministic token IDs with another result")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.new_tokens <= 2048:
        parser.error("--new-tokens must be 1..2048")
    if args.chunk <= 0 or args.num_blocks <= 0 or args.repeat <= 0 or args.warmup < 0:
        parser.error("chunk, num-blocks, repeat must be positive; warmup must be nonnegative")
    if args.output.resolve().is_relative_to(args.model.resolve()):
        parser.error("--output must be outside the model directory")
    if args.output.exists():
        parser.error("--output already exists; preserve the previous evidence")
    return args


def main(argv=None):
    args = parse_args(argv)
    prompt = read_prompt(args.prompt_ids)
    model_config = json.loads((args.model / "genai_config.json").read_text(encoding="utf-8"))
    if model_config.get("runtime_profiles"):
        raise ValueError("Materialize a profile-free model config; runtime profiles can override benchmark policy")
    if not isinstance(model_config.get("engine", {}).get("dynamic_batching"), dict):
        raise ValueError("Benchmark requires an exported dynamic-batching Engine model")
    files = sorted(path for path in args.model.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Model directory is empty")
    # Hashing is outside every load/request timer. Include external weights,
    # tokenizers and config, rather than only the small graph protobuf.
    model_files = {str(path.relative_to(args.model)): file_record(path) for path in files}
    runtime_files = {str(path.resolve()): file_record(path) for path in args.runtime_artifact}
    if args.plugin_library:
        runtime_files[str(args.plugin_library.resolve())] = file_record(args.plugin_library)
    og = importlib.import_module("onnxruntime_genai")
    if args.plugin_library:
        og.register_execution_provider_library(args.provider, str(args.plugin_library.resolve()))
    config = og.Config(str(args.model))
    config.clear_providers()
    config.append_provider(args.provider)
    overlay = {
        "engine": {
            "dynamic_batching": {
                "prefix_caching": False,
                "max_batch_size": 1,
                "max_scheduled_tokens": args.chunk,
                "num_blocks": args.num_blocks,
            }
        },
        "search": {"chunk_size": args.chunk},
    }
    config.overlay(json.dumps(overlay))
    started = time.perf_counter()
    model = og.Model(config)
    load_seconds = time.perf_counter() - started
    engine = og.Engine(model)
    result = {
        "schema_version": 1,
        "provider": args.provider,
        "platform": platform.platform(),
        "runtime_version": getattr(og, "__version__", None),
        "runtime_module": str(Path(og.__file__).resolve()),
        "runtime_files": runtime_files,
        "model_files": model_files,
        "model_sha256": json_digest(model_files),
        "prompt_sha256": json_digest(prompt),
        "prompt_tokens": len(prompt),
        "output_token_cap": args.new_tokens,
        "overlay": overlay,
        "prefix_caching": False,
        "model_load_seconds": load_seconds,
        "warmup_requests": args.warmup,
    }
    result["first_request"] = run_request(og, engine, prompt, args.new_tokens)
    for _ in range(args.warmup):
        run_request(og, engine, prompt, args.new_tokens)
    result["warm_process_requests"] = [run_request(og, engine, prompt, args.new_tokens) for _ in range(args.repeat)]
    result["within_process_token_ids_equal"] = all(
        run["tokens"] == result["first_request"]["tokens"] for run in result["warm_process_requests"]
    )
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        result["qualification"] = qualify(result, reference)
        result["reference_sha256"] = file_record(args.reference)["sha256"]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Recorded first request and {args.repeat} warm-process requests in {args.output}")
    if not result["within_process_token_ids_equal"]:
        raise SystemExit("Deterministic requests produced different token IDs")
    if args.reference and not result["qualification"]["token_ids_equal"]:
        raise SystemExit("Reference token IDs differ")


if __name__ == "__main__":
    main()

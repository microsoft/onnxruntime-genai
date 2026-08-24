# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "temperature_f": {"type": "integer"},
        "conditions": {"type": "string"},
    },
    "required": ["city", "temperature_f", "conditions"],
    "additionalProperties": False,
    "x-guidance": {
        "whitespace_flexible": False,
        "key_separator": ": ",
        "item_separator": ", ",
    },
}


def percentile(values, percentile_value):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def generate(engine, request, tokenizer, request_started_at):
    stream = tokenizer.create_stream()
    fragments = []
    token_count = 0
    first_token_at = None
    last_token_at = None
    started = time.perf_counter()

    while engine.has_pending_requests():
        ready_request = engine.run()
        if ready_request is None:
            raise RuntimeError("Engine returned no request while work remained")
        if ready_request is not request:
            raise RuntimeError("Engine returned an unknown request")
        while ready_request.has_unseen_tokens():
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            last_token_at = now
            token_count += 1
            fragments.append(stream.decode(ready_request.get_unseen_token()))

    completed = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("Request completed without producing a token")

    decode_seconds = max(0.0, last_token_at - first_token_at)
    return "".join(fragments).strip(), {
        "tokens": token_count,
        "ttft_after_admission_ms": (first_token_at - started) * 1000,
        "end_to_end_ttft_ms": (first_token_at - request_started_at) * 1000,
        "generation_after_admission_ms": (completed - started) * 1000,
        "end_to_end_latency_ms": (completed - request_started_at) * 1000,
        "inter_token_ms": decode_seconds * 1000 / max(1, token_count - 1),
        "decode_tokens_per_second": (token_count - 1) / decode_seconds if decode_seconds > 0 else 0.0,
    }


def validate_output(output):
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return False, False

    schema_compliant = (
        isinstance(parsed, dict)
        and set(parsed) == {"city", "temperature_f", "conditions"}
        and isinstance(parsed["city"], str)
        and isinstance(parsed["temperature_f"], int)
        and not isinstance(parsed["temperature_f"], bool)
        and isinstance(parsed["conditions"], str)
    )
    return True, schema_compliant


def run_once(engine, model, tokenizer, prompt_tokens, max_length, guided):
    setup_started = time.perf_counter()
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=max_length)
    if guided:
        params.set_guidance("lark_grammar", f"start: %json {json.dumps(SCHEMA)}\n")
    request = engine.create_request(params)
    setup_ms = (time.perf_counter() - setup_started) * 1000

    try:
        admission_started = time.perf_counter()
        request.begin_turn(np.asarray(prompt_tokens, dtype=np.int32))
        admission_ms = (time.perf_counter() - admission_started) * 1000
        output, metrics = generate(engine, request, tokenizer, setup_started)
    finally:
        request.close()

    metrics["setup_ms"] = setup_ms
    metrics["admission_ms"] = admission_ms
    metrics["parseable_json"], metrics["schema_compliant"] = validate_output(output)
    metrics["output"] = output
    return metrics


def aggregate(runs):
    metric_names = [
        "setup_ms",
        "admission_ms",
        "ttft_after_admission_ms",
        "end_to_end_ttft_ms",
        "generation_after_admission_ms",
        "end_to_end_latency_ms",
        "inter_token_ms",
        "decode_tokens_per_second",
    ]
    result = {name: summarize([run[name] for run in runs]) for name in metric_names}
    result["iterations"] = len(runs)
    result["tokens"] = summarize([run["tokens"] for run in runs])
    result["parseable_json_runs"] = sum(run["parseable_json"] for run in runs)
    result["schema_compliant_runs"] = sum(run["schema_compliant"] for run in runs)
    result["runs"] = runs
    return result


def overhead_percent(guided, unguided, metric):
    baseline = unguided[metric]["p50"]
    if baseline == 0:
        return None
    return (guided[metric]["p50"] - baseline) * 100 / baseline


def run(args):
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        config.append_provider(args.execution_provider)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)

    messages = [
        {"role": "system", "content": "Return only a JSON object matching the requested shape."},
        {
            "role": "user",
            "content": (
                "Report this weather observation as JSON with city, integer temperature_f, and conditions: "
                "Redmond, 52 F, partly cloudy."
            ),
        },
    ]
    prompt = tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True)
    prompt_tokens = list(tokenizer.encode(prompt))
    max_length = len(prompt_tokens) + args.max_new_tokens

    modes = [False, True] if args.mode == "both" else [args.mode == "on"]
    for _ in range(args.warmup):
        for guided in modes:
            run_once(engine, model, tokenizer, prompt_tokens, max_length, guided)

    runs = {"unguided": [], "guided": []}
    for _ in range(args.iterations):
        for guided in modes:
            name = "guided" if guided else "unguided"
            runs[name].append(run_once(engine, model, tokenizer, prompt_tokens, max_length, guided))

    summary = {
        "execution_provider": args.execution_provider,
        "prompt_tokens": len(prompt_tokens),
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
    }
    for name, mode_runs in runs.items():
        if mode_runs:
            summary[name] = aggregate(mode_runs)

    if runs["guided"] and runs["unguided"]:
        guided_summary = summary["guided"]
        unguided_summary = summary["unguided"]
        summary["guided_change_percent_p50"] = {
            metric: overhead_percent(guided_summary, unguided_summary, metric)
            for metric in ["setup_ms", "end_to_end_ttft_ms", "inter_token_ms"]
        }
        baseline_throughput = unguided_summary["decode_tokens_per_second"]["p50"]
        guided_throughput = guided_summary["decode_tokens_per_second"]["p50"]
        summary["guided_decode_throughput_change_percent_p50"] = (
            (guided_throughput - baseline_throughput) * 100 / baseline_throughput
            if baseline_throughput > 0
            else None
        )

    output = json.dumps(summary, indent=2)
    print(output)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile guided versus unguided Engine generation")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-e", "--execution_provider", required=True, choices=["cpu", "cuda", "dml", "webgpu"])
    parser.add_argument("--mode", choices=["off", "on", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--output", help="Optional path for the JSON summary")
    run(parser.parse_args())

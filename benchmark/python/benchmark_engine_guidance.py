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

OPEN_REGEX = r"[A-Za-z ]*"

TOOL_NAMES = [
    "view",
    "rg",
    "glob",
    "bash",
    "apply_patch",
    "read_agent",
    "write_agent",
    "web_fetch",
    "web_search",
    "github_file",
    "github_code_search",
    "sql",
    "run_tests",
    "compile",
    "list_directory",
    "create_file",
]

TOOL_CATALOG_SCHEMA = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "name": {"const": name},
                "arguments": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "query": {"type": "string"},
                        "count": {"type": "integer"},
                        "recursive": {"type": "boolean"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for name in TOOL_NAMES
    ]
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
    summary = {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }
    if len(values) > 1:
        margin = 1.96 * statistics.stdev(values) / len(values) ** 0.5
        summary["mean_ci95_low"] = summary["mean"] - margin
        summary["mean_ci95_high"] = summary["mean"] + margin
    else:
        summary["mean_ci95_low"] = summary["mean"]
        summary["mean_ci95_high"] = summary["mean"]
    return summary


def generate(engine, requests, trackers, tokenizer, request_started_at):
    streams = [tokenizer.create_stream() for _ in requests]
    fragments = [[] for _ in requests]
    token_counts = [0 for _ in requests]
    first_token_at = [None for _ in requests]
    last_token_at = [None for _ in requests]
    inter_token_gaps = [[] for _ in requests]
    completion_notifications = set()
    started = time.perf_counter()

    while len(completion_notifications) != len(requests):
        ready_request = engine.step()
        if ready_request is None:
            raise RuntimeError("Engine stopped before all requests completed")
        ready_index = ready_request.get_opaque_data()["index"]
        for request_index, request in enumerate(requests):
            while request.has_unseen_tokens():
                now = time.perf_counter()
                if first_token_at[request_index] is None:
                    first_token_at[request_index] = now
                else:
                    inter_token_gaps[request_index].append(
                        now - last_token_at[request_index]
                    )
                last_token_at[request_index] = now
                token_counts[request_index] += 1
                fragments[request_index].append(
                    streams[request_index].decode(request.get_unseen_token())
                )
        if ready_request.is_turn_complete():
            completion_notifications.add(ready_index)

    completed = time.perf_counter()
    if any(timestamp is None for timestamp in first_token_at):
        raise RuntimeError("A request completed without producing a token")

    ttft_after_admission = [
        (timestamp - started) * 1000 for timestamp in first_token_at
    ]
    end_to_end_ttft = [
        (timestamp - request_started_at) * 1000 for timestamp in first_token_at
    ]
    gaps = [gap for request_gaps in inter_token_gaps for gap in request_gaps]
    decode_started = min(first_token_at)
    decode_finished = max(last_token_at)
    decode_seconds = max(0.0, decode_finished - decode_started)
    decode_token_count = sum(max(0, count - 1) for count in token_counts)
    return ["".join(parts).strip() for parts in fragments], {
        "tokens": statistics.fmean(token_counts),
        "ttft_after_admission_ms": percentile(ttft_after_admission, 0.50),
        "end_to_end_ttft_ms": percentile(end_to_end_ttft, 0.50),
        "generation_after_admission_ms": (completed - started) * 1000,
        "end_to_end_latency_ms": (completed - request_started_at) * 1000,
        "inter_token_ms": statistics.fmean(gaps) * 1000 if gaps else 0.0,
        "decode_tokens_per_second": (
            decode_token_count / decode_seconds if decode_seconds > 0 else 0.0
        ),
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


def run_once(
    engine,
    model,
    tokenizer,
    prompt_tokens,
    max_length,
    guided,
    guidance_workload,
    fixed_tokens,
    concurrency,
    unique_grammars,
    grammar_batch_id,
):
    setup_started = time.perf_counter()
    request_construction_ms = 0.0
    requests = []
    trackers = [{"index": index} for index in range(concurrency)]
    for index in range(concurrency):
        construction_started = time.perf_counter()
        request_params = og.GeneratorParams(model)
        search_options = {"do_sample": False, "max_length": max_length}
        if fixed_tokens:
            search_options["min_length"] = max_length
        request_params.set_search_options(**search_options)
        if guided:
            if guidance_workload == "json_schema":
                request_params.set_guidance(
                    "lark_grammar", f"start: %json {json.dumps(SCHEMA)}\n"
                )
            elif guidance_workload == "tool_catalog":
                request_params.set_guidance(
                    "lark_grammar",
                    f"start: %json {json.dumps(TOOL_CATALOG_SCHEMA)}\n",
                )
            else:
                regex = OPEN_REGEX
                if unique_grammars:
                    regex = (
                        rf"(?:[A-Za-z ]*|UNIQUE_{grammar_batch_id}_{index})"
                    )
                request_params.set_guidance("regex", regex)
        request = og.Request(request_params)
        request_construction_ms += (
            time.perf_counter() - construction_started
        ) * 1000
        request.add_tokens(np.asarray(prompt_tokens, dtype=np.int32))
        request.set_opaque_data(trackers[index])
        requests.append(request)
    setup_ms = (time.perf_counter() - setup_started) * 1000

    try:
        admission_started = time.perf_counter()
        for request in requests:
            engine.add_request(request)
        admission_ms = (time.perf_counter() - admission_started) * 1000
        outputs, metrics = generate(
            engine, requests, trackers, tokenizer, setup_started
        )
    finally:
        for request in requests:
            engine.remove_request(request)

    metrics["setup_ms"] = setup_ms
    metrics["setup_per_request_ms"] = setup_ms / concurrency
    metrics["request_construction_ms"] = request_construction_ms
    metrics["request_construction_per_request_ms"] = (
        request_construction_ms / concurrency
    )
    metrics["admission_ms"] = admission_ms
    if guidance_workload == "json_schema":
        validation = [validate_output(output) for output in outputs]
        metrics["parseable_json"] = all(result[0] for result in validation)
        metrics["schema_compliant"] = all(result[1] for result in validation)
    elif guidance_workload == "tool_catalog":
        parsed_outputs = []
        for output in outputs:
            try:
                parsed_outputs.append(json.loads(output))
            except json.JSONDecodeError:
                parsed_outputs.append(None)
        metrics["parseable_json"] = all(parsed is not None for parsed in parsed_outputs)
        metrics["schema_compliant"] = all(
            isinstance(parsed, dict)
            and set(parsed) == {"name", "arguments"}
            and parsed.get("name") in TOOL_NAMES
            and isinstance(parsed.get("arguments"), dict)
            and set(parsed["arguments"]).issubset(
                {"path", "query", "count", "recursive"}
            )
            and isinstance(parsed["arguments"].get("path"), str)
            and (
                "query" not in parsed["arguments"]
                or isinstance(parsed["arguments"]["query"], str)
            )
            and (
                "count" not in parsed["arguments"]
                or (
                    isinstance(parsed["arguments"]["count"], int)
                    and not isinstance(parsed["arguments"]["count"], bool)
                )
            )
            and (
                "recursive" not in parsed["arguments"]
                or isinstance(parsed["arguments"]["recursive"], bool)
            )
            for parsed in parsed_outputs
        )
    else:
        metrics["parseable_json"] = None
        metrics["schema_compliant"] = None
    metrics["outputs"] = outputs
    return metrics


def aggregate(runs):
    metric_names = [
        "setup_ms",
        "setup_per_request_ms",
        "request_construction_ms",
        "request_construction_per_request_ms",
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
    parseable_json = [run["parseable_json"] for run in runs if run["parseable_json"] is not None]
    schema_compliant = [run["schema_compliant"] for run in runs if run["schema_compliant"] is not None]
    result["parseable_json_runs"] = sum(parseable_json) if parseable_json else None
    result["schema_compliant_runs"] = sum(schema_compliant) if schema_compliant else None
    result["runs"] = runs
    return result


def run_continuations(
    engine,
    model,
    tokenizer,
    prompt_tokens,
    concurrency,
    continuation_turns,
    max_new_tokens,
):
    suffix_tokens = list(tokenizer.encode(" Next observation."))
    max_length = len(prompt_tokens) + (continuation_turns + 1) * (
        max_new_tokens + len(suffix_tokens)
    )
    requests = []
    trackers = [{"index": index} for index in range(concurrency)]
    for index in range(concurrency):
        request_params = og.GeneratorParams(model)
        request_params.set_search_options(do_sample=False, max_length=max_length)
        request_params.set_guidance(
            "lark_grammar", f"start: %json {json.dumps(SCHEMA)}\n"
        )
        request = og.Request(request_params)
        request.add_tokens(np.asarray(prompt_tokens, dtype=np.int32))
        request.set_opaque_data(trackers[index])
        requests.append(request)

    continuation_setup_ms = []
    compliant_turns = 0
    try:
        for request in requests:
            engine.add_request(request)
        outputs, _ = generate(
            engine, requests, trackers, tokenizer, time.perf_counter()
        )
        compliant_turns += sum(validate_output(output)[1] for output in outputs)

        for _ in range(continuation_turns):
            for request in requests:
                started = time.perf_counter()
                request.continue_with(np.asarray(suffix_tokens, dtype=np.int32))
                continuation_setup_ms.append((time.perf_counter() - started) * 1000)
            outputs, _ = generate(
                engine, requests, trackers, tokenizer, time.perf_counter()
            )
            compliant_turns += sum(validate_output(output)[1] for output in outputs)
    finally:
        for request in requests:
            engine.remove_request(request)

    return {
        "turns_per_request": continuation_turns + 1,
        "continuation_calls": len(continuation_setup_ms),
        "continuation_setup_ms": summarize(continuation_setup_ms),
        "schema_compliant_turns": compliant_turns,
        "expected_schema_compliant_turns": concurrency * (continuation_turns + 1),
    }


def overhead_percent(guided, unguided, metric):
    baseline = unguided[metric]["p50"]
    if baseline == 0:
        return None
    return (guided[metric]["p50"] - baseline) * 100 / baseline


def run(args):
    if (
        args.fixed_tokens
        and args.mode != "off"
        and args.guidance_workload != "open_regex"
    ):
        raise ValueError(
            "--fixed-tokens with guidance requires --guidance-workload open_regex; "
            "closed JSON grammars can finish before min_length."
        )

    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        config.append_provider(args.execution_provider)
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    engine = og.Engine(model)
    cache_before = model.get_guidance_cache_stats()

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
    prompt = (
        args.raw_prompt
        if args.raw_prompt is not None
        else tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True)
    )
    prompt_tokens = list(tokenizer.encode(prompt))
    if args.prompt_tokens is not None:
        if args.prompt_tokens < 1:
            raise ValueError("--prompt-tokens must be positive")
        seed_tokens = prompt_tokens or [tokenizer.encode("x")[0]]
        repeats = (args.prompt_tokens + len(seed_tokens) - 1) // len(seed_tokens)
        prompt_tokens = (seed_tokens * repeats)[: args.prompt_tokens]
    max_length = len(prompt_tokens) + args.max_new_tokens

    modes = [False, True] if args.mode == "both" else [args.mode == "on"]
    grammar_batch_id = 0
    for _ in range(args.warmup):
        for guided in modes:
            run_once(
                engine,
                model,
                tokenizer,
                prompt_tokens,
                max_length,
                guided,
                args.guidance_workload,
                args.fixed_tokens,
                args.concurrency,
                args.unique_grammars,
                grammar_batch_id,
            )
            grammar_batch_id += 1

    runs = {"unguided": [], "guided": []}
    for _ in range(args.iterations):
        for guided in modes:
            name = "guided" if guided else "unguided"
            runs[name].append(
                run_once(
                    engine,
                    model,
                    tokenizer,
                    prompt_tokens,
                    max_length,
                    guided,
                    args.guidance_workload,
                    args.fixed_tokens,
                    args.concurrency,
                    args.unique_grammars,
                    grammar_batch_id,
                )
            )
            grammar_batch_id += 1

    summary = {
        "execution_provider": args.execution_provider,
        "prompt_tokens": len(prompt_tokens),
        "max_new_tokens": args.max_new_tokens,
        "fixed_tokens": args.fixed_tokens,
        "guidance_workload": args.guidance_workload,
        "unique_grammars": args.unique_grammars,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "guidance_cache_before": cache_before,
        "guidance_cache_after": model.get_guidance_cache_stats(),
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
        paired_changes = {}
        for metric in ["setup_ms", "end_to_end_ttft_ms", "inter_token_ms"]:
            changes = [
                (guided[metric] - unguided[metric]) * 100 / unguided[metric]
                for guided, unguided in zip(runs["guided"], runs["unguided"])
                if unguided[metric] != 0
            ]
            paired_changes[metric] = summarize(changes)
        throughput_changes = [
            (guided["decode_tokens_per_second"] - unguided["decode_tokens_per_second"])
            * 100
            / unguided["decode_tokens_per_second"]
            for guided, unguided in zip(runs["guided"], runs["unguided"])
            if unguided["decode_tokens_per_second"] != 0
        ]
        paired_changes["decode_tokens_per_second"] = summarize(throughput_changes)
        summary["guided_paired_change_percent"] = paired_changes

    if args.continuation_turns:
        summary["continuation"] = run_continuations(
            engine,
            model,
            tokenizer,
            prompt_tokens,
            args.concurrency,
            args.continuation_turns,
            args.max_new_tokens,
        )
        summary["guidance_cache_after_continuation"] = (
            model.get_guidance_cache_stats()
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
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4, 8], default=1)
    parser.add_argument("--continuation-turns", type=int, default=0)
    parser.add_argument(
        "--guidance-workload",
        choices=["json_schema", "open_regex", "tool_catalog"],
        default="json_schema",
    )
    parser.add_argument(
        "--fixed-tokens",
        action="store_true",
        help=(
            "Set min_length equal to max_length so both benchmark arms generate the requested token count. "
            "Guided runs require --guidance-workload open_regex because closed grammars may finish earlier."
        ),
    )
    parser.add_argument(
        "--unique-grammars",
        action="store_true",
        help="Use a distinct regex cache key for every guided request in every batch.",
    )
    parser.add_argument(
        "--raw-prompt",
        help="Encode this prompt directly instead of applying the model chat template.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        help="Repeat/truncate the encoded prompt to exactly this many input tokens.",
    )
    parser.add_argument("--output", help="Optional path for the JSON summary")
    run(parser.parse_args())

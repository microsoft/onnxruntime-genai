# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Tool-calling benchmark for any GenAI-supported ONNX chat model.
#
# Tool definitions are handed to the tokenizer as ordinary OpenAI function specs, so
# the run exercises the whole serving path: chat-template rendering of the tool block,
# generation, stopping, and the emitted call itself. Two metrics are worth watching
# beyond overall accuracy, because each catches a configuration fault rather than a
# weakness of the weights:
#
#   enum_valid / no_unknown_params  the prompt advertised a degraded tool schema
#   clean_stop                      the template's end-of-turn token is not a stop id
#
# Example:
#   python benchmark_toolcalling.py -m {model folder} -e cuda -o toolcalling.json

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime_genai as og

sys.path.insert(0, str(Path(__file__).resolve().parent))
from toolcall_parsing import score_case

DEFAULT_CASES = Path(__file__).resolve().parent / "toolcall_cases.json"

METRICS = [
    "correct",
    "correct_function",
    "required_present",
    "enum_valid",
    "no_unknown_params",
    "args_exact",
    "clean_stop",
]


def load_cases(path):
    """Expand each case's tool name list into full OpenAI tool definitions."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    library = data["tools"]
    cases = []
    for case in data["cases"]:
        names = list(library) if case["tools"] == "all" else case["tools"]
        cases.append(
            {
                "name": case["name"],
                "tools": [library[name] for name in names],
                "messages": [{"role": "user", "content": case["query"]}],
                "expected_function": case.get("expected_function"),
                "expected_arguments": case.get("expected_arguments") or {},
            }
        )
    return cases


def is_paged(model_path):
    config = json.loads((Path(model_path) / "genai_config.json").read_text(encoding="utf-8"))
    return "block_table" in config["model"]["decoder"]["inputs"]


def build_model(args):
    config = og.Config(args.model_path)
    config.clear_providers()
    if args.execution_provider != "cpu":
        config.append_provider(args.execution_provider)
    if args.gpu_utilization is not None:
        config.overlay(json.dumps({"engine": {"dynamic_batching": {"gpu_utilization_factor": args.gpu_utilization}}}))
    model = og.Model(config)
    return model, og.Tokenizer(model)


def render_prompts(tokenizer, cases, template_str):
    return [
        tokenizer.apply_chat_template(
            messages=json.dumps(case["messages"]),
            tools=json.dumps(case["tools"]),
            add_generation_prompt=True,
            template_str=template_str,
        )
        for case in cases
    ]


def require_request_event(event):
    if event.flags & og.EngineEventFlags.FAILED:
        raise RuntimeError(f"Generation failed; error_code={event.error_code}")
    if event.request is None:
        raise RuntimeError(f"Engine returned a request-less event; error_code={event.error_code}")
    return event.request


def run_with_engine(model, tokenizer, prompts, max_new_tokens, concurrency):
    """Batch prompts through the Engine in waves of `concurrency`."""
    engine = og.Engine(model)
    event_buffer = engine.create_event_buffer(max(16, concurrency * 4))
    outputs = {}

    for start in range(0, len(prompts), concurrency):
        wave = list(enumerate(prompts))[start : start + concurrency]
        live, by_request = {}, {}
        for index, tokens in wave:
            params = og.GeneratorParams(model)
            params.set_search_options(do_sample=False, max_length=len(tokens) + max_new_tokens)
            request = engine.create_request(params)
            live[index] = []
            by_request[request] = index
            request.begin_turn(np.asarray(tokens, dtype=np.int32))
        try:
            while engine.has_pending_requests():
                for event in engine.run(event_buffer):
                    index = by_request.get(require_request_event(event))
                    if index is not None and event.flags & og.EngineEventFlags.TOKEN:
                        live[index].append(int(event.token))
        finally:
            for request in by_request:
                request.close()
        for index, tokens in live.items():
            outputs[index] = tokenizer.decode(tokens)
    return outputs


def run_with_generator(model, tokenizer, prompts, max_new_tokens):
    outputs = {}
    for index, tokens in enumerate(prompts):
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(tokens) + max_new_tokens)
        generator = og.Generator(model, params)
        generator.append_tokens(tokens)
        while not generator.is_done():
            generator.generate_next_token()
        sequence = generator.get_sequence(0).tolist()
        outputs[index] = tokenizer.decode(sequence[len(tokens) :])
        del generator
    return outputs


def compare_with_reference(model_path, cases, template_str, rendered):
    """Optional guard: HuggingFace renders the same tools without rewriting them.

    A mismatch means the tool block reaching the model is not the one the model was
    trained on, which shows up downstream as invalid argument values. This is a
    diagnostic, so a missing dependency is reported rather than raised: the
    generations are already finished by the time it runs.
    """
    try:
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        mismatches = []
        for case, actual in zip(cases, rendered, strict=False):
            expected = tokenizer.apply_chat_template(
                case["messages"],
                tools=case["tools"],
                add_generation_prompt=True,
                tokenize=False,
                chat_template=template_str,
            )
            if expected != actual:
                mismatches.append(case["name"])
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "hint": "needs transformers and jinja2"}
    return {"checked": len(cases), "mismatched": len(mismatches), "mismatched_cases": mismatches}


def run(args):
    cases = load_cases(args.cases)
    model, tokenizer = build_model(args)

    template_str = None
    if args.chat_template_file:
        template_str = Path(args.chat_template_file).read_text(encoding="utf-8")

    rendered = render_prompts(tokenizer, cases, template_str)
    prompts = [list(tokenizer.encode(text)) for text in rendered]

    runner = args.runner
    if runner == "auto":
        runner = "engine" if is_paged(args.model_path) else "generator"

    started = time.perf_counter()
    if runner == "engine":
        outputs = run_with_engine(model, tokenizer, prompts, args.max_new_tokens, args.concurrency)
    else:
        outputs = run_with_generator(model, tokenizer, prompts, args.max_new_tokens)
    elapsed = time.perf_counter() - started

    rows = []
    for index, case in enumerate(cases):
        text = outputs.get(index, "")
        row = score_case(case, text)
        row["output"] = text
        rows.append(row)

    total = len(rows)
    summary = {
        "model_path": args.model_path,
        "execution_provider": args.execution_provider,
        "runner": runner,
        "cases": total,
        "seconds": round(elapsed, 2),
    }
    for metric in METRICS:
        passed = sum(bool(row[metric]) for row in rows)
        summary[metric] = passed
        summary[f"{metric}_percent"] = round(100.0 * passed / total, 2) if total else None
    summary["call_formats"] = sorted({row["call_format"] for row in rows if row["call_format"]})

    if args.check_prompt_fidelity:
        summary["prompt_fidelity"] = compare_with_reference(args.model_path, cases, template_str, rendered)

    summary["results"] = rows
    text = json.dumps(summary, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))

    failures = [row["name"] for row in rows if not row["correct"]]
    if failures:
        print(f"failed cases: {failures}")
    if args.fail_under is not None and summary["correct_percent"] < args.fail_under:
        print(f"FAIL: correct {summary['correct_percent']}% < required {args.fail_under}%")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tool calling for an ONNX chat model")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-e", "--execution_provider", required=True, choices=["cpu", "cuda", "dml", "webgpu"])
    parser.add_argument("-o", "--output", help="write the full report, including per-case output")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument(
        "--runner",
        choices=["auto", "engine", "generator"],
        default="auto",
        help="auto selects the Engine for paged models, else the Generator",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Engine runner only")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--gpu_utilization",
        type=float,
        help="paged pool share of free VRAM; lower it when a draft head also needs a cache",
    )
    parser.add_argument("--chat_template_file", help="override the model's chat template, e.g. to pin reasoning effort")
    parser.add_argument(
        "--check_prompt_fidelity",
        action="store_true",
        help="compare rendered prompts against HuggingFace (needs transformers, jinja2)",
    )
    parser.add_argument("--fail_under", type=float, help="exit non-zero when the correct percentage falls below this")
    sys.exit(run(parser.parse_args()))

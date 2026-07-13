# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Benchmark n-gram speculative decoding locally or in ep-cert.

Compares standard greedy decoding with model-free n-gram speculative decoding
on identical prompts. The script reuses the model resolution, MT-Bench dataset
loading, chat-template handling, and local-build import logic from
benchmark_speculative.py, but requires only one target model.

Examples:
    # Quick Qwen3-4B CPU smoke test with the bundled prompts.
    python benchmark_ngram.py --model 4b --builtin --max-prompts 2 \
        --ngram-size 3 --k 4 --max-new-tokens 32 --reps 1

    # Small representative run from question.jsonl.
    python benchmark_ngram.py --model 4b --by-category \
        --tasks coding,extraction,math --limit-per-task 2 \
        --ngram-size 2,3,4 --k 2,4,8 --max-new-tokens 64 --reps 2

Run ``benchmark_ngram.py -h`` for all options.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import threading
import time
from datetime import datetime

import benchmark_speculative as common


class ProcessMemoryMonitor:
    """Sample this process's peak resident memory without requiring psutil."""

    _warned = False

    def __init__(self):
        self.peak_rss_gib = 0.0
        self._stop = False
        self._thread = None
        try:
            import psutil  # noqa: PLC0415

            self._process = psutil.Process(os.getpid())
        except ImportError:
            self._process = None
            if not ProcessMemoryMonitor._warned:
                print("[warn] psutil is not installed; peak process memory will be 0. "
                      "Install it with `pip install psutil`.")
                ProcessMemoryMonitor._warned = True

    def _run(self):
        while not self._stop:
            if self._process is not None:
                rss = self._process.memory_info().rss / 1024**3
                self.peak_rss_gib = max(self.peak_rss_gib, rss)
            time.sleep(0.1)

    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def parse_int_list(parser, value, name, minimum, maximum):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        parser.error(f"{name} must be a comma-separated list of integers")
    if not values:
        parser.error(f"{name} must contain at least one value")
    for item in values:
        if not minimum <= item <= maximum:
            parser.error(f"{name} value {item} is outside [{minimum}, {maximum}]")
    return values


def run_once(og, model, token_ids, max_new_tokens, ngram_size=0, max_draft_tokens=0):
    """Run one greedy generation and return timing, output, and native stats."""
    import numpy as np

    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=len(token_ids) + max_new_tokens,
    )
    if ngram_size:
        params.set_speculative_options(
            ngram_size=ngram_size,
            max_draft_tokens=max_draft_tokens,
        )

    generator = og.Generator(model, params)

    prefill_start = time.perf_counter()
    generator.append_tokens(np.array([token_ids], dtype=np.int32))
    start_length = generator.token_count()
    prefill_s = time.perf_counter() - prefill_start

    target_length = start_length + max_new_tokens
    decode_start = time.perf_counter()
    while not generator.is_done() and generator.token_count() < target_length:
        generator.generate_next_token()
    decode_s = time.perf_counter() - decode_start

    new_tokens = generator.token_count() - start_length
    sequence = [int(token) for token in generator.get_sequence(0)]
    stats = dict(generator.get_speculative_stats()) if ngram_size else {}
    del generator
    gc.collect()

    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "new_tokens": new_tokens,
        "tail": sequence[start_length:],
        "stats": stats,
    }


def compare_tokens(expected, actual):
    common_length = min(len(expected), len(actual))
    matching_positions = sum(
        1 for expected_token, actual_token in zip(expected, actual)
        if expected_token == actual_token
    )
    first_difference = next(
        (index for index, pair in enumerate(zip(expected, actual))
         if pair[0] != pair[1]),
        common_length if len(expected) != len(actual) else -1,
    )
    denominator = max(len(expected), len(actual))
    return {
        "exact_match": expected == actual,
        "token_match_rate": matching_positions / denominator if denominator else 1.0,
        "first_difference": first_difference,
    }


CSV_COLUMNS = [
    "model",
    "provider",
    "device",
    "genai_version",
    "python_version",
    "task",
    "subcategory",
    "question_id",
    "prompt_id",
    "rep",
    "decoder",
    "ngram_size",
    "K",
    "prompt_tokens",
    "new_tokens",
    "prefill_s",
    "decode_s",
    "decode_tok_s",
    "e2e_tok_s",
    "speedup_decode",
    "speedup_e2e",
    "exact_match",
    "token_match_rate",
    "first_difference",
    "rounds",
    "draft_proposed",
    "draft_evaluated",
    "draft_accepted",
    "acceptance_rate",
    "mean_accepted_tokens_per_round",
    "mean_emitted_tokens_per_round",
    "corrections",
    "bonuses",
    "draft_forward_passes",
    "target_forward_passes",
    "target_passes_per_token",
    "total_draft_ms",
    "total_target_ms",
    "observed_speedup_estimate",
    "peak_process_rss_gib",
]


def write_results(rows, csv_path, json_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def median_value(rows, task, ngram_size, max_draft_tokens, key):
    values = [
        row[key]
        for row in rows
        if row["decoder"] == "ngram"
        and row["task"] == task
        and row["ngram_size"] == ngram_size
        and row["K"] == max_draft_tokens
        and row[key] != ""
    ]
    return statistics.median(values) if values else None


def geometric_mean(values):
    values = [value for value in values if value is not None and value > 0]
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def print_summary(rows, ngram_sizes, draft_lengths, context):
    tasks = list(dict.fromkeys(
        row["task"] for row in rows if row["decoder"] == "ngram"
    ))
    if not tasks:
        print("\nNo n-gram rows were produced.")
        return

    scores = {}
    for ngram_size in ngram_sizes:
        for max_draft_tokens in draft_lengths:
            scores[(ngram_size, max_draft_tokens)] = geometric_mean([
                median_value(
                    rows, task, ngram_size, max_draft_tokens, "speedup_decode"
                )
                for task in tasks
            ])

    best = max(
        ((config, score) for config, score in scores.items() if score is not None),
        key=lambda item: item[1],
        default=None,
    )

    width = 88
    print("\n" + "=" * width)
    print("N-GRAM SPECULATIVE DECODING SUMMARY".center(width))
    print("=" * width)
    print(
        f"model={context['model']}  EP={context['provider']}  "
        f"device={context['device'] or '-'}  prompts={context['prompts']}  "
        f"reps={context['reps']}  max_new={context['max_new']}  "
        f"peak_process_rss={context['peak_rss']:.2f} GiB"
    )
    if best:
        (best_n, best_k), best_score = best
        verdict = "faster" if best_score >= 1.0 else "did not beat baseline"
        print(f"\n>> BEST: n={best_n}, K={best_k}: {best_score:.2f}x ({verdict})")

    print("\nDecode speedup by configuration (>1.0 is faster; ! marks a regression)")
    header = f"  {'task':18}" + "".join(
        f"{f'n={n}/K={k}':>12}" for n in ngram_sizes for k in draft_lengths
    )
    print(header)
    for task in tasks:
        cells = []
        for ngram_size in ngram_sizes:
            for max_draft_tokens in draft_lengths:
                value = median_value(
                    rows, task, ngram_size, max_draft_tokens, "speedup_decode"
                )
                if value is None:
                    cells.append(f"{'-':>12}")
                else:
                    cells.append(f"{value:>11.2f}{'!' if value < 1.0 else ' '}")
        marker = "*" if task in common.INPUT_GUIDED_TASKS else ""
        print(f"  {task + marker:18}{''.join(cells)}")
    print("  " + "-" * (16 + 12 * len(ngram_sizes) * len(draft_lengths)))
    print(f"  {'GEOMEAN':18}" + "".join(
        f"{scores[(n, k)]:>11.2f}{'!' if scores[(n, k)] < 1.0 else ' '}"
        if scores[(n, k)] is not None else f"{'-':>12}"
        for n in ngram_sizes for k in draft_lengths
    ))

    print("\nConfiguration details (medians across all measured rows)")
    print(
        f"  {'config':12}{'speedup':>10}{'accept':>10}{'accepted/r':>12}"
        f"{'emit/r':>10}{'exact':>9}{'target/tok':>12}"
    )
    for ngram_size in ngram_sizes:
        for max_draft_tokens in draft_lengths:
            selected = [
                row for row in rows
                if row["decoder"] == "ngram"
                and row["ngram_size"] == ngram_size
                and row["K"] == max_draft_tokens
            ]
            if not selected:
                continue
            speedup = statistics.median(row["speedup_decode"] for row in selected)
            acceptance = statistics.median(row["acceptance_rate"] for row in selected)
            accepted_per_round = statistics.median(
                row["mean_accepted_tokens_per_round"] for row in selected
            )
            emitted_per_round = statistics.median(
                row["mean_emitted_tokens_per_round"] for row in selected
            )
            exact = sum(bool(row["exact_match"]) for row in selected) / len(selected)
            target_per_token = statistics.median(
                row["target_passes_per_token"] for row in selected
            )
            print(
                f"  {f'n={ngram_size}/K={max_draft_tokens}':12}"
                f"{speedup:>10.2f}{acceptance:>10.0%}{accepted_per_round:>12.2f}"
                f"{emitted_per_round:>10.2f}{exact:>9.0%}{target_per_token:>12.3f}"
            )

    print("\n* Input-guided task: prompt reuse can increase n-gram proposal coverage.")
    print("Exact must be 100% for every supported deterministic greedy configuration.")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))

    parser.add_argument("--build-root", default=repo_root,
                        help="repository root containing build\\Windows\\Release")
    parser.add_argument(
        "--models-root",
        default=os.path.join(repo_root, "test", "test_models", "qwen3-speculative"),
        help="directory containing Qwen3 model folders",
    )
    parser.add_argument(
        "--model",
        "--target",
        dest="model",
        default="4b",
        help="model key such as 4b, or an explicit directory containing genai_config.json",
    )
    parser.add_argument("--model-prefix", default="qwen3-",
                        help="prefix used when --model is a key")
    parser.add_argument("--ngram-size", default="2,3,4",
                        help="comma-separated n-gram orders to benchmark")
    parser.add_argument("--k", default="2,4,8",
                        help="comma-separated max draft-token counts")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--reps", type=int, default=1,
                        help="measured repetitions per prompt/configuration")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup generations per decoder configuration")

    default_dataset = os.path.join(here, "question.jsonl")
    parser.add_argument(
        "--dataset",
        default=default_dataset if os.path.exists(default_dataset) else None,
        help="Spec-Bench/MT-Bench question.jsonl path",
    )
    parser.add_argument("--builtin", action="store_true",
                        help="use benchmark_speculative.py's small built-in prompt set")
    parser.add_argument(
        "--tasks",
        default=None,
        help="optional comma-separated category filter; omit it with --by-category "
             "to run all categories in the dataset",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=1,
        help="prompts per task/category; defaults to 1 for a manageable CPU run",
    )
    parser.add_argument(
        "--by-category",
        "--mt-bench-by-subcategory",
        dest="by_category",
        action="store_true",
        help="report MT-Bench categories separately instead of one mt_bench task",
    )
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="cap built-in prompts; 0 uses all")
    parser.add_argument("--raw", action="store_true",
                        help="skip the model chat template")
    parser.add_argument("--think", action="store_true",
                        help="enable Qwen3 reasoning; default uses concise no-think mode")
    parser.add_argument("--use-installed", action="store_true",
                        help="use installed onnxruntime-genai instead of the local build")
    parser.add_argument(
        "-e",
        "--execution-provider",
        default="follow_config",
        choices=[
            "follow_config",
            "cpu",
            "OpenVINOExecutionProvider",
            "VitisAIExecutionProvider",
            "QNNExecutionProvider",
            "NvTensorRTRTXExecutionProvider",
            "cuda",
            "dml",
            "webgpu",
        ],
        help="EP to run on. follow_config/cpu uses the model's CPU configuration.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "gpu", "npu"],
        help="hardware device filter for providers such as OpenVINO.",
    )
    parser.add_argument(
        "--device-dir",
        default=None,
        help="Foundry Local onnx/<device-dir> used when resolving a model key.",
    )
    parser.add_argument(
        "--use-winml",
        action="store_true",
        help="register plug-in execution providers through the Windows ML catalog.",
    )
    parser.add_argument(
        "--ep-library-path",
        default=None,
        help="explicit EP plug-in library to register instead of the Windows ML catalog.",
    )
    parser.add_argument("-o", "--output", default=None,
                        help="output prefix without an extension")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    ngram_sizes = parse_int_list(
        parser, args.ngram_size, "--ngram-size", 2, 16
    )
    draft_lengths = parse_int_list(parser, args.k, "--k", 1, 16)
    provider = (
        None
        if args.execution_provider in ("follow_config", "cpu")
        else args.execution_provider
    )
    provider_label = args.execution_provider
    device_dir = args.device_dir or common.default_fl_device_dir(provider)
    use_winml = (
        args.use_winml
        or provider in common._WINML_PLUGIN_EPS and not args.ep_library_path
    )

    if args.dataset and not args.builtin:
        task_filter = (
            {task.strip() for task in args.tasks.split(",") if task.strip()}
            if args.tasks else None
        )
        prompt_items = common.load_dataset_prompts(
            args.dataset,
            task_filter,
            args.limit_per_task,
            mt_bench_by_subcategory=args.by_category,
        )
        if not prompt_items:
            parser.error("no prompts matched --dataset/--tasks")
    else:
        prompt_items = common.builtin_prompts(args.max_prompts)

    model_label = (
        os.path.basename(os.path.normpath(args.model))
        if os.path.isdir(args.model) else f"{args.model_prefix}{args.model}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_folder = (args.device or ("cpu" if provider is None else provider)).lower()
    out_prefix = args.output or os.path.join(
        here, "results", ep_folder, f"ngram_{model_label}_{timestamp}"
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    log_path = common._setup_run_log(out_prefix)
    csv_path = out_prefix + ".csv"
    json_path = out_prefix + ".json"

    og = common._import_og(args.build_root, use_installed=args.use_installed)
    if provider == "webgpu":
        if not common._maybe_register_webgpu(og):
            raise RuntimeError(
                "webgpu EP requested but 'onnxruntime-ep-webgpu' is not installed"
            )
    elif provider:
        common.register_execution_providers(
            og, use_winml, args.ep_library_path, provider
        )

    model_path = common.resolve_model_arg(
        args.models_root,
        args.model,
        args.model_prefix,
        device_dir,
    )
    run_metadata = {
        "model": model_label,
        "provider": provider_label,
        "device": args.device or "",
        "genai_version": getattr(og, "__version__", ""),
        "python_version": sys.version.split()[0],
    }

    print(f"onnxruntime_genai={og.__file__}")
    print(f"model={model_path}")
    print(
        f"prompts={len(prompt_items)}  ngram_sizes={ngram_sizes}  "
        f"K={draft_lengths}  max_new={args.max_new_tokens}  reps={args.reps}"
    )
    print(
        f"execution_provider={provider or 'cpu'}  device={args.device or '-'}  "
        f"device_dir={device_dir or '-'}  use_winml={use_winml}"
    )
    print(f"results={out_prefix}")

    monitor = ProcessMemoryMonitor().start()
    load_start = time.perf_counter()
    model = common.load_model(og, model_path, provider, args.device)
    tokenizer = og.Tokenizer(model)
    print(f"model loaded in {time.perf_counter() - load_start:.1f}s")

    encoded = [
        common.encode_prompt(
            tokenizer,
            item["text"],
            chat=not args.raw,
            think=args.think,
        )
        for item in prompt_items
    ]
    for index, (item, token_ids) in enumerate(zip(prompt_items, encoded)):
        print(
            f"  [{index}] {item['task']}/{item['question_id']}: "
            f"{len(token_ids)} prompt tokens"
        )

    # Fail early with a targeted message if the local extension predates n-gram support.
    try:
        probe = og.GeneratorParams(model)
        probe.set_search_options(do_sample=False, max_length=len(encoded[0]) + 1)
        probe.set_speculative_options(
            ngram_size=ngram_sizes[0],
            max_draft_tokens=draft_lengths[0],
        )
        del probe
    except Exception as error:
        raise RuntimeError(
            "The loaded onnxruntime-genai build does not accept the n-gram speculative "
            "options. Rebuild this branch before running the benchmark."
        ) from error

    if args.warmup:
        print("Warming standard decoding ...", flush=True)
        for _ in range(args.warmup):
            run_once(og, model, encoded[0], min(16, args.max_new_tokens))

    rows = []
    baselines = {}
    benchmark_start = time.perf_counter()

    for prompt_index, (item, token_ids) in enumerate(zip(prompt_items, encoded)):
        decode_rates = []
        e2e_rates = []
        reference_tail = None
        for rep in range(args.reps):
            result = run_once(
                og, model, token_ids, args.max_new_tokens
            )
            decode_rate = (
                result["new_tokens"] / result["decode_s"]
                if result["decode_s"] else 0.0
            )
            e2e_time = result["prefill_s"] + result["decode_s"]
            e2e_rate = result["new_tokens"] / e2e_time if e2e_time else 0.0
            decode_rates.append(decode_rate)
            e2e_rates.append(e2e_rate)
            if reference_tail is None:
                reference_tail = result["tail"]
            elif reference_tail != result["tail"]:
                raise RuntimeError(
                    f"standard greedy output changed across repetitions for "
                    f"prompt {prompt_index}"
                )
            rows.append({
                **run_metadata,
                "task": item["task"],
                "subcategory": item["subcategory"],
                "question_id": item["question_id"],
                "prompt_id": prompt_index,
                "rep": rep,
                "decoder": "standard",
                "ngram_size": "",
                "K": "",
                "prompt_tokens": len(token_ids),
                "new_tokens": result["new_tokens"],
                "prefill_s": round(result["prefill_s"], 6),
                "decode_s": round(result["decode_s"], 6),
                "decode_tok_s": round(decode_rate, 4),
                "e2e_tok_s": round(e2e_rate, 4),
                "speedup_decode": "",
                "speedup_e2e": "",
                "exact_match": "",
                "token_match_rate": "",
                "first_difference": "",
                "rounds": "",
                "draft_proposed": "",
                "draft_evaluated": "",
                "draft_accepted": "",
                "acceptance_rate": "",
                "mean_accepted_tokens_per_round": "",
                "mean_emitted_tokens_per_round": "",
                "corrections": "",
                "bonuses": "",
                "draft_forward_passes": "",
                "target_forward_passes": "",
                "target_passes_per_token": "",
                "total_draft_ms": "",
                "total_target_ms": "",
                "observed_speedup_estimate": "",
                "peak_process_rss_gib": "",
            })
        baselines[prompt_index] = {
            "decode_rate": statistics.median(decode_rates),
            "e2e_rate": statistics.median(e2e_rates),
            "tail": reference_tail,
        }
        print(
            f"[baseline {prompt_index + 1}/{len(prompt_items)}] "
            f"{item['task']}/{item['question_id']}: "
            f"{baselines[prompt_index]['decode_rate']:.2f} tok/s",
            flush=True,
        )
        write_results(rows, csv_path, json_path)

    total_configs = len(ngram_sizes) * len(draft_lengths)
    config_index = 0
    for ngram_size in ngram_sizes:
        for max_draft_tokens in draft_lengths:
            config_index += 1
            print(
                f"\n[config {config_index}/{total_configs}] "
                f"n={ngram_size}, K={max_draft_tokens}",
                flush=True,
            )
            if args.warmup:
                for _ in range(args.warmup):
                    run_once(
                        og,
                        model,
                        encoded[0],
                        min(16, args.max_new_tokens),
                        ngram_size,
                        max_draft_tokens,
                    )

            for prompt_index, (item, token_ids) in enumerate(
                zip(prompt_items, encoded)
            ):
                baseline = baselines[prompt_index]
                measured_speedups = []
                measured_acceptance = []
                measured_exact = []
                for rep in range(args.reps):
                    result = run_once(
                        og,
                        model,
                        token_ids,
                        args.max_new_tokens,
                        ngram_size,
                        max_draft_tokens,
                    )
                    stats = result["stats"]
                    decode_rate = (
                        result["new_tokens"] / result["decode_s"]
                        if result["decode_s"] else 0.0
                    )
                    e2e_time = result["prefill_s"] + result["decode_s"]
                    e2e_rate = result["new_tokens"] / e2e_time if e2e_time else 0.0
                    speedup_decode = (
                        decode_rate / baseline["decode_rate"]
                        if baseline["decode_rate"] else 0.0
                    )
                    speedup_e2e = (
                        e2e_rate / baseline["e2e_rate"]
                        if baseline["e2e_rate"] else 0.0
                    )
                    comparison = compare_tokens(
                        baseline["tail"], result["tail"]
                    )
                    rounds = int(stats.get("rounds", 0))
                    proposed = int(stats.get("draft_tokens_proposed", 0))
                    evaluated = int(stats.get("draft_tokens_evaluated", 0))
                    accepted = int(stats.get("draft_tokens_accepted", 0))
                    target_passes = int(stats.get("target_forward_passes", 0))
                    mean_accepted = accepted / rounds if rounds else 0.0
                    target_passes_per_token = (
                        target_passes / result["new_tokens"]
                        if result["new_tokens"] else 0.0
                    )

                    measured_speedups.append(speedup_decode)
                    measured_acceptance.append(
                        float(stats.get("acceptance_rate", 0.0))
                    )
                    measured_exact.append(comparison["exact_match"])
                    rows.append({
                        **run_metadata,
                        "task": item["task"],
                        "subcategory": item["subcategory"],
                        "question_id": item["question_id"],
                        "prompt_id": prompt_index,
                        "rep": rep,
                        "decoder": "ngram",
                        "ngram_size": ngram_size,
                        "K": max_draft_tokens,
                        "prompt_tokens": len(token_ids),
                        "new_tokens": result["new_tokens"],
                        "prefill_s": round(result["prefill_s"], 6),
                        "decode_s": round(result["decode_s"], 6),
                        "decode_tok_s": round(decode_rate, 4),
                        "e2e_tok_s": round(e2e_rate, 4),
                        "speedup_decode": round(speedup_decode, 4),
                        "speedup_e2e": round(speedup_e2e, 4),
                        "exact_match": comparison["exact_match"],
                        "token_match_rate": round(
                            comparison["token_match_rate"], 6
                        ),
                        "first_difference": comparison["first_difference"],
                        "rounds": rounds,
                        "draft_proposed": proposed,
                        "draft_evaluated": evaluated,
                        "draft_accepted": accepted,
                        "acceptance_rate": round(
                            float(stats.get("acceptance_rate", 0.0)), 6
                        ),
                        "mean_accepted_tokens_per_round": round(
                            mean_accepted, 6
                        ),
                        "mean_emitted_tokens_per_round": round(
                            float(stats.get("mean_emitted_tokens_per_round", 0.0)),
                            6,
                        ),
                        "corrections": int(stats.get("correction_tokens", 0)),
                        "bonuses": int(stats.get("bonus_tokens", 0)),
                        "draft_forward_passes": int(
                            stats.get("draft_forward_passes", 0)
                        ),
                        "target_forward_passes": target_passes,
                        "target_passes_per_token": round(
                            target_passes_per_token, 6
                        ),
                        "total_draft_ms": round(
                            float(stats.get("total_draft_ms", 0.0)), 6
                        ),
                        "total_target_ms": round(
                            float(stats.get("total_target_ms", 0.0)), 6
                        ),
                        "observed_speedup_estimate": round(
                            float(stats.get("observed_speedup", 0.0)), 6
                        ),
                        "peak_process_rss_gib": "",
                    })

                print(
                    f"  {item['task']}/{item['question_id']}: "
                    f"x{statistics.median(measured_speedups):.2f}, "
                    f"accept={statistics.median(measured_acceptance):.0%}, "
                    f"exact={sum(measured_exact)}/{len(measured_exact)}",
                    flush=True,
                )
                write_results(rows, csv_path, json_path)

    monitor.stop()
    for row in rows:
        row["peak_process_rss_gib"] = round(monitor.peak_rss_gib, 3)
    write_results(rows, csv_path, json_path)

    elapsed = time.perf_counter() - benchmark_start
    print(
        f"\nCompleted measured benchmark in {elapsed:.1f}s. "
        f"Wrote {csv_path}, {json_path}, and {log_path}"
    )
    print_summary(
        rows,
        ngram_sizes,
        draft_lengths,
        {
            "model": model_label,
            "provider": provider_label,
            "device": args.device or "",
            "prompts": len(prompt_items),
            "reps": args.reps,
            "max_new": args.max_new_tokens,
            "peak_rss": monitor.peak_rss_gib,
        },
    )


if __name__ == "__main__":
    main()

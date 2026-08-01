# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Benchmark n-gram speculative decoding locally or in ep-cert.

Compares standard decoding with model-free n-gram speculative decoding on
identical prompts. Greedy decoding is the default; sampling can be selected
with fixed seeds and configurable search controls. MT-Bench measures broad
speed behavior, while optional instruction-following, math, long-context, and
code suites add task-level quality measurements.
The script reuses model resolution, chat-template handling, and local-build
import logic from benchmark_speculative.py, but requires only one target model.

Examples:
    # Quick Qwen3-4B CPU smoke test with the bundled prompts.
    python benchmark_ngram.py --model 4b --builtin --max-prompts 2 \
        --ngram-size 3 --k 4 --max-new-tokens 32 --reps 1

    # Small representative run from question.jsonl.
    python benchmark_ngram.py --model 4b --by-category \
        --tasks coding,extraction,math --limit-per-task 2 \
        --ngram-size 2,3,4 --k 2,4,8 --max-new-tokens 64 --reps 2

    # Add GSM8K accuracy and HumanEval pass@1 to the MT-Bench run.
    python benchmark_ngram.py --model 4b \
        --suites mtbench,gsm8k,humaneval --gsm8k-problems 50 \
        --humaneval-problems 20 --max-new-tokens 512 --allow-code-execution

    # Exercise sampling with two fixed seeds and verify reproducibility.
    python benchmark_ngram.py --model 4b --builtin --modes sampling \
        --sampling-seeds 0,1 --temperature 0.7 --top-k 20 --top-p 0.95 \
        --repetition-penalty 1.0 --reps 2

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

import benchmark_suites as suite_utils
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
    values = []
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


def parse_suites(parser, value):
    return suite_utils.parse_suites(parser, value)


def parse_modes(parser, value):
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not modes:
        parser.error("--modes must contain at least one value")
    unknown = sorted(set(modes) - {"greedy", "sampling"})
    if unknown:
        parser.error(
            f"--modes contains unsupported values: {', '.join(unknown)}; "
            "choose from greedy,sampling"
        )
    return list(dict.fromkeys(modes))


def run_once(
    og,
    model,
    token_ids,
    max_new_tokens,
    mode="greedy",
    seed=0,
    sampling_options=None,
    ngram_size=0,
    max_draft_tokens=0,
    adaptive_k=False,
    adaptive_k_min=2,
    chained_lookup=False,
    cooldown=False,
):
    """Run one generation and return timing, output, and native stats."""
    import numpy as np

    params = og.GeneratorParams(model)
    search_options = {
        "do_sample": mode == "sampling",
        "max_length": len(token_ids) + max_new_tokens,
    }
    if mode == "sampling":
        sampling_options = sampling_options or {}
        search_options.update(
            temperature=sampling_options["temperature"],
            top_k=sampling_options["top_k"],
            top_p=sampling_options["top_p"],
            repetition_penalty=sampling_options["repetition_penalty"],
            min_length=len(token_ids) + sampling_options["min_new_tokens"],
            random_seed=seed,
        )
    params.set_search_options(**search_options)
    if ngram_size:
        speculative_options = dict(
            ngram_size=ngram_size,
            max_draft_tokens=max_draft_tokens,
        )
        if adaptive_k:
            speculative_options.update(
                adaptive_k_bool=True,
                adaptive_k_min=adaptive_k_min,
            )
        if chained_lookup:
            speculative_options["ngram_chained_lookup_bool"] = True
        if cooldown:
            speculative_options["cooldown_bool"] = True
        params.set_speculative_options(**speculative_options)
        common.verify_speculative_options(params, speculative_options)

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
    if stats:
        common.validate_speculative_stats(stats)
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
    "mode",
    "seed",
    "temperature",
    "top_k",
    "top_p",
    "repetition_penalty",
    "min_new_tokens",
    "reproducibility_checked",
    "reproducible",
    "task",
    "subcategory",
    "question_id",
    "quality_metric",
    "quality_score_type",
    "quality_score",
    "baseline_quality_score",
    "quality_score_delta",
    "quality_prediction",
    "quality_reference_answer",
    "quality_detail",
    "quality_transition",
    "prompt_id",
    "rep",
    "decoder",
    "ngram_size",
    "K",
    "chained_lookup",
    "cooldown",
    "adaptive_k",
    "adaptive_k_min",
    "effective_k",
    "adaptive_k_increases",
    "adaptive_k_decreases",
    "adaptive_k_observations",
    "adaptive_k_probes",
    "adaptive_k_throughput",
    "prompt_tokens",
    "output_token_budget",
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
    "divergence_type",
    "rounds",
    "completed_rounds",
    "interrupted_rounds",
    "active_rounds",
    "draft_proposed",
    "draft_evaluated",
    "draft_accepted",
    "acceptance_rate",
    "avg_draft_tokens_per_round",
    "mean_accepted_tokens_per_round",
    "mean_emitted_tokens_per_round",
    "expected_tokens_per_round",
    "corrections",
    "bonuses",
    "tokens_queued",
    "tokens_emitted",
    "tokens_discarded",
    "tokens_buffered",
    "draft_forward_passes",
    "target_forward_passes",
    "target_verify_forward_passes",
    "target_reanchor_forward_passes",
    "target_reconciliation_forward_passes",
    "target_passes_per_token",
    "cooldown_entries",
    "cooldown_steps",
    "cooldown_remaining",
    "standard_fallback_steps",
    "full_accept_rounds",
    "partial_accept_rounds",
    "zero_accept_rounds",
    "ngram_lookup_hits",
    "ngram_lookup_misses",
    "ngram_lookup_tokens_proposed",
    "ngram_chained_tokens_proposed",
    "ngram_grammar_candidate_rejections",
    "ngram_history_syncs",
    "ngram_history_tokens_synced",
    "total_draft_ms",
    "total_target_ms",
    "total_reconciliation_ms",
    "total_target_verify_ms",
    "total_target_reanchor_ms",
    "total_ngram_history_sync_ms",
    "total_ngram_lookup_ms",
    "avg_draft_ms_per_token",
    "avg_target_ms_per_round",
    "target_baseline_ms_per_token",
    "target_overhead_ratio",
    "formula_supported",
    "estimated_speedup",
    "observed_speedup",
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


def percentile(values, percent):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def classify_divergence(comparison, expected_length, actual_length):
    if comparison["exact_match"]:
        return "exact"
    if comparison["first_difference"] == min(expected_length, actual_length):
        return "length_only"
    return "token_divergence"


def classify_quality_transition(baseline_quality, ngram_quality):
    return suite_utils.classify_quality_transition(
        baseline_quality, ngram_quality, "ngram"
    )


def format_quality(quality):
    return suite_utils.format_quality(quality)


def format_quality_with_reference(quality):
    return suite_utils.format_quality_with_reference(quality)


def format_seed(mode, seed):
    return str(seed) if mode == "sampling" else "-"


def format_config(ngram_size, k):
    return f"n={ngram_size}/adaptive" if k == "adaptive" else f"n={ngram_size}/K={k}"


def print_detailed_run(item, rep, baseline, row):
    seed_text = f" seed={row['seed']}" if row["mode"] == "sampling" else ""
    print(
        f"  {item['task']}/{item['question_id']} mode={row['mode']}{seed_text} "
        f"{format_config(row['ngram_size'], row['K'])} "
        f"chained_lookup={'on' if row['chained_lookup'] else 'off'} "
        f"cooldown={'on' if row['cooldown'] else 'off'} rep={rep + 1}"
    )
    print(
        f"    speed: decode={row['speedup_decode']:.2f}x "
        f"e2e={row['speedup_e2e']:.2f}x "
        f"baseline={baseline['decode_rate']:.2f} tok/s "
        f"ngram={row['decode_tok_s']:.2f} tok/s"
    )
    print(
        f"    timing: prefill={row['prefill_s']:.4f}s "
        f"decode={row['decode_s']:.4f}s "
        f"e2e={row['prefill_s'] + row['decode_s']:.4f}s"
    )
    print(
        f"    output: baseline={len(baseline['tail'])} tokens "
        f"ngram={row['new_tokens']} exact={row['exact_match']} "
        f"first_diff={row['first_difference']} "
        f"match={row['token_match_rate']:.1%} "
        f"type={row['divergence_type']}"
    )
    print(
        f"    spec: accept={row['acceptance_rate']:.1%} "
        f"proposed={row['draft_proposed']} evaluated={row['draft_evaluated']} "
        f"accepted={row['draft_accepted']} rounds={row['rounds']} "
        f"accepted/round={row['mean_accepted_tokens_per_round']:.2f} "
        f"emitted/round={row['mean_emitted_tokens_per_round']:.2f} "
        f"target_passes/token={row['target_passes_per_token']:.3f}"
    )
    print(
        f"    target: verify={row['target_verify_forward_passes']} "
        f"reanchor={row['target_reanchor_forward_passes']} "
        f"reconcile={row['target_reconciliation_forward_passes']} "
        f"timing={row['total_target_verify_ms']:.1f}/"
        f"{row['total_target_reanchor_ms']:.1f}/"
        f"{row['total_reconciliation_ms']:.1f}ms"
    )
    print(
        f"    n-gram: hits={row['ngram_lookup_hits']} "
        f"misses={row['ngram_lookup_misses']} "
        f"lookup_tokens={row['ngram_lookup_tokens_proposed']} "
        f"chained_tokens={row['ngram_chained_tokens_proposed']} "
        f"grammar_rejections={row['ngram_grammar_candidate_rejections']} "
        f"lookup={row['total_ngram_lookup_ms']:.1f}ms "
        f"history_sync={row['ngram_history_syncs']}/"
        f"{row['ngram_history_tokens_synced']} tokens/"
        f"{row['total_ngram_history_sync_ms']:.1f}ms"
    )
    if row["cooldown"]:
        print(
            f"    cooldown: entries={row['cooldown_entries']} "
            f"steps={row['cooldown_steps']} "
            f"fallback={row['standard_fallback_steps']} "
            f"remaining={row['cooldown_remaining']} "
            f"accept_rounds={row['full_accept_rounds']}/"
            f"{row['partial_accept_rounds']}/{row['zero_accept_rounds']} "
            "(full/partial/zero)"
        )
    if row["adaptive_k"]:
        print(
            f"    adaptive K: start={row['adaptive_k_min']} "
            f"final={row['effective_k']} "
            f"moves=+{row['adaptive_k_increases']}/-{row['adaptive_k_decreases']} "
            f"probes={row['adaptive_k_probes']} "
            f"observations={row['adaptive_k_observations']} "
            f"throughput={row['adaptive_k_throughput']:.4f} tok/ms"
        )
    if row["quality_metric"]:
        print(
            f"    quality: baseline={format_quality_with_reference(baseline['quality'])} "
            f"ngram={format_quality_with_reference(row)} "
            f"transition={row['quality_transition']}"
        )


def print_mismatch_completions(tokenizer, baseline_tokens, ngram_tokens):
    limit = 2000
    baseline_text = tokenizer.decode(baseline_tokens)
    ngram_text = tokenizer.decode(ngram_tokens)
    if len(baseline_text) > limit:
        baseline_text = baseline_text[:limit] + "... [truncated]"
    if len(ngram_text) > limit:
        ngram_text = ngram_text[:limit] + "... [truncated]"
    print("    baseline completion:")
    print(baseline_text)
    print("    n-gram completion:")
    print(ngram_text)


def print_progress_summary(config_rows, completed, total, ngram_size, max_draft_tokens):
    speedups = [row["speedup_decode"] for row in config_rows]
    acceptance = [row["acceptance_rate"] for row in config_rows]
    exact = sum(bool(row["exact_match"]) for row in config_rows)
    regressions = sum(row["speedup_decode"] < 1.0 for row in config_rows)
    print(
        f"\n  [progress {completed}/{total} mode={config_rows[0]['mode']} "
        f"seed={format_seed(config_rows[0]['mode'], config_rows[0]['seed'])} "
        f"{format_config(ngram_size, max_draft_tokens)}] "
        f"decode speedup median={statistics.median(speedups):.2f}x "
        f"p10={percentile(speedups, 10):.2f}x "
        f"p90={percentile(speedups, 90):.2f}x; "
        f"regressions={regressions}/{len(config_rows)}; "
        f"acceptance median={statistics.median(acceptance):.1%}; "
        f"exact={exact}/{len(config_rows)}"
    )
    if config_rows[0]["adaptive_k"]:
        print(
            f"    adaptive K: start={config_rows[0]['adaptive_k_min']} "
            f"final_p50={statistics.median(row['effective_k'] for row in config_rows):.1f} "
            f"final_range={min(row['effective_k'] for row in config_rows)}-"
            f"{max(row['effective_k'] for row in config_rows)} "
            f"moves=+{sum(row['adaptive_k_increases'] for row in config_rows)}"
            f"/-{sum(row['adaptive_k_decreases'] for row in config_rows)} "
            f"probes={sum(row['adaptive_k_probes'] for row in config_rows)} "
            f"observations={sum(row['adaptive_k_observations'] for row in config_rows)}"
        )
    for line in suite_utils.format_quality_summary_lines(
        config_rows, "standard", "ngram"
    ):
        print(f"    {line}")
    print(flush=True)


def print_summary_group(rows, ngram_sizes, draft_lengths, context):
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

    width = 104
    seed_label = context.get(
        "seed_label", format_seed(context["mode"], context["seed"])
    )
    print("\n" + "=" * width)
    print("N-GRAM SPECULATIVE DECODING SUMMARY".center(width))
    print("=" * width)
    print(
        f"model={context['model']}  EP={context['provider']}  "
        f"device={context['device'] or '-'}  prompts={context['prompts']}  "
        f"mode={context['mode']}  seed={seed_label}  "
        f"chained_lookup={'on' if context.get('chained_lookup', False) else 'off'}  "
        f"cooldown={'on' if context.get('cooldown', False) else 'off'}  "
        f"reps={context['reps']}  max_new={context['max_new']}  "
        f"peak_process_rss={context['peak_rss']:.2f} GiB"
    )
    if best:
        (best_n, best_k), best_score = best
        verdict = "faster" if best_score >= 1.0 else "did not beat baseline"
        print(f"\n>> BEST: {format_config(best_n, best_k)}: {best_score:.2f}x ({verdict})")

    print("\nDecode speedup by configuration (>1.0 is faster; ! marks a regression)")
    header = f"  {'task':18}" + "".join(
        f"{format_config(n, k):>16}" for n in ngram_sizes for k in draft_lengths
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
                    cells.append(f"{'-':>16}")
                else:
                    cells.append(f"{value:>15.2f}{'!' if value < 1.0 else ' '}")
        marker = "*" if task in common.INPUT_GUIDED_TASKS else ""
        print(f"  {task + marker:18}{''.join(cells)}")
    print("  " + "-" * (16 + 16 * len(ngram_sizes) * len(draft_lengths)))
    print(f"  {'GEOMEAN':18}" + "".join(
        f"{scores[(n, k)]:>15.2f}{'!' if scores[(n, k)] < 1.0 else ' '}"
        if scores[(n, k)] is not None else f"{'-':>16}"
        for n in ngram_sizes for k in draft_lengths
    ))

    print("\nPerformance distribution across all measured rows")
    print(
        f"  {'config':16}{'min':>8}{'mean':>8}{'p10':>8}{'p50':>8}{'p90':>8}{'max':>8}"
        f"{'geomean':>10}{'>1x':>8}{'e2e p50':>10}"
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
            speedups = [row["speedup_decode"] for row in selected]
            e2e_speedups = [row["speedup_e2e"] for row in selected]
            faster = sum(value > 1.0 for value in speedups) / len(speedups)
            print(
                f"  {format_config(ngram_size, max_draft_tokens):16}"
                f"{min(speedups):>8.2f}"
                f"{statistics.mean(speedups):>8.2f}"
                f"{percentile(speedups, 10):>8.2f}"
                f"{statistics.median(speedups):>8.2f}"
                f"{percentile(speedups, 90):>8.2f}"
                f"{max(speedups):>8.2f}"
                f"{geometric_mean(speedups):>10.2f}"
                f"{faster:>8.0%}"
                f"{statistics.median(e2e_speedups):>10.2f}"
            )

    print("\nSpeculation efficiency and correctness")
    print(
        f"  {'config':16}{'accept':>9}{'weighted':>10}{'accepted/r':>12}"
        f"{'emit/r':>9}{'target/tok':>11}{'exact':>8}{'match':>9}"
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
            acceptance = statistics.median(row["acceptance_rate"] for row in selected)
            total_accepted = sum(row["draft_accepted"] for row in selected)
            total_evaluated = sum(row["draft_evaluated"] for row in selected)
            weighted_acceptance = (
                total_accepted / total_evaluated if total_evaluated else 0.0
            )
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
            token_match = statistics.median(
                row["token_match_rate"] for row in selected
            )
            print(
                f"  {format_config(ngram_size, max_draft_tokens):16}"
                f"{acceptance:>9.1%}{weighted_acceptance:>10.1%}"
                f"{accepted_per_round:>12.2f}{emitted_per_round:>9.2f}"
                f"{target_per_token:>11.3f}{exact:>8.1%}{token_match:>9.1%}"
            )
            divergence_counts = {}
            for row in selected:
                divergence_counts[row["divergence_type"]] = (
                    divergence_counts.get(row["divergence_type"], 0) + 1
                )
            counts = ", ".join(
                f"{name}={value}" for name, value in sorted(divergence_counts.items())
            )
            mismatch_positions = [
                row["first_difference"] for row in selected
                if not row["exact_match"] and row["first_difference"] >= 0
            ]
            first_diff = (
                f"{statistics.median(mismatch_positions):.0f}"
                if mismatch_positions else "-"
            )
            print(
                f"    totals: proposed={sum(row['draft_proposed'] for row in selected)} "
                f"evaluated={total_evaluated} accepted={total_accepted} "
                f"rounds={sum(row['rounds'] for row in selected)} "
                f"corrections={sum(row['corrections'] for row in selected)} "
                f"bonuses={sum(row['bonuses'] for row in selected)} "
                f"target_passes={sum(row['target_forward_passes'] for row in selected)}"
            )
            print(
                f"    timing: draft={sum(row['total_draft_ms'] for row in selected):.1f}ms "
                f"target={sum(row['total_target_ms'] for row in selected):.1f}ms; "
                f"mismatch_first_diff_p50={first_diff}; {counts}"
            )
            print(
                f"    target breakdown: verify="
                f"{sum(row['target_verify_forward_passes'] for row in selected)} passes/"
                f"{sum(row['total_target_verify_ms'] for row in selected):.1f}ms "
                f"reanchor={sum(row['target_reanchor_forward_passes'] for row in selected)} passes/"
                f"{sum(row['total_target_reanchor_ms'] for row in selected):.1f}ms "
                f"reconcile="
                f"{sum(row['target_reconciliation_forward_passes'] for row in selected)} passes/"
                f"{sum(row['total_reconciliation_ms'] for row in selected):.1f}ms"
            )
            lookup_hits = sum(row["ngram_lookup_hits"] for row in selected)
            lookup_misses = sum(row["ngram_lookup_misses"] for row in selected)
            lookup_total = lookup_hits + lookup_misses
            print(
                f"    n-gram lookup: hits={lookup_hits} misses={lookup_misses} "
                f"hit_rate={lookup_hits / lookup_total if lookup_total else 0.0:.1%} "
                f"proposed={sum(row['ngram_lookup_tokens_proposed'] for row in selected)} "
                f"chained={sum(row['ngram_chained_tokens_proposed'] for row in selected)} "
                f"grammar_rejections="
                f"{sum(row['ngram_grammar_candidate_rejections'] for row in selected)} "
                f"lookup={sum(row['total_ngram_lookup_ms'] for row in selected):.1f}ms "
                f"history_sync={sum(row['ngram_history_syncs'] for row in selected)}/"
                f"{sum(row['ngram_history_tokens_synced'] for row in selected)} tokens/"
                f"{sum(row['total_ngram_history_sync_ms'] for row in selected):.1f}ms"
            )
            print(
                f"    round outcomes: full="
                f"{sum(row['full_accept_rounds'] for row in selected)} "
                f"partial={sum(row['partial_accept_rounds'] for row in selected)} "
                f"zero={sum(row['zero_accept_rounds'] for row in selected)}"
            )
            print(
                f"    lifecycle: completed={sum(row['completed_rounds'] for row in selected)} "
                f"interrupted={sum(row['interrupted_rounds'] for row in selected)} "
                f"active={sum(row['active_rounds'] for row in selected)} "
                f"queued/emitted/discarded/buffered="
                f"{sum(row['tokens_queued'] for row in selected)}/"
                f"{sum(row['tokens_emitted'] for row in selected)}/"
                f"{sum(row['tokens_discarded'] for row in selected)}/"
                f"{sum(row['tokens_buffered'] for row in selected)}"
            )
            formula_rows = [row for row in selected if row["formula_supported"]]
            if formula_rows:
                print(
                    f"    formula: supported={len(formula_rows)}/{len(selected)} "
                    f"estimated_p50="
                    f"{statistics.median(row['estimated_speedup'] for row in formula_rows):.2f}x "
                    f"observed_p50="
                    f"{statistics.median(row['observed_speedup'] for row in formula_rows):.2f}x "
                    f"target_overhead_p50="
                    f"{statistics.median(row['target_overhead_ratio'] for row in formula_rows):.3f}"
                )
            if selected[0]["cooldown"]:
                print(
                    f"    cooldown: entries={sum(row['cooldown_entries'] for row in selected)} "
                    f"steps={sum(row['cooldown_steps'] for row in selected)} "
                    f"fallback={sum(row['standard_fallback_steps'] for row in selected)} "
                    f"remaining={sum(row['cooldown_remaining'] for row in selected)}"
                )
            if selected[0]["adaptive_k"]:
                print(
                    f"    adaptive K: start={selected[0]['adaptive_k_min']} "
                    f"final_p50={statistics.median(row['effective_k'] for row in selected):.1f} "
                    f"final_range={min(row['effective_k'] for row in selected)}-"
                    f"{max(row['effective_k'] for row in selected)} "
                    f"moves=+{sum(row['adaptive_k_increases'] for row in selected)}"
                    f"/-{sum(row['adaptive_k_decreases'] for row in selected)} "
                    f"probes={sum(row['adaptive_k_probes'] for row in selected)} "
                    f"observations={sum(row['adaptive_k_observations'] for row in selected)} "
                    f"throughput_p50="
                    f"{statistics.median(row['adaptive_k_throughput'] for row in selected):.4f} tok/ms"
                )

    print("\n* Input-guided task: prompt reuse can increase n-gram proposal coverage.")
    if context["mode"] == "greedy":
        print(
            "Exact measures bit-identical greedy output. Batched verification can diverge "
            "from sequential greedy due to floating-point accumulation order."
        )
    else:
        print(
            "Exact/match compare paired fixed-seed sampling paths for diagnostics only; "
            "different valid samples are not a correctness failure."
        )

    quality_tasks = list(dict.fromkeys(
        row["task"] for row in rows
        if row["quality_metric"] and row["rep"] == 0
    ))
    if quality_tasks:
        print("\nTask quality (first deterministic repetition)")
        print(f"  {'task':18}{'metric':>12}{'standard':>12}" + "".join(
            f"{format_config(n, k):>16}" for n in ngram_sizes for k in draft_lengths
        ))
        for task in quality_tasks:
            task_rows = [
                row for row in rows
                if row["task"] == task and row["rep"] == 0
            ]
            metric = next(row["quality_metric"] for row in task_rows)
            baseline_scores = [
                float(row["quality_score"]) for row in task_rows
                if row["decoder"] == "standard"
            ]
            baseline = (
                sum(baseline_scores) / len(baseline_scores)
                if baseline_scores else None
            )
            cells = []
            for ngram_size in ngram_sizes:
                for max_draft_tokens in draft_lengths:
                    config_scores = [
                        float(row["quality_score"]) for row in task_rows
                        if row["decoder"] == "ngram"
                        and row["ngram_size"] == ngram_size
                        and row["K"] == max_draft_tokens
                    ]
                    score = (
                        sum(config_scores) / len(config_scores)
                        if config_scores else None
                    )
                    cells.append(
                        f"{score:>15.1%} " if score is not None else f"{'-':>16}"
                    )
            baseline_text = (
                f"{baseline:>11.1%} " if baseline is not None else f"{'-':>12}"
            )
            print(f"  {task:18}{metric:>12}{baseline_text}{''.join(cells)}")
            for ngram_size in ngram_sizes:
                for max_draft_tokens in draft_lengths:
                    transitions = {}
                    for row in task_rows:
                        if (
                            row["decoder"] == "ngram"
                            and row["ngram_size"] == ngram_size
                            and row["K"] == max_draft_tokens
                        ):
                            transitions[row["quality_transition"]] = (
                                transitions.get(row["quality_transition"], 0) + 1
                            )
                    if transitions:
                        details = ", ".join(
                            f"{name}={value}"
                            for name, value in sorted(transitions.items())
                        )
                        print(
                            f"    {task} {format_config(ngram_size, max_draft_tokens)}: "
                            f"{details}"
                        )
    print("=" * width)


def print_summary(rows, ngram_sizes, draft_lengths, context):
    modes = list(dict.fromkeys(
        row["mode"] for row in rows if row["decoder"] == "ngram"
    ))
    for mode in modes:
        selected = [
            row for row in rows
            if row["mode"] == mode
        ]
        print_summary_group(
            selected,
            ngram_sizes,
            draft_lengths,
            {
                **context,
                "mode": mode,
                "seed": None,
                "seed_label": (
                    "-"
                    if mode == "greedy"
                    else "all configured seeds"
                ),
            },
        )


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
                        help="comma-separated fixed draft-token counts; ignored with --adaptive-k")
    parser.add_argument(
        "--adaptive-k",
        action="store_true",
        help="adapt K from --adaptive-k-min up to the native limit instead of sweeping --k",
    )
    parser.add_argument(
        "--adaptive-k-min",
        type=int,
        default=2,
        help="starting K and floor when --adaptive-k is enabled",
    )
    parser.add_argument(
        "--chained-lookup",
        action="store_true",
        help="refill n-gram proposals by repeatedly looking up synthetic context",
    )
    parser.add_argument(
        "--cooldown",
        action="store_true",
        help="temporarily use standard decoding after repeated zero-accept rounds",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="new tokens for Spec-Bench/builtin prompts; quality suites use pinned task budgets",
    )
    parser.add_argument("--reps", type=int, default=1,
                        help="measured repetitions per prompt/configuration")
    parser.add_argument("--warmup", type=int, default=1,
                        help="warmup generations per decoder configuration")
    parser.add_argument(
        "--modes",
        default="greedy",
        help="comma-separated decoding modes: greedy,sampling",
    )
    parser.add_argument(
        "--sampling-seeds",
        default="0",
        help="comma-separated fixed seeds; used only when sampling is selected",
    )
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=20,
                        help="sampling top-k; 0 disables top-k filtering")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="sampling nucleus probability")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="sampling repetition penalty")
    parser.add_argument("--min-new-tokens", type=int, default=0,
                        help="minimum new tokens before EOS is allowed in sampling mode")
    parser.add_argument(
        "--suites",
        default="mtbench",
        help="comma-separated suites: mtbench,gsm8k,ifeval,math500,longbench,"
             "humaneval,humanevalplus,mbppplus,livecodebench",
    )

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
    parser.add_argument(
        "--gsm8k-path",
        default=os.path.join(here, ".cache", "gsm8k_test.jsonl"),
        help="GSM8K test JSONL path; downloaded from the official repository if absent",
    )
    parser.add_argument(
        "--gsm8k-problems",
        type=int,
        default=200,
        help="number of GSM8K problems; 0 uses the full 1,319-problem test set",
    )
    parser.add_argument(
        "--ifeval-path",
        default=os.path.join(here, ".cache", "ifeval_input_data.jsonl"),
        help="IFEval JSONL path; downloaded from the pinned official dataset if absent",
    )
    parser.add_argument("--ifeval-problems", type=int, default=541,
                        help="number of IFEval prompts; 0 uses all 541")
    parser.add_argument(
        "--math500-path",
        default=os.path.join(here, ".cache", "math500_test.jsonl"),
        help="MATH-500 JSONL path; downloaded from the pinned dataset if absent",
    )
    parser.add_argument("--math500-problems", type=int, default=500,
                        help="number of MATH-500 problems; 0 uses all 500")
    parser.add_argument(
        "--longbench-tasks",
        default="qasper,hotpotqa,gov_report,passage_retrieval_en",
        help="LongBench v1 tasks: qasper,hotpotqa,gov_report,passage_retrieval_en",
    )
    parser.add_argument("--longbench-problems-per-task", type=int, default=50,
                        help="problems per selected LongBench task; 0 uses all")
    parser.add_argument("--longbench-max-input-tokens", type=int, default=16384,
                        help="middle-truncate LongBench prompts above this input-token count")
    parser.add_argument(
        "--humaneval-problems",
        type=int,
        default=164,
        help="number of HumanEval problems; 0 uses all 164",
    )
    parser.add_argument("--humanevalplus-problems", type=int, default=164,
                        help="number of HumanEval+ problems; 0 uses all 164")
    parser.add_argument("--mbppplus-problems", type=int, default=378,
                        help="number of MBPP+ problems; 0 uses all 378")
    # BigCodeBench is temporarily disabled pending a compatible evaluator environment.
    # parser.add_argument("--bigcodebench-problems", type=int, default=148,
    #                     help="number of BigCodeBench-Hard problems; 0 uses all 148")
    # parser.add_argument("--bigcodebench-subset", choices=["hard"], default="hard",
    #                     help="pinned BigCodeBench subset")
    parser.add_argument("--livecodebench-problems", type=int, default=100,
                        help="number of newest LiveCodeBench problems; 0 uses the full release")
    parser.add_argument("--livecodebench-release", default="release_v6",
                        help="pinned LiveCodeBench release to select")
    parser.add_argument(
        "--code-execution-timeout",
        "--humaneval-timeout",
        dest="code_execution_timeout",
        type=float,
        default=3.0,
        help="per-test timeout for generated Python execution",
    )
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        help="required for all executable code suites; use only on an isolated agent",
    )
    parser.add_argument(
        "--log-level",
        choices=["summary", "progress", "detailed"],
        default="progress",
        help="console detail: final summary only, periodic progress, or every repetition",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print a rolling summary every N prompts; 0 disables periodic summaries",
    )
    parser.add_argument(
        "--log-completions-on-mismatch",
        action="store_true",
        help="log decoded baseline/n-gram outputs for non-exact generations",
    )
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
    count_options = {
        "--gsm8k-problems": args.gsm8k_problems,
        "--ifeval-problems": args.ifeval_problems,
        "--math500-problems": args.math500_problems,
        "--longbench-problems-per-task": args.longbench_problems_per_task,
        "--humaneval-problems": args.humaneval_problems,
        "--humanevalplus-problems": args.humanevalplus_problems,
        "--mbppplus-problems": args.mbppplus_problems,
        # "--bigcodebench-problems": args.bigcodebench_problems,
        "--livecodebench-problems": args.livecodebench_problems,
    }
    for option, value in count_options.items():
        if value < 0:
            parser.error(f"{option} cannot be negative")
    if args.longbench_max_input_tokens < 512:
        parser.error("--longbench-max-input-tokens must be at least 512")
    if args.code_execution_timeout <= 0:
        parser.error("--code-execution-timeout must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    modes = parse_modes(parser, args.modes)
    sampling_seeds = []
    if "sampling" in modes:
        sampling_seeds = parse_int_list(
            parser, args.sampling_seeds, "--sampling-seeds", 0, 2**31 - 1
        )
        if args.temperature <= 0:
            parser.error("--temperature must be positive")
        if args.top_k < 0:
            parser.error("--top-k cannot be negative")
        if args.top_k == 1:
            parser.error("--top-k 1 is greedy decoding; use --modes greedy instead")
        if not 0 < args.top_p <= 1:
            parser.error("--top-p must be in (0, 1]")
        if args.repetition_penalty <= 0:
            parser.error("--repetition-penalty must be positive")
        if not 0 <= args.min_new_tokens <= args.max_new_tokens:
            parser.error("--min-new-tokens must be in [0, --max-new-tokens]")
    suites = parse_suites(parser, args.suites)
    args.longbench_tasks = suite_utils.parse_longbench_tasks(
        parser, args.longbench_tasks
    )
    executable_suites = suite_utils.CODE_EXECUTION_SUITES & set(suites)
    if executable_suites and not args.allow_code_execution:
        parser.error(
            f"{', '.join(sorted(executable_suites))} executes generated Python. "
            "Pass --allow-code-execution only on an isolated machine or container."
        )
    if executable_suites:
        print(
            "WARNING: evaluator guards are not security sandboxes. "
            "Run generated code only on an isolated, disposable machine.",
            file=sys.stderr,
        )
    ngram_sizes = parse_int_list(
        parser, args.ngram_size, "--ngram-size", 2, 16
    )
    if args.adaptive_k and not 1 <= args.adaptive_k_min <= 16:
        parser.error("--adaptive-k-min must be in [1, 16]")
    if args.adaptive_k:
        draft_lengths = ["adaptive"]
        runtime_draft_lengths = {"adaptive": args.adaptive_k_min}
    else:
        draft_lengths = parse_int_list(parser, args.k, "--k", 1, 16)
        runtime_draft_lengths = {value: value for value in draft_lengths}
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

    prompt_items = []
    if "mtbench" in suites:
        if args.dataset and not args.builtin:
            task_filter = (
                {task.strip() for task in args.tasks.split(",") if task.strip()}
                if args.tasks else None
            )
            mtbench_items = common.load_dataset_prompts(
                args.dataset,
                task_filter,
                args.limit_per_task,
                mt_bench_by_subcategory=args.by_category,
            )
            if not mtbench_items:
                parser.error("no prompts matched --dataset/--tasks")
            prompt_items.extend(mtbench_items)
        else:
            prompt_items.extend(common.builtin_prompts(args.max_prompts))
    prompt_items.extend(suite_utils.load_additional_suite_prompts(args, suites))
    if not prompt_items:
        parser.error("the selected suites produced no prompts")

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
    sampling_options = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "min_new_tokens": args.min_new_tokens,
    }
    run_configs = []
    if "greedy" in modes:
        run_configs.append(("greedy", 0))
    if "sampling" in modes:
        run_configs.extend(("sampling", seed) for seed in sampling_seeds)

    print(f"onnxruntime_genai={og.__file__}")
    print(f"model={model_path}")
    print(
        f"suites={suites}  prompts={len(prompt_items)}  ngram_sizes={ngram_sizes}  "
        f"K={draft_lengths}  modes={modes}  sampling_seeds={sampling_seeds}  "
        f"chained_lookup={args.chained_lookup}  cooldown={args.cooldown}  "
        f"max_new={args.max_new_tokens}  reps={args.reps}"
    )
    if args.adaptive_k:
        print(
            f"adaptive K enabled: start/floor={args.adaptive_k_min}, "
            "native maximum=16; --k sweep ignored"
        )
    if "sampling" in modes:
        print(
            f"sampling: temperature={args.temperature} top_k={args.top_k} "
            f"top_p={args.top_p} repetition_penalty={args.repetition_penalty} "
            f"min_new_tokens={args.min_new_tokens}"
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
        suite_utils.truncate_prompt_tokens(
            common.encode_prompt(
                tokenizer,
                item["text"],
                chat=not (args.raw or item.get("raw_prompt", False)),
                think=args.think,
            ),
            item,
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
        probe_options = dict(
            ngram_size=ngram_sizes[0],
            max_draft_tokens=runtime_draft_lengths[draft_lengths[0]],
        )
        if args.adaptive_k:
            probe_options.update(
                adaptive_k_bool=True,
                adaptive_k_min=args.adaptive_k_min,
            )
        if args.chained_lookup:
            probe_options["ngram_chained_lookup_bool"] = True
        if args.cooldown:
            probe_options["cooldown_bool"] = True
        probe.set_speculative_options(**probe_options)
        common.verify_speculative_options(probe, probe_options)
        del probe
    except Exception as error:
        raise RuntimeError(
            "The loaded onnxruntime-genai build does not accept the n-gram speculative "
            "options. Rebuild this branch before running the benchmark."
        ) from error

    rows = []
    baselines = {}
    quality_cache = {}
    benchmark_start = time.perf_counter()
    warmup_tokens = min(
        args.max_new_tokens,
        max(16, args.min_new_tokens if "sampling" in modes else 0),
    )

    for mode, seed in run_configs:
        seed_value = seed if mode == "sampling" else ""
        if args.warmup:
            print(
                f"Warming standard {mode} decoding"
                f"{f' (seed={seed})' if mode == 'sampling' else ''} ...",
                flush=True,
            )
            for _ in range(args.warmup):
                run_once(
                    og,
                    model,
                    encoded[0],
                    warmup_tokens,
                    mode=mode,
                    seed=seed,
                    sampling_options=sampling_options,
                )

        for prompt_index, (item, token_ids) in enumerate(zip(prompt_items, encoded)):
            prompt_max_new = suite_utils.generation_limit(
                item, args.max_new_tokens
            )
            decode_rates = []
            e2e_rates = []
            prefill_times = []
            decode_times = []
            reference_tail = None
            baseline_quality = None
            for rep in range(args.reps):
                result = run_once(
                    og,
                    model,
                    token_ids,
                    prompt_max_new,
                    mode=mode,
                    seed=seed,
                    sampling_options=sampling_options,
                )
                decode_rate = (
                    result["new_tokens"] / result["decode_s"]
                    if result["decode_s"] else 0.0
                )
                e2e_time = result["prefill_s"] + result["decode_s"]
                e2e_rate = result["new_tokens"] / e2e_time if e2e_time else 0.0
                decode_rates.append(decode_rate)
                e2e_rates.append(e2e_rate)
                prefill_times.append(result["prefill_s"])
                decode_times.append(result["decode_s"])
                if reference_tail is None:
                    reference_tail = result["tail"]
                elif reference_tail != result["tail"]:
                    raise RuntimeError(
                        f"standard {mode} output was not reproducible for "
                        f"prompt {prompt_index}, seed {seed}"
                    )
                quality = suite_utils.score_completion(
                    item,
                    result["tail"],
                    args.code_execution_timeout,
                    quality_cache,
                    tokenizer.decode,
                )
                baseline_quality = quality
                rows.append({
                    **run_metadata,
                    "mode": mode,
                    "seed": seed_value,
                    "temperature": args.temperature if mode == "sampling" else "",
                    "top_k": args.top_k if mode == "sampling" else "",
                    "top_p": args.top_p if mode == "sampling" else "",
                    "repetition_penalty": (
                        args.repetition_penalty if mode == "sampling" else ""
                    ),
                    "min_new_tokens": args.min_new_tokens if mode == "sampling" else "",
                    "reproducibility_checked": mode == "sampling",
                    "reproducible": True if mode == "sampling" else "",
                    "task": item["task"],
                    "subcategory": item["subcategory"],
                    "question_id": item["question_id"],
                    **quality,
                    "baseline_quality_score": quality["quality_score"],
                    "quality_score_delta": (
                        0.0 if quality["quality_metric"] else ""
                    ),
                    "quality_transition": "",
                    "prompt_id": prompt_index,
                    "rep": rep,
                    "decoder": "standard",
                    "ngram_size": "",
                    "K": "",
                    "prompt_tokens": len(token_ids),
                    "output_token_budget": prompt_max_new,
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
                    "divergence_type": "",
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
                    "estimated_speedup": "",
                    "observed_speedup": "",
                    "peak_process_rss_gib": "",
                })
                if args.log_level == "detailed":
                    print(
                        f"  [baseline {prompt_index + 1}/{len(prompt_items)}] "
                        f"{item['task']}/{item['question_id']} mode={mode} "
                        f"seed={format_seed(mode, seed_value)} rep={rep + 1}: "
                        f"decode={decode_rate:.2f} tok/s e2e={e2e_rate:.2f} tok/s "
                        f"tokens={result['new_tokens']} prefill={result['prefill_s']:.4f}s "
                        f"decode_time={result['decode_s']:.4f}s "
                        f"quality={format_quality_with_reference(quality)}",
                        flush=True,
                    )
            if mode == "sampling" and args.reps == 1:
                reproducibility_result = run_once(
                    og,
                    model,
                    token_ids,
                    prompt_max_new,
                    mode=mode,
                    seed=seed,
                    sampling_options=sampling_options,
                )
                if reproducibility_result["tail"] != reference_tail:
                    raise RuntimeError(
                        f"standard sampling output was not reproducible for "
                        f"prompt {prompt_index}, seed {seed}"
                    )
            baseline_key = (mode, seed, prompt_index)
            baselines[baseline_key] = {
                "decode_rate": statistics.median(decode_rates),
                "e2e_rate": statistics.median(e2e_rates),
                "prefill_s": statistics.median(prefill_times),
                "decode_s": statistics.median(decode_times),
                "tail": reference_tail,
                "quality": baseline_quality,
            }
            if args.log_level == "progress":
                print(
                    f"[baseline {prompt_index + 1}/{len(prompt_items)}] "
                    f"{item['task']}/{item['question_id']} mode={mode} "
                    f"seed={format_seed(mode, seed_value)}: "
                    f"decode={baselines[baseline_key]['decode_rate']:.2f} tok/s "
                    f"e2e={baselines[baseline_key]['e2e_rate']:.2f} tok/s "
                    f"tokens={len(reference_tail)} "
                    f"quality={format_quality_with_reference(baseline_quality)}",
                    flush=True,
                )
            write_results(rows, csv_path, json_path)

    total_configs = len(run_configs) * len(ngram_sizes) * len(draft_lengths)
    config_index = 0
    for mode, seed in run_configs:
        seed_value = seed if mode == "sampling" else ""
        for ngram_size in ngram_sizes:
            for max_draft_tokens in draft_lengths:
                runtime_max_draft_tokens = runtime_draft_lengths[max_draft_tokens]
                config_rows = []
                config_index += 1
                print(
                    f"\n[config {config_index}/{total_configs}] mode={mode}, "
                    f"seed={format_seed(mode, seed_value)}, "
                    f"{format_config(ngram_size, max_draft_tokens)}",
                    flush=True,
                )
                if args.warmup:
                    for _ in range(args.warmup):
                        run_once(
                            og,
                            model,
                            encoded[0],
                            warmup_tokens,
                            mode=mode,
                            seed=seed,
                            sampling_options=sampling_options,
                            ngram_size=ngram_size,
                            max_draft_tokens=runtime_max_draft_tokens,
                            adaptive_k=args.adaptive_k,
                            adaptive_k_min=args.adaptive_k_min,
                            chained_lookup=args.chained_lookup,
                            cooldown=args.cooldown,
                        )

                for prompt_index, (item, token_ids) in enumerate(
                    zip(prompt_items, encoded)
                ):
                    prompt_max_new = suite_utils.generation_limit(
                        item, args.max_new_tokens
                    )
                    baseline = baselines[(mode, seed, prompt_index)]
                    measured_speedups = []
                    measured_acceptance = []
                    measured_exact = []
                    prompt_rows = []
                    reference_ngram_tail = None
                    for rep in range(args.reps):
                        result = run_once(
                            og,
                            model,
                            token_ids,
                            prompt_max_new,
                            mode=mode,
                            seed=seed,
                            sampling_options=sampling_options,
                            ngram_size=ngram_size,
                            max_draft_tokens=runtime_max_draft_tokens,
                            adaptive_k=args.adaptive_k,
                            adaptive_k_min=args.adaptive_k_min,
                            chained_lookup=args.chained_lookup,
                            cooldown=args.cooldown,
                        )
                        if reference_ngram_tail is None:
                            reference_ngram_tail = result["tail"]
                        elif reference_ngram_tail != result["tail"]:
                            raise RuntimeError(
                                f"n-gram {mode} output was not reproducible for "
                                f"prompt {prompt_index}, seed {seed}, "
                                f"{format_config(ngram_size, max_draft_tokens)}"
                            )
                        stats = result["stats"]
                        decode_rate = (
                            result["new_tokens"] / result["decode_s"]
                            if result["decode_s"] else 0.0
                        )
                        e2e_time = result["prefill_s"] + result["decode_s"]
                        e2e_rate = (
                            result["new_tokens"] / e2e_time if e2e_time else 0.0
                        )
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
                        quality = suite_utils.score_completion(
                            item,
                            result["tail"],
                            args.code_execution_timeout,
                            quality_cache,
                            tokenizer.decode,
                        )
                        quality_transition = classify_quality_transition(
                            baseline["quality"], quality
                        )
                        divergence_type = classify_divergence(
                            comparison, len(baseline["tail"]), len(result["tail"])
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
                        row = {
                            **run_metadata,
                            "mode": mode,
                            "seed": seed_value,
                            "temperature": (
                                args.temperature if mode == "sampling" else ""
                            ),
                            "top_k": args.top_k if mode == "sampling" else "",
                            "top_p": args.top_p if mode == "sampling" else "",
                            "repetition_penalty": (
                                args.repetition_penalty if mode == "sampling" else ""
                            ),
                            "min_new_tokens": (
                                args.min_new_tokens if mode == "sampling" else ""
                            ),
                            "reproducibility_checked": (
                                mode == "sampling"
                            ),
                            "reproducible": (
                                True if mode == "sampling" else ""
                            ),
                            "task": item["task"],
                            "subcategory": item["subcategory"],
                            "question_id": item["question_id"],
                            **quality,
                            "baseline_quality_score": baseline["quality"][
                                "quality_score"
                            ],
                            "quality_score_delta": (
                                float(quality["quality_score"])
                                - float(baseline["quality"]["quality_score"])
                                if quality["quality_metric"] else ""
                            ),
                            "quality_transition": quality_transition,
                            "prompt_id": prompt_index,
                            "rep": rep,
                            "decoder": "ngram",
                            "ngram_size": ngram_size,
                            "K": max_draft_tokens,
                            "chained_lookup": args.chained_lookup,
                            "cooldown": args.cooldown,
                            "adaptive_k": args.adaptive_k,
                            "adaptive_k_min": (
                                args.adaptive_k_min if args.adaptive_k else ""
                            ),
                            "effective_k": int(
                                stats.get("effective_k", runtime_max_draft_tokens)
                            ),
                            "adaptive_k_increases": int(
                                stats.get("adaptive_k_increases", 0)
                            ),
                            "adaptive_k_decreases": int(
                                stats.get("adaptive_k_decreases", 0)
                            ),
                            "adaptive_k_observations": int(
                                stats.get("adaptive_k_observations", 0)
                            ),
                            "adaptive_k_probes": int(
                                stats.get("adaptive_k_probes", 0)
                            ),
                            "adaptive_k_throughput": round(
                                float(stats.get("adaptive_k_throughput", 0.0)), 6
                            ),
                            "prompt_tokens": len(token_ids),
                            "output_token_budget": prompt_max_new,
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
                            "divergence_type": divergence_type,
                            "rounds": rounds,
                            "completed_rounds": int(stats["completed_rounds"]),
                            "interrupted_rounds": int(stats["interrupted_rounds"]),
                            "active_rounds": int(stats["active_rounds"]),
                            "draft_proposed": proposed,
                            "draft_evaluated": evaluated,
                            "draft_accepted": accepted,
                            "acceptance_rate": round(
                                float(stats["acceptance_rate"]), 6
                            ),
                            "avg_draft_tokens_per_round": round(
                                float(stats["avg_draft_tokens_per_round"]), 6
                            ),
                            "mean_accepted_tokens_per_round": round(
                                mean_accepted, 6
                            ),
                            "mean_emitted_tokens_per_round": round(
                                float(stats["mean_emitted_tokens_per_round"]),
                                6,
                            ),
                            "expected_tokens_per_round": round(
                                float(stats["expected_tokens_per_round"]), 6
                            ),
                            "corrections": int(stats["correction_tokens"]),
                            "bonuses": int(stats["bonus_tokens"]),
                            "tokens_queued": int(stats["tokens_queued"]),
                            "tokens_emitted": int(stats["tokens_emitted"]),
                            "tokens_discarded": int(stats["tokens_discarded"]),
                            "tokens_buffered": int(stats["tokens_buffered"]),
                            "draft_forward_passes": int(
                                stats["draft_forward_passes"]
                            ),
                            "target_forward_passes": target_passes,
                            "target_verify_forward_passes": int(
                                stats.get("target_verify_forward_passes", 0)
                            ),
                            "target_reanchor_forward_passes": int(
                                stats.get("target_reanchor_forward_passes", 0)
                            ),
                            "target_reconciliation_forward_passes": int(
                                stats.get("target_reconciliation_forward_passes", 0)
                            ),
                            "target_passes_per_token": round(
                                target_passes_per_token, 6
                            ),
                            "cooldown_entries": int(stats.get("cooldown_entries", 0)),
                            "cooldown_steps": int(stats.get("cooldown_steps", 0)),
                            "cooldown_remaining": int(
                                stats.get("cooldown_remaining", 0)
                            ),
                            "standard_fallback_steps": int(
                                stats.get("standard_fallback_steps", 0)
                            ),
                            "full_accept_rounds": int(
                                stats.get("full_accept_rounds", 0)
                            ),
                            "partial_accept_rounds": int(
                                stats.get("partial_accept_rounds", 0)
                            ),
                            "zero_accept_rounds": int(
                                stats.get("zero_accept_rounds", 0)
                            ),
                            "ngram_lookup_hits": int(
                                stats.get("ngram_lookup_hits", 0)
                            ),
                            "ngram_lookup_misses": int(
                                stats.get("ngram_lookup_misses", 0)
                            ),
                            "ngram_lookup_tokens_proposed": int(
                                stats.get("ngram_lookup_tokens_proposed", 0)
                            ),
                            "ngram_chained_tokens_proposed": int(
                                stats.get("ngram_chained_tokens_proposed", 0)
                            ),
                            "ngram_grammar_candidate_rejections": int(
                                stats.get("ngram_grammar_candidate_rejections", 0)
                            ),
                            "ngram_history_syncs": int(
                                stats.get("ngram_history_syncs", 0)
                            ),
                            "ngram_history_tokens_synced": int(
                                stats.get("ngram_history_tokens_synced", 0)
                            ),
                            "total_draft_ms": round(
                                float(stats.get("total_draft_ms", 0.0)), 6
                            ),
                            "total_target_ms": round(
                                float(stats.get("total_target_ms", 0.0)), 6
                            ),
                            "total_reconciliation_ms": round(
                                float(stats.get("total_reconciliation_ms", 0.0)), 6
                            ),
                            "total_target_verify_ms": round(
                                float(stats.get("total_target_verify_ms", 0.0)), 6
                            ),
                            "total_target_reanchor_ms": round(
                                float(stats.get("total_target_reanchor_ms", 0.0)), 6
                            ),
                            "total_ngram_history_sync_ms": round(
                                float(stats.get("total_ngram_history_sync_ms", 0.0)), 6
                            ),
                            "total_ngram_lookup_ms": round(
                                float(stats["total_ngram_lookup_ms"]), 6
                            ),
                            "avg_draft_ms_per_token": round(
                                float(stats["avg_draft_ms_per_token"]), 6
                            ),
                            "avg_target_ms_per_round": round(
                                float(stats["avg_target_ms_per_round"]), 6
                            ),
                            "target_baseline_ms_per_token": round(
                                float(stats["target_baseline_ms_per_token"]), 6
                            ),
                            "target_overhead_ratio": round(
                                float(stats["target_overhead_ratio"]), 6
                            ),
                            "formula_supported": bool(stats["formula_supported"]),
                            "estimated_speedup": round(
                                float(stats["estimated_speedup"]), 6
                            ),
                            "observed_speedup": round(
                                float(stats["observed_speedup"]), 6
                            ),
                            "peak_process_rss_gib": "",
                        }
                        rows.append(row)
                        config_rows.append(row)
                        prompt_rows.append(row)
                        if args.log_level == "detailed":
                            print_detailed_run(item, rep, baseline, row)
                        if (
                            args.log_completions_on_mismatch
                            and not comparison["exact_match"]
                        ):
                            print_mismatch_completions(
                                tokenizer, baseline["tail"], result["tail"]
                            )

                    if mode == "sampling" and args.reps == 1:
                        reproducibility_result = run_once(
                            og,
                            model,
                            token_ids,
                            prompt_max_new,
                            mode=mode,
                            seed=seed,
                            sampling_options=sampling_options,
                            ngram_size=ngram_size,
                            max_draft_tokens=runtime_max_draft_tokens,
                            adaptive_k=args.adaptive_k,
                            adaptive_k_min=args.adaptive_k_min,
                            chained_lookup=args.chained_lookup,
                            cooldown=args.cooldown,
                        )
                        if reproducibility_result["tail"] != reference_ngram_tail:
                            raise RuntimeError(
                                f"n-gram sampling output was not reproducible for "
                                f"prompt {prompt_index}, seed {seed}, "
                                f"{format_config(ngram_size, max_draft_tokens)}"
                            )

                    if args.log_level in ("progress", "detailed"):
                        representative = prompt_rows[0]
                        print(
                            f"  {item['task']}/{item['question_id']}: "
                            f"decode={statistics.median(measured_speedups):.2f}x "
                            f"e2e={statistics.median(row['speedup_e2e'] for row in prompt_rows):.2f}x "
                            f"baseline={baseline['decode_rate']:.2f} tok/s "
                            f"ngram={statistics.median(row['decode_tok_s'] for row in prompt_rows):.2f} tok/s"
                        )
                        print(
                            f"    spec: accept={statistics.median(measured_acceptance):.1%} "
                            f"accepted/round={statistics.median(row['mean_accepted_tokens_per_round'] for row in prompt_rows):.2f} "
                            f"emitted/round={statistics.median(row['mean_emitted_tokens_per_round'] for row in prompt_rows):.2f} "
                            f"target_passes/token={statistics.median(row['target_passes_per_token'] for row in prompt_rows):.3f}"
                        )
                        print(
                            f"    target: verify={sum(row['target_verify_forward_passes'] for row in prompt_rows)} "
                            f"reanchor={sum(row['target_reanchor_forward_passes'] for row in prompt_rows)} "
                            f"reconcile={sum(row['target_reconciliation_forward_passes'] for row in prompt_rows)}"
                        )
                        print(
                            f"    lifecycle: completed={sum(row['completed_rounds'] for row in prompt_rows)} "
                            f"interrupted={sum(row['interrupted_rounds'] for row in prompt_rows)} "
                            f"discarded={sum(row['tokens_discarded'] for row in prompt_rows)} "
                            f"buffered={sum(row['tokens_buffered'] for row in prompt_rows)}"
                        )
                        print(
                            f"    n-gram: hits={sum(row['ngram_lookup_hits'] for row in prompt_rows)} "
                            f"misses={sum(row['ngram_lookup_misses'] for row in prompt_rows)} "
                            f"lookup_tokens={sum(row['ngram_lookup_tokens_proposed'] for row in prompt_rows)} "
                            f"chained_tokens={sum(row['ngram_chained_tokens_proposed'] for row in prompt_rows)} "
                            f"lookup_ms={sum(row['total_ngram_lookup_ms'] for row in prompt_rows):.1f}"
                        )
                        if args.cooldown:
                            print(
                                f"    cooldown: entries={sum(row['cooldown_entries'] for row in prompt_rows)} "
                                f"steps={sum(row['cooldown_steps'] for row in prompt_rows)} "
                                f"fallback={sum(row['standard_fallback_steps'] for row in prompt_rows)} "
                                f"accept_rounds={sum(row['full_accept_rounds'] for row in prompt_rows)}/"
                                f"{sum(row['partial_accept_rounds'] for row in prompt_rows)}/"
                                f"{sum(row['zero_accept_rounds'] for row in prompt_rows)}"
                            )
                        if args.adaptive_k:
                            print(
                                f"    adaptive K: start={args.adaptive_k_min} "
                                f"final_p50={statistics.median(row['effective_k'] for row in prompt_rows):.1f} "
                                f"final_range={min(row['effective_k'] for row in prompt_rows)}-"
                                f"{max(row['effective_k'] for row in prompt_rows)} "
                                f"moves=+{sum(row['adaptive_k_increases'] for row in prompt_rows)}"
                                f"/-{sum(row['adaptive_k_decreases'] for row in prompt_rows)} "
                                f"probes={sum(row['adaptive_k_probes'] for row in prompt_rows)}"
                            )
                        print(
                            f"    output: baseline={len(baseline['tail'])} tokens "
                            f"ngram={representative['new_tokens']} tokens "
                            f"exact={sum(measured_exact)}/{len(measured_exact)} "
                            f"match={statistics.median(row['token_match_rate'] for row in prompt_rows):.1%} "
                            f"first_diff={representative['first_difference']} "
                            f"type={representative['divergence_type']}"
                        )
                        print(
                            f"    quality: baseline={format_quality_with_reference(baseline['quality'])} "
                            f"ngram={format_quality_with_reference(representative)} "
                            f"transition={representative['quality_transition'] or 'not_scored'}",
                            flush=True,
                        )
                    completed = prompt_index + 1
                    if (
                        args.log_level in ("progress", "detailed")
                        and args.progress_every
                        and (
                            completed % args.progress_every == 0
                            or completed == len(prompt_items)
                        )
                    ):
                        print_progress_summary(
                            config_rows,
                            completed,
                            len(prompt_items),
                            ngram_size,
                            max_draft_tokens,
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
            "chained_lookup": args.chained_lookup,
            "cooldown": args.cooldown,
            "peak_rss": monitor.peak_rss_gib,
        },
    )


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
r"""Benchmark EAGLE-3 tree speculative decoding against its target-only model.

The benchmark is intentionally strict:

* target-only and EAGLE use the same target graph, external data, tokenizer,
  prompt token IDs, CUDA provider configuration, and greedy search settings;
* the target-only model is unloaded before the EAGLE model is loaded;
* every measured EAGLE output must exactly match target-only token IDs;
* per-token statistics polling is confined to a separate untimed telemetry run;
* measured wall-clock throughput is authoritative (the generic speculative
  speedup formula does not describe EAGLE tree decoding);
* JSON and CSV evidence is atomically checkpointed after each prompt.

Examples:
    # Four-prompt, 32-token correctness smoke test.
    python benchmark_eagle.py --eagle-model C:\models\phase5-runtime-bf16 \
        --builtin --output-lengths 32 --warmups 2 --repetitions 3 \
        -o results\eagle_smoke

    # Two prompts from each Spec-Bench category.
    python benchmark_eagle.py --eagle-model C:\models\phase5-runtime-bf16 \
        --limit-per-category 2 --output-lengths 128 \
        -o results\eagle_specbench_sample

    # Shared quality suites (same loaders/scorers as the other benchmarks).
    python benchmark_eagle.py --eagle-model C:\models\phase5-runtime-bf16 \
        --suites gsm8k,ifeval,math500 \
        --gsm8k-problems 50 --ifeval-problems 50 --math500-problems 50 \
        -o results\eagle_quality

    # Generated-code suites require explicit consent and an isolated machine.
    python benchmark_eagle.py --eagle-model C:\models\phase5-runtime-bf16 \
        --suites humaneval,humanevalplus,mbppplus,livecodebench \
        --allow-code-execution -o results\eagle_code_quality

    # Model-independent helper validation.
    python benchmark_eagle.py --self-test
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import csv
import ctypes
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, TextIO

import benchmark_suites as suite_utils


CONFIG_FILENAME = "genai_config.json"
MAX_DRAFT_TOKENS = 8
TREE_TOTAL_TOKENS = 60
TREE_DRAFT_NODES = TREE_TOTAL_TOKENS - 1
TREE_DEPTH = 7
TREE_TOP_K = 10
TREE_SCORED_CANDIDATES = 710
EAGLE_CALLS_PER_FULL_ROUND = 8
MT_BENCH_SUBCATEGORIES = {
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
}
QUALITY_FIELDS = (
    "quality_metric",
    "quality_score_type",
    "quality_score",
    "baseline_quality_score",
    "quality_score_delta",
    "quality_prediction",
    "quality_reference_answer",
    "quality_detail",
    "quality_transition",
)

BUILTIN_PROMPTS = [
    "The capital of France is",
    "The result of 2 + 2 is",
    "1, 1, 2, 3, 5,",
    "Write a short greeting:",
]

COUNT_STATS = (
    "rounds",
    "completed_rounds",
    "interrupted_rounds",
    "active_rounds",
    "draft_tokens_proposed",
    "draft_tokens_evaluated",
    "draft_tokens_accepted",
    "correction_tokens",
    "bonus_tokens",
    "tokens_queued",
    "tokens_emitted",
    "tokens_discarded",
    "tokens_buffered",
    "draft_forward_passes",
    "target_forward_passes",
    "effective_k",
    "adaptive_k_increases",
    "adaptive_k_decreases",
    "adaptive_k_observations",
    "adaptive_k_probes",
    "cooldown_entries",
    "cooldown_steps",
    "cooldown_remaining",
    "standard_fallback_steps",
    "full_accept_rounds",
    "partial_accept_rounds",
    "zero_accept_rounds",
    "target_verify_forward_passes",
    "target_reanchor_forward_passes",
    "target_reconciliation_forward_passes",
    "ngram_lookup_hits",
    "ngram_lookup_misses",
    "ngram_lookup_tokens_proposed",
    "ngram_chained_tokens_proposed",
    "ngram_grammar_candidate_rejections",
    "ngram_history_syncs",
    "ngram_history_tokens_synced",
)

FLOAT_STATS = (
    "total_draft_ms",
    "total_target_ms",
    "total_reconciliation_ms",
    "total_target_verify_ms",
    "total_target_reanchor_ms",
    "total_ngram_history_sync_ms",
    "total_ngram_lookup_ms",
    "avg_draft_ms_per_token",
    "acceptance_rate",
    "avg_draft_tokens_per_round",
    "mean_emitted_tokens_per_round",
    "expected_tokens_per_round",
    "avg_target_ms_per_round",
    "target_baseline_ms_per_token",
    "target_overhead_ratio",
    "estimated_speedup",
    "observed_speedup",
    "adaptive_k_throughput",
)

BOOL_STATS = ("formula_supported",)
REQUIRED_STATS = frozenset((*COUNT_STATS, *FLOAT_STATS, *BOOL_STATS))

TREE_METRIC_FIELDS = (
    "accepted_tokens_per_verification",
    "emitted_tokens_per_verification",
    "selected_path_utilization",
    "tree_node_yield",
    "eagle_calls_per_emitted_token",
    "target_verifications_per_emitted_token",
    "native_full_accept_proportion",
    "native_partial_accept_proportion",
    "native_zero_accept_proportion",
    "completed_round_proportion",
    "interrupted_round_proportion",
    "unattributed_decode_ms",
)

CSV_COLUMNS = (
    "run_id",
    "decoder",
    "task",
    "subcategory",
    "category",
    "question_id",
    "prompt_index",
    "configured_output_token_budget",
    "output_token_budget",
    "repetition",
    "prompt_tokens",
    "generated_tokens",
    "prefill_s",
    "first_decode_s",
    "ttft_s",
    "decode_s",
    "end_to_end_s",
    "decode_tokens_per_s",
    "end_to_end_tokens_per_s",
    "baseline_decode_tokens_per_s",
    "baseline_end_to_end_tokens_per_s",
    "decode_speedup",
    "end_to_end_speedup",
    "output_token_sha256",
    "expected_output_token_sha256",
    "exact_match",
    "token_position_match_rate",
    "first_difference_index",
    "expected_token_at_difference",
    "actual_token_at_difference",
    "divergence_type",
    *QUALITY_FIELDS,
    *COUNT_STATS,
    *BOOL_STATS,
    *FLOAT_STATS,
    *TREE_METRIC_FIELDS,
    "phase_baseline_process_rss_mib",
    "phase_peak_process_rss_mib",
    "phase_baseline_total_gpu_used_mib",
    "phase_peak_total_gpu_used_mib",
    "phase_baseline_process_gpu_used_mib",
    "phase_peak_process_gpu_used_mib",
)


class CorrectnessError(RuntimeError):
    """Raised after correctness evidence has been checkpointed."""


class _Tee:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def geometric_mean(values: Sequence[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def parse_output_lengths(value: str) -> list[int]:
    values: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        length = int(item)
        if length <= 0:
            raise ValueError("Output lengths must be positive integers")
        if length not in values:
            values.append(length)
    if not values:
        raise ValueError("At least one output length is required")
    return values


def parse_categories(value: str | None) -> list[str] | None:
    if value is None:
        return None
    categories = [item.strip() for item in value.split(",") if item.strip()]
    if not categories:
        raise ValueError("--categories must contain at least one category")
    return list(dict.fromkeys(categories))


def parse_suite_names(value: str) -> list[str]:
    suites = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if normalized:
            suites.append(
                suite_utils.SUITE_ALIASES.get(normalized, normalized)
            )
    if not suites:
        raise ValueError("--suites must contain at least one suite")
    unknown = sorted(set(suites) - suite_utils.SUPPORTED_SUITES)
    if unknown:
        raise ValueError(
            "--suites contains unsupported values: "
            f"{', '.join(unknown)}; choose from "
            f"{', '.join(sorted(suite_utils.SUPPORTED_SUITES))}"
        )
    return list(dict.fromkeys(suites))


def parse_longbench_tasks(value: str) -> list[str]:
    tasks = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not tasks:
        raise ValueError("--longbench-tasks must contain at least one task")
    unknown = sorted(set(tasks) - suite_utils.LONG_BENCH_TASKS.keys())
    if unknown:
        raise ValueError(
            "--longbench-tasks contains unsupported values: "
            f"{', '.join(unknown)}; choose from "
            f"{', '.join(suite_utils.LONG_BENCH_TASKS)}"
        )
    return list(dict.fromkeys(tasks))


def executable_suite_selection(
    suites: Sequence[str],
    allow_code_execution: bool,
) -> set[str]:
    executable = suite_utils.CODE_EXECUTION_SUITES & set(suites)
    if executable and not allow_code_execution:
        raise ValueError(
            f"{', '.join(sorted(executable))} executes generated Python. "
            "Pass --allow-code-execution only on an isolated machine or container."
        )
    return executable


def token_sequence_sha256(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(tokens)))
    for token in tokens:
        digest.update(struct.pack("<q", int(token)))
    return digest.hexdigest()


def compare_tokens(expected: Sequence[int], actual: Sequence[int]) -> dict[str, Any]:
    common = min(len(expected), len(actual))
    matching = sum(
        expected[index] == actual[index]
        for index in range(common)
    )
    first_difference = next(
        (
            index
            for index in range(common)
            if expected[index] != actual[index]
        ),
        common if len(expected) != len(actual) else None,
    )
    denominator = max(len(expected), len(actual))
    if first_difference is None:
        divergence_type = "exact"
    elif first_difference == common:
        divergence_type = (
            "actual_shorter" if len(actual) < len(expected) else "actual_longer"
        )
    else:
        divergence_type = "token_divergence"
    return {
        "exact_match": list(expected) == list(actual),
        "token_position_match_rate": matching / denominator if denominator else 1.0,
        "first_difference_index": first_difference,
        "expected_token_at_difference": (
            int(expected[first_difference])
            if first_difference is not None and first_difference < len(expected)
            else None
        ),
        "actual_token_at_difference": (
            int(actual[first_difference])
            if first_difference is not None and first_difference < len(actual)
            else None
        ),
        "divergence_type": divergence_type,
    }


def derive_target_config(eagle_config: dict[str, Any]) -> dict[str, Any]:
    target_config = copy.deepcopy(eagle_config)
    model = target_config.get("model")
    if not isinstance(model, dict) or "eagle" not in model:
        raise ValueError("EAGLE configuration is missing model.eagle")
    if "speculative" not in target_config:
        raise ValueError("EAGLE configuration is missing top-level speculative settings")
    del model["eagle"]
    del target_config["speculative"]
    return target_config


def _provider_names(component: dict[str, Any], label: str) -> list[str]:
    session_options = component.get("session_options")
    if not isinstance(session_options, dict):
        raise ValueError(f"{label}.session_options must be an object")
    provider_options = session_options.get("provider_options")
    if not isinstance(provider_options, list) or not provider_options:
        raise ValueError(f"{label} must explicitly configure the CUDA provider")
    names: list[str] = []
    for entry in provider_options:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"{label}.provider_options entries must name one provider")
        names.append(str(next(iter(entry))).lower())
    return names


def validate_eagle_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model")
    search = config.get("search")
    speculative = config.get("speculative")
    if not isinstance(model, dict):
        raise ValueError("Configuration is missing model")
    if not isinstance(search, dict):
        raise ValueError("Configuration is missing search")
    if not isinstance(speculative, dict):
        raise ValueError("Configuration is missing speculative")
    decoder = model.get("decoder")
    eagle = model.get("eagle")
    if not isinstance(decoder, dict) or not isinstance(eagle, dict):
        raise ValueError("Configuration must contain model.decoder and model.eagle")
    if model.get("type") != "qwen3":
        raise ValueError(
            "This Phase 5 harness currently requires model.type='qwen3'"
        )

    decoder_providers = _provider_names(decoder, "model.decoder")
    eagle_providers = _provider_names(eagle, "model.eagle")
    if decoder_providers != ["cuda"] or eagle_providers != ["cuda"]:
        raise ValueError(
            "This harness requires target and EAGLE sessions to use only CUDA; "
            f"got decoder={decoder_providers}, eagle={eagle_providers}"
        )

    expected_values = {
        "model.eagle.total_tokens": (eagle.get("total_tokens"), TREE_TOTAL_TOKENS),
        "model.eagle.depth": (eagle.get("depth"), TREE_DEPTH),
        "model.eagle.top_k": (eagle.get("top_k"), TREE_TOP_K),
        "speculative.max_draft_tokens": (
            speculative.get("max_draft_tokens"),
            MAX_DRAFT_TOKENS,
        ),
        "search.num_beams": (search.get("num_beams", 1), 1),
        "search.num_return_sequences": (
            search.get("num_return_sequences", 1),
            1,
        ),
        "search.past_present_share_buffer": (
            search.get("past_present_share_buffer", False),
            False,
        ),
    }
    mismatches = [
        f"{name}={actual!r} (expected {expected!r})"
        for name, (actual, expected) in expected_values.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("Unsupported EAGLE runtime contract: " + "; ".join(mismatches))
    if bool(search.get("do_sample", False)):
        raise ValueError("EAGLE benchmark configuration must default to greedy search")
    if float(search.get("repetition_penalty", 1.0)) != 1.0:
        raise ValueError("EAGLE v0 requires repetition_penalty=1.0")
    if int(search.get("min_length", 0)) != 0:
        raise ValueError("EAGLE v0 requires min_length=0")

    return {
        "max_draft_tokens": MAX_DRAFT_TOKENS,
        "tree_total_tokens": TREE_TOTAL_TOKENS,
        "tree_draft_nodes": TREE_DRAFT_NODES,
        "tree_depth": TREE_DEPTH,
        "tree_top_k": TREE_TOP_K,
        "tree_scored_candidates": TREE_SCORED_CANDIDATES,
        "target_verifications_per_round": 1,
        "eagle_calls_per_full_round": EAGLE_CALLS_PER_FULL_ROUND,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _safe_bundle_path(bundle: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a traversal-free relative path: {relative!r}")
    resolved_bundle = bundle.resolve()
    resolved = (resolved_bundle / candidate).resolve()
    try:
        resolved.relative_to(resolved_bundle)
    except ValueError as error:
        raise ValueError(f"{label} escapes the model bundle: {relative!r}") from error
    return resolved


def decoder_artifacts(bundle: Path, config: dict[str, Any]) -> list[Path]:
    filename = config["model"]["decoder"].get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("model.decoder.filename must be a non-empty string")
    graph = _safe_bundle_path(bundle, filename, "model.decoder.filename")
    if not graph.is_file():
        raise FileNotFoundError(f"Target decoder graph does not exist: {graph}")
    artifacts = [graph]
    prefix = graph.name + "."
    artifacts.extend(
        sorted(
            (
                path
                for path in graph.parent.iterdir()
                if path.is_file() and path.name.startswith(prefix)
            ),
            key=lambda path: path.name,
        )
    )
    return artifacts


def eagle_graph_path(bundle: Path, config: dict[str, Any]) -> Path:
    filename = config["model"]["eagle"].get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("model.eagle.filename must be a non-empty string")
    graph = _safe_bundle_path(bundle, filename, "model.eagle.filename")
    if not graph.is_file():
        raise FileNotFoundError(f"EAGLE graph does not exist: {graph}")
    return graph


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    relative = str(path.relative_to(root)) if root is not None else str(path)
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_target_bundle(
    eagle_bundle: Path,
    target_bundle: Path,
    eagle_config: dict[str, Any],
) -> dict[str, Any]:
    eagle_root = eagle_bundle.resolve()
    target_root = target_bundle.resolve()
    target_config_path = target_root / CONFIG_FILENAME
    if not target_config_path.is_file():
        raise FileNotFoundError(f"Target configuration does not exist: {target_config_path}")
    actual_target_config = _load_json(target_config_path)
    expected_target_config = derive_target_config(eagle_config)
    if actual_target_config != expected_target_config:
        raise ValueError(
            "Supplied target-only configuration differs from the EAGLE configuration "
            "after removing only model.eagle and top-level speculative"
        )

    eagle_artifacts = decoder_artifacts(eagle_root, eagle_config)
    target_artifacts = decoder_artifacts(target_root, actual_target_config)
    eagle_relative = [path.relative_to(eagle_root) for path in eagle_artifacts]
    target_relative = [path.relative_to(target_root) for path in target_artifacts]
    if eagle_relative != target_relative:
        raise ValueError(
            "Supplied target-only decoder artifact set differs from EAGLE: "
            f"{target_relative!r} != {eagle_relative!r}"
        )

    comparisons = []
    for relative in eagle_relative:
        source = eagle_root / relative
        target = target_root / relative
        source_hash = sha256_file(source)
        target_hash = sha256_file(target)
        comparisons.append(
            {
                "path": str(relative),
                "eagle_sha256": source_hash,
                "target_sha256": target_hash,
                "match": source_hash == target_hash,
            }
        )
    mismatches = [item["path"] for item in comparisons if not item["match"]]
    if mismatches:
        raise ValueError(
            "Supplied target-only decoder artifacts differ from EAGLE: "
            + ", ".join(mismatches)
        )
    return {
        "mode": "supplied",
        "path": str(target_root),
        "config_match": True,
        "decoder_artifacts": comparisons,
    }


def stage_target_bundle(
    eagle_bundle: Path,
    eagle_config: dict[str, Any],
    staging_root: Path | None,
) -> tuple[Path, dict[str, Any]]:
    source_root = eagle_bundle.resolve()
    root = staging_root.resolve() if staging_root else source_root.parent
    if root == source_root or source_root in root.parents:
        raise ValueError("The target staging root cannot be inside the EAGLE bundle")
    root.mkdir(parents=True, exist_ok=True)
    staged = Path(
        tempfile.mkdtemp(prefix=f".{eagle_bundle.name}-target-only-", dir=root)
    )
    eagle_graph = eagle_graph_path(source_root, eagle_config)
    excluded_prefix = eagle_graph.name + "."
    linked_files = 0
    copied_files = 0
    linked_bytes = 0
    copied_bytes = 0
    link_failures: list[dict[str, str]] = []
    try:
        for source in sorted(source_root.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"Model bundles may not contain symbolic links: {source}")
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            if relative == Path(CONFIG_FILENAME):
                continue
            if source == eagle_graph or (
                source.parent == eagle_graph.parent
                and source.name.startswith(excluded_prefix)
            ):
                continue
            destination = staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            size = source.stat().st_size
            try:
                os.link(source, destination)
                linked_files += 1
                linked_bytes += size
            except OSError as error:
                shutil.copy2(source, destination)
                copied_files += 1
                copied_bytes += size
                link_failures.append(
                    {"path": str(relative), "reason": str(error)}
                )

        target_config = derive_target_config(eagle_config)
        with (staged / CONFIG_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(target_config, file, indent=2, ensure_ascii=False)
            file.write("\n")
        decoder_artifacts(staged, target_config)
    except BaseException:
        shutil.rmtree(staged)
        raise

    return staged, {
        "mode": "staged",
        "path": str(staged),
        "staging_root": str(root),
        "linked_files": linked_files,
        "copied_files": copied_files,
        "linked_bytes": linked_bytes,
        "copied_bytes": copied_bytes,
        "link_failures": link_failures,
        "excluded_eagle_graph": str(eagle_graph.relative_to(source_root)),
        "cleaned": False,
    }


def load_prompt_items(
    dataset: Path,
    *,
    builtin: bool,
    custom_prompts: Sequence[str] | None,
    tasks: Sequence[str] | None,
    categories: Sequence[str] | None,
    limit_per_task: int,
    by_category: bool,
    max_prompts: int,
) -> list[dict[str, Any]]:
    if custom_prompts:
        items = [
            {
                "question_id": f"custom-{index}",
                "task": "custom",
                "subcategory": "custom",
                "category": "custom",
                "text": text,
                "source": "command_line",
            }
            for index, text in enumerate(custom_prompts)
        ]
        return items[:max_prompts] if max_prompts else items
    if builtin:
        items = [
            {
                "question_id": f"builtin-{index}",
                "task": "builtin",
                "subcategory": "builtin",
                "category": "builtin",
                "text": text,
                "source": "builtin",
            }
            for index, text in enumerate(BUILTIN_PROMPTS)
        ]
        return items[:max_prompts] if max_prompts else items
    if not dataset.is_file():
        raise FileNotFoundError(f"Spec-Bench dataset does not exist: {dataset}")

    requested_tasks = set(tasks or ())
    requested_categories = set(categories or ())
    buckets: collections.OrderedDict[str, list[dict[str, Any]]] = (
        collections.OrderedDict()
    )
    with dataset.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{dataset}:{line_number} is not a JSON object")
            category = record.get("category")
            turns = record.get("turns")
            if not isinstance(category, str) or not category:
                raise ValueError(f"{dataset}:{line_number} has no category")
            if not isinstance(turns, list) or not turns or not isinstance(turns[0], str):
                raise ValueError(f"{dataset}:{line_number} has no string first turn")
            task = (
                category
                if by_category or category not in MT_BENCH_SUBCATEGORIES
                else "mt_bench"
            )
            if requested_tasks and task not in requested_tasks:
                continue
            if requested_categories and category not in requested_categories:
                continue
            buckets.setdefault(task, []).append(
                {
                    "question_id": record.get("question_id", line_number),
                    "task": task,
                    "subcategory": category,
                    "category": category,
                    "text": turns[0],
                    "source": str(dataset),
                }
            )

    available_tasks = set(buckets)
    missing_tasks = requested_tasks - available_tasks
    if missing_tasks:
        raise ValueError(
            "Requested tasks were not found: " + ", ".join(sorted(missing_tasks))
        )
    loaded_categories = {
        item["category"]
        for task_items in buckets.values()
        for item in task_items
    }
    missing_categories = requested_categories - loaded_categories
    if missing_categories:
        raise ValueError(
            "Requested categories were not found: "
            + ", ".join(sorted(missing_categories))
        )

    items: list[dict[str, Any]] = []
    for task_items in buckets.values():
        if not limit_per_task or limit_per_task >= len(task_items):
            items.extend(task_items)
            continue
        subcategories: collections.OrderedDict[
            str, list[dict[str, Any]]
        ] = collections.OrderedDict()
        for item in task_items:
            subcategories.setdefault(item["subcategory"], []).append(item)
        selected = []
        round_index = 0
        while len(selected) < limit_per_task:
            advanced = False
            for subcategory_items in subcategories.values():
                if round_index < len(subcategory_items):
                    selected.append(subcategory_items[round_index])
                    advanced = True
                    if len(selected) == limit_per_task:
                        break
            if not advanced:
                break
            round_index += 1
        items.extend(selected)
    if not items:
        raise ValueError("Prompt selection produced no prompts")
    return items


def load_selected_prompts(
    args: argparse.Namespace | SimpleNamespace,
    suites: Sequence[str],
    dataset: Path,
    tasks: Sequence[str] | None,
    categories: Sequence[str] | None,
) -> list[dict[str, Any]]:
    prompt_items: list[dict[str, Any]] = []
    if "mtbench" in suites:
        limit_per_task = (
            args.limit_per_category
            if args.limit_per_category is not None
            else args.limit_per_task
        )
        prompt_items.extend(
            load_prompt_items(
                dataset,
                builtin=args.builtin,
                custom_prompts=args.prompt,
                tasks=tasks,
                categories=categories,
                limit_per_task=limit_per_task,
                by_category=(
                    args.by_category
                    or args.limit_per_category is not None
                    or categories is not None
                ),
                max_prompts=args.max_prompts,
            )
        )
    prompt_items.extend(suite_utils.load_additional_suite_prompts(args, suites))
    if not prompt_items:
        raise ValueError("The selected suites produced no prompts")

    normalized = []
    for item in prompt_items:
        value = dict(item)
        task = str(value.get("task") or value.get("category") or "unknown")
        subcategory = str(value.get("subcategory") or task)
        value["task"] = task
        value["subcategory"] = subcategory
        value["category"] = str(value.get("category") or subcategory)
        value["source"] = str(value.get("source") or f"suite:{task}")
        normalized.append(value)
    return normalized


def build_execution_cases(
    prompt_items: Sequence[dict[str, Any]],
    configured_output_lengths: Sequence[int],
) -> list[dict[str, int]]:
    cases = []
    seen: set[tuple[int, int]] = set()
    for prompt_index, item in enumerate(prompt_items):
        for configured_budget in configured_output_lengths:
            output_budget = suite_utils.generation_limit(
                item,
                configured_budget,
            )
            key = (prompt_index, output_budget)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                {
                    "prompt_index": prompt_index,
                    "configured_output_token_budget": configured_budget,
                    "output_token_budget": output_budget,
                }
            )
    return cases


def prompt_provenance_record(
    item: dict[str, Any],
    prompt_index: int,
    token_ids: Sequence[int],
) -> dict[str, Any]:
    record = {
        "prompt_index": prompt_index,
        "task": item["task"],
        "subcategory": item["subcategory"],
        "category": item["category"],
        "question_id": item["question_id"],
        "source": item["source"],
        "raw_prompt": bool(item.get("raw_prompt", False)),
        "quality_metric": item.get("quality_metric", ""),
        "quality_score_type": item.get("quality_score_type", ""),
        "max_input_tokens": item.get("max_input_tokens"),
        "max_output_tokens": item.get("max_output_tokens"),
        "prompt_tokens": len(token_ids),
        "prompt_token_sha256": token_sequence_sha256(token_ids),
        "prompt_text_sha256": hashlib.sha256(
            item["text"].encode("utf-8")
        ).hexdigest(),
    }
    if item["source"] in {"builtin", "command_line"}:
        record["text"] = item["text"]
    return record


def encode_prompt(tokenizer: Any, text: str, *, chat: bool, think: bool) -> list[int]:
    encoded_text = text
    if chat:
        messages = json.dumps([{"role": "user", "content": text}])
        encoded_text = tokenizer.apply_chat_template(
            messages=messages,
            add_generation_prompt=True,
        )
        if not think:
            encoded_text += "<think>\n\n</think>\n\n"
    tokens = [int(token) for token in tokenizer.encode(encoded_text)]
    if not tokens:
        raise ValueError("Tokenizer produced an empty prompt")
    return tokens


def score_prompt_quality(
    item: dict[str, Any],
    token_ids: Sequence[int],
    timeout: float,
    cache: dict[Any, Any],
    decode: Any,
) -> dict[str, Any]:
    quality = suite_utils.score_completion(
        item,
        token_ids,
        timeout,
        cache,
        decode,
    )
    if not isinstance(quality, dict):
        raise RuntimeError("Suite quality evaluator returned an invalid result")
    return quality


def validate_python_api(og: Any) -> None:
    required = (
        (og, "Model"),
        (og, "Tokenizer"),
        (og, "GeneratorParams"),
        (og, "Generator"),
        (og.GeneratorParams, "set_search_options"),
        (og.GeneratorParams, "set_speculative_options"),
        (og.GeneratorParams, "get_speculative_options"),
        (og.Generator, "get_speculative_stats"),
    )
    missing = [
        f"{getattr(owner, '__name__', 'onnxruntime_genai')}.{name}"
        for owner, name in required
        if not hasattr(owner, name)
    ]
    if missing:
        raise RuntimeError(
            "Loaded onnxruntime-genai wheel lacks required EAGLE APIs: "
            + ", ".join(missing)
        )


def normalize_stats(raw: Any) -> dict[str, Any]:
    stats = dict(raw)
    missing = sorted(REQUIRED_STATS - set(stats))
    if missing:
        raise RuntimeError(
            "Loaded onnxruntime-genai wheel has an incomplete speculative "
            "statistics contract. Missing: " + ", ".join(missing)
        )
    normalized: dict[str, Any] = {}
    for name in COUNT_STATS:
        normalized[name] = int(stats[name])
    for name in FLOAT_STATS:
        normalized[name] = float(stats[name])
    for name in BOOL_STATS:
        normalized[name] = bool(stats[name])
    return normalized


def validate_eagle_stats(stats: dict[str, Any]) -> None:
    rounds = stats["rounds"]
    if stats["draft_tokens_proposed"] != TREE_DRAFT_NODES * rounds:
        raise RuntimeError(
            "EAGLE statistics contract drift: draft_tokens_proposed is not "
            f"{TREE_DRAFT_NODES} per round"
        )
    if stats["draft_tokens_evaluated"] != TREE_DRAFT_NODES * rounds:
        raise RuntimeError(
            "EAGLE statistics contract drift: draft_tokens_evaluated is not "
            f"{TREE_DRAFT_NODES} per round"
        )
    if stats["target_verify_forward_passes"] != rounds:
        raise RuntimeError(
            "EAGLE statistics contract drift: expected one target tree "
            "verification per round"
        )
    if stats["draft_tokens_accepted"] > MAX_DRAFT_TOKENS * rounds:
        raise RuntimeError("EAGLE accepted more than eight path tokens per round")
    if stats["active_rounds"] != 0:
        raise RuntimeError("EAGLE generation ended with an active speculative round")
    if stats["formula_supported"]:
        raise RuntimeError(
            "EAGLE unexpectedly reports generic speculative formula support"
        )


def verify_speculative_options(params: Any) -> None:
    actual = dict(params.get_speculative_options())
    if "max_draft_tokens" not in actual:
        raise RuntimeError("Wheel did not expose max_draft_tokens after setting it")
    if int(actual["max_draft_tokens"]) != MAX_DRAFT_TOKENS:
        raise RuntimeError(
            "Wheel retained max_draft_tokens="
            f"{actual['max_draft_tokens']!r}, expected {MAX_DRAFT_TOKENS}"
        )


def tree_metrics(
    stats: dict[str, Any],
    decode_seconds: float | None = None,
) -> dict[str, float | None]:
    verifications = stats["target_verify_forward_passes"]
    accepted = stats["draft_tokens_accepted"]
    emitted = stats["tokens_emitted"]
    outcomes = (
        stats["full_accept_rounds"]
        + stats["partial_accept_rounds"]
        + stats["zero_accept_rounds"]
    )
    rounds = stats["rounds"]
    residual = None
    if decode_seconds is not None:
        residual = (
            decode_seconds * 1000.0
            - stats["total_draft_ms"]
            - stats["total_target_verify_ms"]
        )
    return {
        "accepted_tokens_per_verification": safe_ratio(accepted, verifications),
        "emitted_tokens_per_verification": safe_ratio(emitted, verifications),
        "selected_path_utilization": safe_ratio(
            accepted,
            MAX_DRAFT_TOKENS * verifications,
        ),
        "tree_node_yield": safe_ratio(
            accepted,
            TREE_DRAFT_NODES * verifications,
        ),
        "eagle_calls_per_emitted_token": safe_ratio(
            stats["draft_forward_passes"],
            emitted,
        ),
        "target_verifications_per_emitted_token": safe_ratio(
            verifications,
            emitted,
        ),
        "native_full_accept_proportion": safe_ratio(
            stats["full_accept_rounds"],
            outcomes,
        ),
        "native_partial_accept_proportion": safe_ratio(
            stats["partial_accept_rounds"],
            outcomes,
        ),
        "native_zero_accept_proportion": safe_ratio(
            stats["zero_accept_rounds"],
            outcomes,
        ),
        "completed_round_proportion": safe_ratio(
            stats["completed_rounds"],
            rounds,
        ),
        "interrupted_round_proportion": safe_ratio(
            stats["interrupted_rounds"],
            rounds,
        ),
        "unattributed_decode_ms": residual,
    }


def _run_command(arguments: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip()


def git_metadata(repo_root: Path, expected_commit: str | None) -> dict[str, Any]:
    commit = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    branch = _run_command(["git", "branch", "--show-current"], cwd=repo_root)
    status_text = _run_command(["git", "status", "--porcelain"], cwd=repo_root)
    if expected_commit and not commit.lower().startswith(expected_commit.lower()):
        raise RuntimeError(
            f"Git HEAD {commit} does not match --expected-commit {expected_commit}"
        )
    return {
        "repo_root": str(repo_root),
        "branch": branch,
        "commit": commit,
        "dirty": bool(status_text),
        "status": status_text.splitlines(),
        "expected_commit": expected_commit,
    }


def _installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_runtime(
    expected_ort_version: str | None,
    expected_genai_version: str | None,
) -> tuple[Any, dict[str, Any]]:
    genai_distributions = {
        name: _installed_version(name)
        for name in (
            "onnxruntime-genai-cuda",
            "onnxruntime-genai",
            "onnxruntime-genai-directml",
            "onnxruntime-genai-winml",
        )
    }
    installed_genai = {
        name: version
        for name, version in genai_distributions.items()
        if version is not None
    }
    if set(installed_genai) != {"onnxruntime-genai-cuda"}:
        raise RuntimeError(
            "CUDA EAGLE benchmarking requires exactly the "
            "onnxruntime-genai-cuda distribution; found "
            f"{installed_genai or 'none'}"
        )

    runtime_distributions = {
        name: _installed_version(name)
        for name in (
            "onnxruntime-gpu",
            "onnxruntime",
            "onnxruntime-directml",
        )
    }
    installed_runtime = {
        name: version
        for name, version in runtime_distributions.items()
        if version is not None
    }
    if "onnxruntime-gpu" not in installed_runtime:
        raise RuntimeError("onnxruntime-gpu is not installed")
    conflicting = set(installed_runtime) - {"onnxruntime-gpu"}
    if conflicting:
        raise RuntimeError(
            "Conflicting ONNX Runtime distributions are installed alongside "
            "onnxruntime-gpu: " + ", ".join(sorted(conflicting))
        )

    if (
        expected_ort_version
        and installed_runtime["onnxruntime-gpu"] != expected_ort_version
    ):
        raise RuntimeError(
            "onnxruntime-gpu version "
            f"{installed_runtime['onnxruntime-gpu']} does not match "
            f"--expected-ort-version {expected_ort_version}"
        )
    if (
        expected_genai_version
        and installed_genai["onnxruntime-genai-cuda"] != expected_genai_version
    ):
        raise RuntimeError(
            "onnxruntime-genai-cuda version "
            f"{installed_genai['onnxruntime-genai-cuda']} does not match "
            f"--expected-genai-version {expected_genai_version}"
        )

    og = importlib.import_module("onnxruntime_genai")
    ort = importlib.import_module("onnxruntime")
    validate_python_api(og)
    available_providers = list(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(
            "onnxruntime-gpu does not expose CUDAExecutionProvider; available "
            f"providers: {available_providers}"
        )
    module_id = getattr(og, "__id__", None)
    if module_id not in (None, "onnxruntime-genai-cuda"):
        raise RuntimeError(
            f"Loaded onnxruntime_genai reports unexpected package ID {module_id!r}"
        )
    og_file = getattr(og, "__file__", None)
    ort_file = getattr(ort, "__file__", None)
    if not isinstance(og_file, str) or not isinstance(ort_file, str):
        raise RuntimeError("Loaded runtime modules do not expose filesystem paths")
    genai_root = Path(
        str(
            importlib.metadata.distribution(
                "onnxruntime-genai-cuda"
            ).locate_file("")
        )
    ).resolve()
    ort_root = Path(
        str(importlib.metadata.distribution("onnxruntime-gpu").locate_file(""))
    ).resolve()
    try:
        Path(og_file).resolve().relative_to(genai_root)
    except ValueError as error:
        raise RuntimeError(
            "onnxruntime_genai was not imported from the installed CUDA wheel: "
            f"{Path(og_file).resolve()}"
        ) from error
    try:
        Path(ort_file).resolve().relative_to(ort_root)
    except ValueError as error:
        raise RuntimeError(
            "onnxruntime was not imported from the installed GPU distribution: "
            f"{Path(ort_file).resolve()}"
        ) from error
    return og, {
        "onnxruntime_genai_distribution": installed_genai,
        "onnxruntime_distribution": installed_runtime,
        "onnxruntime_genai_module": str(Path(og_file).resolve()),
        "onnxruntime_module": str(Path(ort_file).resolve()),
        "onnxruntime_genai_distribution_root": str(genai_root),
        "onnxruntime_distribution_root": str(ort_root),
        "onnxruntime_genai_module_id": module_id,
        "onnxruntime_available_providers": available_providers,
    }


class NvidiaSmi:
    def __init__(self) -> None:
        summary = _run_command(["nvidia-smi"])
        self.cuda_compatibility = None
        marker = "CUDA Version:"
        if marker in summary:
            self.cuda_compatibility = summary.split(marker, 1)[1].split()[0]
        self.gpus = self._query_gpus()
        if not self.gpus:
            raise RuntimeError("nvidia-smi reported no GPUs")

    @staticmethod
    def _csv_rows(text: str) -> list[list[str]]:
        return [
            [column.strip() for column in row]
            for row in csv.reader(text.splitlines())
            if row
        ]

    def _query_gpus(self) -> list[dict[str, Any]]:
        text = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        gpus = []
        for row in self._csv_rows(text):
            if len(row) != 5:
                raise RuntimeError(f"Unexpected nvidia-smi GPU row: {row!r}")
            gpus.append(
                {
                    "index": int(row[0]),
                    "name": row[1],
                    "uuid": row[2],
                    "driver_version": row[3],
                    "memory_total_mib": float(row[4]),
                }
            )
        return gpus

    @staticmethod
    def _memory_value(value: str) -> float | None:
        normalized = value.strip()
        if normalized in {"N/A", "[N/A]", "Not Supported"}:
            return None
        return float(normalized)

    def usage(self, process_id: int) -> dict[str, Any]:
        gpu_text = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        total_by_gpu: dict[str, float] = {}
        for row in self._csv_rows(gpu_text):
            if len(row) != 2:
                raise RuntimeError(f"Unexpected nvidia-smi memory row: {row!r}")
            value = self._memory_value(row[1])
            if value is None:
                raise RuntimeError("nvidia-smi did not expose total GPU memory usage")
            total_by_gpu[row[0]] = value

        apps_text = _run_command(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        process_by_gpu: dict[str, float] = {}
        process_memory_supported = True
        for row in self._csv_rows(apps_text):
            if len(row) != 3:
                raise RuntimeError(f"Unexpected nvidia-smi process row: {row!r}")
            if int(row[0]) != process_id:
                continue
            value = self._memory_value(row[2])
            if value is None:
                process_memory_supported = False
                continue
            process_by_gpu[row[1]] = process_by_gpu.get(row[1], 0.0) + value

        process_total = (
            sum(process_by_gpu.values()) if process_memory_supported else None
        )
        return {
            "total_used_mib": sum(total_by_gpu.values()),
            "process_used_mib": process_total,
            "process_memory_supported": process_memory_supported,
            "total_used_by_gpu_mib": total_by_gpu,
            "process_used_by_gpu_mib": (
                process_by_gpu if process_memory_supported else None
            ),
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "reported_cuda_compatibility": self.cuda_compatibility,
            "gpus": self.gpus,
        }


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_rss_mib() -> float:
    if platform.system() == "Windows":
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        process = get_current_process()
        result = get_process_memory_info(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not result:
            raise ctypes.WinError()
        return counters.WorkingSetSize / (1024 * 1024)
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    raise RuntimeError("Process RSS monitoring is unsupported on this platform")


class ResourceMonitor:
    def __init__(self, nvidia: NvidiaSmi, interval: float):
        self._nvidia = nvidia
        self._interval = interval
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._started_at = time.perf_counter()
        self._sample_count = 0
        self._baseline: dict[str, Any] | None = None
        self._markers: dict[str, dict[str, Any]] = {}
        self._peak_process_rss_mib = 0.0
        self._peak_total_gpu_used_mib = 0.0
        self._peak_process_gpu_used_mib: float | None = None
        self._process_gpu_memory_supported = True

    def _capture(self) -> dict[str, Any]:
        gpu = self._nvidia.usage(os.getpid())
        return {
            "timestamp_utc": utc_now(),
            "process_rss_mib": process_rss_mib(),
            **gpu,
        }

    def _record(self, sample: dict[str, Any]) -> None:
        with self._lock:
            self._sample_count += 1
            self._peak_process_rss_mib = max(
                self._peak_process_rss_mib,
                sample["process_rss_mib"],
            )
            self._peak_total_gpu_used_mib = max(
                self._peak_total_gpu_used_mib,
                sample["total_used_mib"],
            )
            process_gpu = sample["process_used_mib"]
            self._process_gpu_memory_supported = (
                self._process_gpu_memory_supported
                and sample["process_memory_supported"]
            )
            if process_gpu is not None:
                self._peak_process_gpu_used_mib = max(
                    self._peak_process_gpu_used_mib or 0.0,
                    process_gpu,
                )

    def _sample(self) -> dict[str, Any]:
        sample = self._capture()
        self._record(sample)
        return sample

    def _run(self) -> None:
        try:
            while not self._stop_event.wait(self._interval):
                self._sample()
        except BaseException as error:
            self._error = error
            self._stop_event.set()

    def start(self) -> "ResourceMonitor":
        self._baseline = self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def mark(self, name: str) -> None:
        self._markers[name] = self._sample()

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        final = self._sample()
        if self._error is not None:
            raise RuntimeError("Resource monitor failed") from self._error
        if self._baseline is None:
            raise RuntimeError("Resource monitor was not started")
        baseline_process_gpu = self._baseline["process_used_mib"]
        process_gpu_increase = (
            self._peak_process_gpu_used_mib - baseline_process_gpu
            if self._process_gpu_memory_supported
            and self._peak_process_gpu_used_mib is not None
            and baseline_process_gpu is not None
            else None
        )
        return {
            "baseline": self._baseline,
            "markers": self._markers,
            "final": final,
            "peak_process_rss_mib": self._peak_process_rss_mib,
            "peak_total_gpu_used_mib": self._peak_total_gpu_used_mib,
            "peak_process_gpu_used_mib": (
                self._peak_process_gpu_used_mib
                if self._process_gpu_memory_supported
                else None
            ),
            "process_gpu_memory_supported": self._process_gpu_memory_supported,
            "process_gpu_peak_increase_mib": process_gpu_increase,
            "sample_count": self._sample_count,
            "duration_s": time.perf_counter() - self._started_at,
        }


def run_generation(
    og: Any,
    model: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    eagle: bool,
) -> dict[str, Any]:
    import numpy as np

    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=len(prompt_ids) + max_new_tokens,
    )
    if eagle:
        params.set_speculative_options(max_draft_tokens=MAX_DRAFT_TOKENS)
        verify_speculative_options(params)
    generator = og.Generator(model, params)

    prefill_start = time.perf_counter()
    generator.append_tokens(np.asarray([prompt_ids], dtype=np.int32))
    start_length = int(generator.token_count())
    prefill_s = time.perf_counter() - prefill_start
    decode_start = time.perf_counter()
    first_decode_s: float | None = None
    while not generator.is_done():
        if first_decode_s is None:
            first_start = time.perf_counter()
            generator.generate_next_token()
            first_decode_s = time.perf_counter() - first_start
        else:
            generator.generate_next_token()
    decode_s = time.perf_counter() - decode_start

    sequence = [int(token) for token in generator.get_sequence(0)]
    tail = sequence[start_length:]
    if not tail:
        raise RuntimeError(
            "Generation emitted zero tokens; throughput and speedup are undefined"
        )
    if len(tail) > max_new_tokens:
        raise RuntimeError(
            f"Generator emitted {len(tail)} tokens for a {max_new_tokens}-token budget"
        )
    stats = None
    if eagle:
        stats = normalize_stats(generator.get_speculative_stats())
        validate_eagle_stats(stats)
    del generator, params
    return {
        "prefill_s": prefill_s,
        "first_decode_s": first_decode_s,
        "ttft_s": (
            prefill_s + first_decode_s if first_decode_s is not None else None
        ),
        "decode_s": decode_s,
        "end_to_end_s": prefill_s + decode_s,
        "generated_tokens": len(tail),
        "tail": tail,
        "stats": stats,
    }


def run_depth_telemetry(
    og: Any,
    model: Any,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    import numpy as np

    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=len(prompt_ids) + max_new_tokens,
    )
    params.set_speculative_options(max_draft_tokens=MAX_DRAFT_TOKENS)
    verify_speculative_options(params)
    generator = og.Generator(model, params)
    generator.append_tokens(np.asarray([prompt_ids], dtype=np.int32))
    start_length = int(generator.token_count())
    previous = normalize_stats(generator.get_speculative_stats())
    histogram: collections.Counter[int] = collections.Counter()

    while not generator.is_done():
        generator.generate_next_token()
        current = normalize_stats(generator.get_speculative_stats())
        round_delta = current["rounds"] - previous["rounds"]
        accepted_delta = (
            current["draft_tokens_accepted"]
            - previous["draft_tokens_accepted"]
        )
        if round_delta not in (0, 1):
            raise RuntimeError(
                f"One generate_next_token call started {round_delta} EAGLE rounds"
            )
        if round_delta == 0 and accepted_delta != 0:
            raise RuntimeError("Accepted-token statistics changed without a new round")
        if round_delta == 1:
            if not 0 <= accepted_delta <= MAX_DRAFT_TOKENS:
                raise RuntimeError(
                    f"Observed invalid selected EAGLE depth {accepted_delta}"
                )
            histogram[accepted_delta] += 1
        previous = current

    sequence = [int(token) for token in generator.get_sequence(0)]
    tail = sequence[start_length:]
    final_stats = normalize_stats(generator.get_speculative_stats())
    validate_eagle_stats(final_stats)
    if sum(histogram.values()) != final_stats["rounds"]:
        raise RuntimeError(
            "Acceptance-depth telemetry did not observe every EAGLE round"
        )
    del generator, params
    return {
        "tail": tail,
        "output_token_sha256": token_sequence_sha256(tail),
        "acceptance_depth_histogram": {
            str(depth): histogram.get(depth, 0)
            for depth in range(MAX_DRAFT_TOKENS + 1)
        },
        "selected_path_full_accept_rounds": histogram.get(MAX_DRAFT_TOKENS, 0),
        "selected_path_partial_accept_rounds": sum(
            count
            for depth, count in histogram.items()
            if 0 < depth < MAX_DRAFT_TOKENS
        ),
        "selected_path_zero_accept_rounds": histogram.get(0, 0),
        "mean_selected_depth": safe_ratio(
            sum(depth * count for depth, count in histogram.items()),
            sum(histogram.values()),
        ),
        "final_stats": final_stats,
        "tree_metrics": tree_metrics(final_stats),
    }


def _base_row(
    run_id: str,
    decoder: str,
    prompt: dict[str, Any],
    prompt_index: int,
    configured_output_budget: int,
    output_budget: int,
    repetition: int,
    prompt_tokens: int,
    result: dict[str, Any],
    expected: Sequence[int],
) -> dict[str, Any]:
    actual = result["tail"]
    comparison = compare_tokens(expected, actual)
    decode_tps = safe_ratio(result["generated_tokens"], result["decode_s"])
    e2e_tps = safe_ratio(result["generated_tokens"], result["end_to_end_s"])
    row = {
        "run_id": run_id,
        "decoder": decoder,
        "task": prompt["task"],
        "subcategory": prompt["subcategory"],
        "category": prompt["category"],
        "question_id": prompt["question_id"],
        "prompt_index": prompt_index,
        "configured_output_token_budget": configured_output_budget,
        "output_token_budget": output_budget,
        "repetition": repetition,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": result["generated_tokens"],
        "prefill_s": result["prefill_s"],
        "first_decode_s": result["first_decode_s"],
        "ttft_s": result["ttft_s"],
        "decode_s": result["decode_s"],
        "end_to_end_s": result["end_to_end_s"],
        "decode_tokens_per_s": decode_tps,
        "end_to_end_tokens_per_s": e2e_tps,
        "baseline_decode_tokens_per_s": None,
        "baseline_end_to_end_tokens_per_s": None,
        "decode_speedup": None,
        "end_to_end_speedup": None,
        "output_token_sha256": token_sequence_sha256(actual),
        "expected_output_token_sha256": token_sequence_sha256(expected),
        **comparison,
    }
    for field in (
        *QUALITY_FIELDS,
        *COUNT_STATS,
        *BOOL_STATS,
        *FLOAT_STATS,
        *TREE_METRIC_FIELDS,
    ):
        row[field] = None
    for field in CSV_COLUMNS:
        if field.startswith("phase_"):
            row[field] = None
    return row


def _fill_phase_memory(rows: list[dict[str, Any]], decoder: str, phase: dict[str, Any]) -> None:
    baseline = phase["baseline"]
    for row in rows:
        if row["decoder"] != decoder:
            continue
        row["phase_baseline_process_rss_mib"] = baseline["process_rss_mib"]
        row["phase_peak_process_rss_mib"] = phase["peak_process_rss_mib"]
        row["phase_baseline_total_gpu_used_mib"] = baseline["total_used_mib"]
        row["phase_peak_total_gpu_used_mib"] = phase["peak_total_gpu_used_mib"]
        row["phase_baseline_process_gpu_used_mib"] = baseline["process_used_mib"]
        row["phase_peak_process_gpu_used_mib"] = phase[
            "peak_process_gpu_used_mib"
        ]


def _decoder_summary(rows: Sequence[dict[str, Any]], decoder: str) -> dict[str, Any]:
    selected = [row for row in rows if row["decoder"] == decoder]
    total_tokens = sum(row["generated_tokens"] for row in selected)
    total_decode_s = sum(row["decode_s"] for row in selected)
    total_e2e_s = sum(row["end_to_end_s"] for row in selected)
    return {
        "runs": len(selected),
        "generated_tokens": total_tokens,
        "weighted_decode_tokens_per_s": safe_ratio(total_tokens, total_decode_s),
        "weighted_end_to_end_tokens_per_s": safe_ratio(total_tokens, total_e2e_s),
        "median_run_decode_tokens_per_s": (
            statistics.median(row["decode_tokens_per_s"] for row in selected)
            if selected
            else None
        ),
        "median_run_end_to_end_tokens_per_s": (
            statistics.median(
                row["end_to_end_tokens_per_s"] for row in selected
            )
            if selected
            else None
        ),
    }


def _group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    target = _decoder_summary(rows, "target")
    eagle = _decoder_summary(rows, "eagle")
    eagle_rows = [row for row in rows if row["decoder"] == "eagle"]
    prompt_groups: collections.defaultdict[
        tuple[int, int], list[dict[str, Any]]
    ] = collections.defaultdict(list)
    for row in rows:
        prompt_groups[
            (row["output_token_budget"], row["prompt_index"])
        ].append(row)

    per_prompt_speedups = []
    per_prompt_target_tps = []
    per_prompt_eagle_tps = []
    for group in prompt_groups.values():
        target_rates = [
            row["decode_tokens_per_s"]
            for row in group
            if row["decoder"] == "target"
        ]
        eagle_rates = [
            row["decode_tokens_per_s"]
            for row in group
            if row["decoder"] == "eagle"
        ]
        if target_rates:
            per_prompt_target_tps.append(statistics.median(target_rates))
        if eagle_rates:
            per_prompt_eagle_tps.append(statistics.median(eagle_rates))
        if target_rates and eagle_rates:
            speedup = safe_ratio(
                statistics.median(eagle_rates),
                statistics.median(target_rates),
            )
            if speedup is not None:
                per_prompt_speedups.append(speedup)

    aggregate_counters = {
        name: sum(int(row[name]) for row in eagle_rows)
        for name in COUNT_STATS
    }
    aggregate_timings = {
        name: sum(float(row[name]) for row in eagle_rows)
        for name in FLOAT_STATS
        if name.startswith("total_")
    }
    aggregate_stats: dict[str, Any] = {
        **aggregate_counters,
        **aggregate_timings,
    }
    aggregate_stats.setdefault("total_draft_ms", 0.0)
    aggregate_stats.setdefault("total_target_verify_ms", 0.0)
    total_eagle_decode_s = sum(row["decode_s"] for row in eagle_rows)
    return {
        "target": target,
        "eagle": eagle,
        "weighted_decode_speedup": safe_ratio(
            eagle["weighted_decode_tokens_per_s"] or 0.0,
            target["weighted_decode_tokens_per_s"] or 0.0,
        ),
        "weighted_end_to_end_speedup": safe_ratio(
            eagle["weighted_end_to_end_tokens_per_s"] or 0.0,
            target["weighted_end_to_end_tokens_per_s"] or 0.0,
        ),
        "median_per_prompt_target_decode_tokens_per_s": (
            statistics.median(per_prompt_target_tps)
            if per_prompt_target_tps
            else None
        ),
        "median_per_prompt_eagle_decode_tokens_per_s": (
            statistics.median(per_prompt_eagle_tps)
            if per_prompt_eagle_tps
            else None
        ),
        "median_per_prompt_decode_speedup": (
            statistics.median(per_prompt_speedups)
            if per_prompt_speedups
            else None
        ),
        "geometric_mean_per_prompt_decode_speedup": geometric_mean(
            per_prompt_speedups
        ),
        "correct_eagle_runs": sum(
            bool(row["exact_match"]) for row in eagle_rows
        ),
        "eagle_correctness_rate": safe_ratio(
            sum(bool(row["exact_match"]) for row in eagle_rows),
            len(eagle_rows),
        ),
        "aggregate_eagle_counters": aggregate_counters,
        "aggregate_eagle_native_timings_ms": aggregate_timings,
        "aggregate_eagle_tree_metrics": tree_metrics(
            aggregate_stats,
            total_eagle_decode_s,
        ),
        "aggregate_native_tree_node_acceptance_rate": safe_ratio(
            aggregate_counters["draft_tokens_accepted"],
            aggregate_counters["draft_tokens_evaluated"],
        ),
    }


def summarize_quality(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    first_repetitions = [
        row
        for row in rows
        if row["repetition"] == 0 and row["quality_metric"]
    ]
    summaries = {}
    keys = sorted(
        {
            (row["task"], row["quality_metric"])
            for row in first_repetitions
        }
    )
    for task, metric in keys:
        task_rows = [
            row
            for row in first_repetitions
            if row["task"] == task and row["quality_metric"] == metric
        ]
        target_scores = [
            float(row["quality_score"])
            for row in task_rows
            if row["decoder"] == "target"
        ]
        eagle_scores = [
            float(row["quality_score"])
            for row in task_rows
            if row["decoder"] == "eagle"
        ]
        target_score = (
            sum(target_scores) / len(target_scores)
            if target_scores
            else None
        )
        eagle_score = (
            sum(eagle_scores) / len(eagle_scores)
            if eagle_scores
            else None
        )
        transitions = collections.Counter(
            row["quality_transition"]
            for row in task_rows
            if row["decoder"] == "eagle"
        )
        summaries[task] = {
            "metric": metric,
            "score_type": task_rows[0]["quality_score_type"],
            "prompt_count": len(target_scores),
            "target_score": target_score,
            "eagle_score": eagle_score,
            "score_delta": (
                eagle_score - target_score
                if eagle_score is not None and target_score is not None
                else None
            ),
            "transitions": dict(sorted(transitions.items())),
        }
    return summaries


def summarize_results(
    rows: Sequence[dict[str, Any]],
    telemetry: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    per_prompt = []
    keys = sorted(
        {
            (
                row["output_token_budget"],
                row["prompt_index"],
                row["task"],
                row["subcategory"],
                row["category"],
                str(row["question_id"]),
            )
            for row in rows
        }
    )
    for (
        budget,
        prompt_index,
        task,
        subcategory,
        category,
        question_id,
    ) in keys:
        group = [
            row
            for row in rows
            if row["output_token_budget"] == budget
            and row["prompt_index"] == prompt_index
        ]
        per_prompt.append(
            {
                "output_token_budget": budget,
                "prompt_index": prompt_index,
                "task": task,
                "subcategory": subcategory,
                "category": category,
                "question_id": question_id,
                **_group_summary(group),
            }
        )

    by_output_length = {
        str(budget): _group_summary(
            [row for row in rows if row["output_token_budget"] == budget]
        )
        for budget in sorted({row["output_token_budget"] for row in rows})
    }
    by_category = {
        category: _group_summary(
            [row for row in rows if row["category"] == category]
        )
        for category in sorted({row["category"] for row in rows})
    }
    by_task = {
        task: _group_summary(
            [row for row in rows if row["task"] == task]
        )
        for task in sorted({row["task"] for row in rows})
    }

    histogram = collections.Counter()
    for item in telemetry:
        histogram.update(
            {
                int(depth): int(count)
                for depth, count in item["acceptance_depth_histogram"].items()
            }
        )
    telemetry_rounds = sum(histogram.values())
    selected_full = histogram[MAX_DRAFT_TOKENS]
    selected_zero = histogram[0]
    selected_partial = telemetry_rounds - selected_full - selected_zero
    telemetry_summary = {
        "runs": len(telemetry),
        "rounds": telemetry_rounds,
        "acceptance_depth_histogram": {
            str(depth): histogram[depth]
            for depth in range(MAX_DRAFT_TOKENS + 1)
        },
        "mean_selected_depth": safe_ratio(
            sum(depth * count for depth, count in histogram.items()),
            telemetry_rounds,
        ),
        "selected_path_full_accept_proportion": safe_ratio(
            selected_full,
            telemetry_rounds,
        ),
        "selected_path_partial_accept_proportion": safe_ratio(
            selected_partial,
            telemetry_rounds,
        ),
        "selected_path_zero_accept_proportion": safe_ratio(
            selected_zero,
            telemetry_rounds,
        ),
    }
    return {
        "overall": _group_summary(rows),
        "by_output_length": by_output_length,
        "by_task": by_task,
        "by_category": by_category,
        "per_prompt": per_prompt,
        "quality": summarize_quality(rows),
        "acceptance_depth_telemetry": telemetry_summary,
    }


def _stage_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    complete = False
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                value,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            file.write("\n")
        complete = True
        return temporary_path
    finally:
        if not complete and temporary_path.exists():
            temporary_path.unlink()


def _stage_csv(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    complete = False
    try:
        with temporary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        name: "" if row.get(name) is None else row.get(name)
                        for name in CSV_COLUMNS
                    }
                )
        complete = True
        return temporary_path
    finally:
        if not complete and temporary_path.exists():
            temporary_path.unlink()


def write_checkpoint(
    document: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    json_path: Path,
    csv_path: Path,
) -> None:
    document["run"]["last_checkpoint_utc"] = utc_now()
    document["results"] = list(rows)
    document["summary"] = summarize_results(
        rows,
        document.get("acceptance_depth_telemetry", []),
    )
    staged_json = _stage_json(json_path, document)
    staged_csv = _stage_csv(csv_path, rows)
    try:
        os.replace(staged_json, json_path)
        os.replace(staged_csv, csv_path)
    finally:
        if staged_json.exists():
            staged_json.unlink()
        if staged_csv.exists():
            staged_csv.unlink()


def _phase_memory_evidence(phase: dict[str, Any]) -> dict[str, Any]:
    baseline = phase["baseline"]
    model_loaded = phase["markers"].get("model_loaded")
    process_increase = phase["process_gpu_peak_increase_mib"]
    total_increase = (
        phase["peak_total_gpu_used_mib"] - baseline["total_used_mib"]
    )
    model_load_process_increase = (
        model_loaded["process_used_mib"] - baseline["process_used_mib"]
        if model_loaded is not None
        and model_loaded["process_used_mib"] is not None
        and baseline["process_used_mib"] is not None
        else None
    )
    model_load_total_increase = (
        model_loaded["total_used_mib"] - baseline["total_used_mib"]
        if model_loaded is not None
        else None
    )
    return {
        "process_gpu_memory_available": phase["process_gpu_memory_supported"],
        "model_load_process_gpu_increase_mib": model_load_process_increase,
        "model_load_total_gpu_increase_mib": model_load_total_increase,
        "process_gpu_peak_increase_mib": process_increase,
        "total_gpu_peak_increase_mib": total_increase,
        "meaningful_gpu_memory_increase": (
            (process_increase is not None and process_increase >= 256)
            or total_increase >= 256
        ),
        "interpretation": (
            "A memory increase supports CUDA execution but does not prove that "
            "every graph node executed on CUDA."
        ),
    }


def _checkpoint_or_raise(
    document: dict[str, Any],
    rows: list[dict[str, Any]],
    json_path: Path,
    csv_path: Path,
    comparison: dict[str, Any],
    continue_on_mismatch: bool,
    message: str,
) -> None:
    if comparison["exact_match"] or continue_on_mismatch:
        return
    write_checkpoint(document, rows, json_path, csv_path)
    raise CorrectnessError(message)


def _benchmark(
    args: argparse.Namespace,
    document: dict[str, Any],
    rows: list[dict[str, Any]],
    json_path: Path,
    csv_path: Path,
    og: Any,
    nvidia: NvidiaSmi,
    target_bundle: Path,
    eagle_bundle: Path,
    eagle_config: dict[str, Any],
    prompt_items: list[dict[str, Any]],
    output_lengths: list[int],
) -> None:
    references: dict[tuple[int, int], dict[str, Any]] = {}
    baseline_rates: dict[tuple[int, int], dict[str, float]] = {}
    output_records: dict[tuple[int, int], dict[str, Any]] = {}
    warmup_reference: list[int] | None = None
    encoded_prompts: list[list[int]] = []
    quality_cache: dict[Any, Any] = {}
    execution_cases = build_execution_cases(prompt_items, output_lengths)
    document["execution_cases"] = execution_cases

    print("[phase 1/2] Loading target-only model ...", flush=True)
    target_monitor = ResourceMonitor(nvidia, args.monitor_interval).start()
    target_model = None
    target_tokenizer = None
    target_phase_error: BaseException | None = None
    target_load_start = time.perf_counter()
    target_load_s: float | None = None
    try:
        target_model = og.Model(str(target_bundle))
        target_load_s = time.perf_counter() - target_load_start
        target_monitor.mark("model_loaded")
        target_tokenizer = og.Tokenizer(target_model)
        encoded_prompts = [
            suite_utils.truncate_prompt_tokens(
                encode_prompt(
                    target_tokenizer,
                    item["text"],
                    chat=(
                        not args.raw_prompts
                        and not item.get("raw_prompt", False)
                    ),
                    think=args.think,
                ),
                item,
            )
            for item in prompt_items
        ]
        context_length = int(eagle_config["model"]["context_length"])
        for index, (item, token_ids) in enumerate(
            zip(prompt_items, encoded_prompts, strict=True)
        ):
            item_budgets = [
                case["output_token_budget"]
                for case in execution_cases
                if case["prompt_index"] == index
            ]
            largest_budget = max(item_budgets)
            if len(token_ids) + largest_budget > context_length:
                raise ValueError(
                    f"Prompt {item['question_id']} has {len(token_ids)} tokens; "
                    f"adding {largest_budget} exceeds context length {context_length}"
                )
        document["prompts"] = [
            prompt_provenance_record(item, index, token_ids)
            for index, (item, token_ids) in enumerate(
                zip(prompt_items, encoded_prompts, strict=True)
            )
        ]

        warmup_tokens = min(
            16,
            min(case["output_token_budget"] for case in execution_cases),
        )
        for warmup_index in range(args.warmups):
            result = run_generation(
                og,
                target_model,
                encoded_prompts[0],
                warmup_tokens,
                eagle=False,
            )
            if warmup_reference is None:
                warmup_reference = list(result["tail"])
            comparison = compare_tokens(warmup_reference, result["tail"])
            document["warmups"].append(
                {
                    "decoder": "target",
                    "warmup_index": warmup_index,
                    "output_token_budget": warmup_tokens,
                    "output_token_sha256": token_sequence_sha256(result["tail"]),
                    **comparison,
                }
            )
            if not comparison["exact_match"]:
                raise CorrectnessError("Target-only warmups were not deterministic")
        target_monitor.mark("warmup_complete")

        for case in execution_cases:
            prompt_index = case["prompt_index"]
            configured_budget = case["configured_output_token_budget"]
            output_budget = case["output_token_budget"]
            item = prompt_items[prompt_index]
            prompt_ids = encoded_prompts[prompt_index]
            reference: list[int] | None = None
            baseline_quality: dict[str, Any] | None = None
            prompt_rows: list[dict[str, Any]] = []
            output_record = {
                "configured_output_token_budget": configured_budget,
                "output_token_budget": output_budget,
                "prompt_index": prompt_index,
                "task": item["task"],
                "subcategory": item["subcategory"],
                "category": item["category"],
                "question_id": item["question_id"],
                "target_repetitions": [],
                "eagle_repetitions": [],
                "depth_telemetry": None,
            }
            document["outputs"].append(output_record)
            output_records[(output_budget, prompt_index)] = output_record

            for repetition in range(args.repetitions):
                result = run_generation(
                    og,
                    target_model,
                    prompt_ids,
                    output_budget,
                    eagle=False,
                )
                if reference is None:
                    reference = list(result["tail"])
                    output_record["expected_token_ids"] = reference
                    output_record["expected_text"] = target_tokenizer.decode(reference)
                    output_record["expected_output_token_sha256"] = (
                        token_sequence_sha256(reference)
                    )
                comparison = compare_tokens(reference, result["tail"])
                quality = score_prompt_quality(
                    item,
                    result["tail"],
                    args.code_execution_timeout,
                    quality_cache,
                    target_tokenizer.decode,
                )
                if baseline_quality is None:
                    baseline_quality = quality
                    output_record["target_quality"] = quality
                row = _base_row(
                    document["run"]["run_id"],
                    "target",
                    item,
                    prompt_index,
                    configured_budget,
                    output_budget,
                    repetition,
                    len(prompt_ids),
                    result,
                    reference,
                )
                row.update(quality)
                row["baseline_quality_score"] = baseline_quality[
                    "quality_score"
                ]
                row["quality_score_delta"] = (
                    float(quality["quality_score"])
                    - float(baseline_quality["quality_score"])
                    if quality["quality_metric"]
                    else None
                )
                row["quality_transition"] = ""
                rows.append(row)
                prompt_rows.append(row)
                repetition_record = {
                    "repetition": repetition,
                    "output_token_sha256": row["output_token_sha256"],
                    "quality": quality,
                    **comparison,
                }
                if not comparison["exact_match"]:
                    repetition_record["actual_token_ids"] = result["tail"]
                output_record["target_repetitions"].append(repetition_record)
                _checkpoint_or_raise(
                    document,
                    rows,
                    json_path,
                    csv_path,
                    comparison,
                    args.continue_on_mismatch,
                    "Target-only measured repetitions were not deterministic",
                )

            if reference is None or baseline_quality is None:
                raise RuntimeError("No target-only repetitions were run")
            references[(output_budget, prompt_index)] = {
                "tail": reference,
                "hash": token_sequence_sha256(reference),
                "quality": baseline_quality,
            }
            baseline_rates[(output_budget, prompt_index)] = {
                "decode": statistics.median(
                    row["decode_tokens_per_s"] for row in prompt_rows
                ),
                "end_to_end": statistics.median(
                    row["end_to_end_tokens_per_s"] for row in prompt_rows
                ),
            }
            quality_text = suite_utils.format_quality_with_reference(
                baseline_quality
            )
            print(
                f"  [target] budget={output_budget} "
                f"{item['task']}/{item['question_id']}: "
                f"{baseline_rates[(output_budget, prompt_index)]['decode']:.2f} "
                f"decode tok/s, quality={quality_text}",
                flush=True,
            )
            write_checkpoint(document, rows, json_path, csv_path)
    except BaseException as error:
        target_phase_error = error
    finally:
        target_tokenizer = None
        target_model = None
        gc.collect()
        target_phase = target_monitor.stop()
        target_phase["model_load_s"] = time.perf_counter() - target_load_start
        if target_load_s is not None:
            target_phase["model_load_s"] = target_load_s
        target_phase["cuda_memory_evidence"] = _phase_memory_evidence(target_phase)
        document["resources"]["target"] = target_phase
        _fill_phase_memory(rows, "target", target_phase)
        write_checkpoint(document, rows, json_path, csv_path)
    if target_phase_error is not None:
        raise target_phase_error

    print("[phase 2/2] Loading EAGLE model ...", flush=True)
    eagle_monitor = ResourceMonitor(nvidia, args.monitor_interval).start()
    eagle_model = None
    eagle_tokenizer = None
    eagle_phase_error: BaseException | None = None
    eagle_load_start = time.perf_counter()
    eagle_load_s: float | None = None
    try:
        eagle_model = og.Model(str(eagle_bundle))
        eagle_load_s = time.perf_counter() - eagle_load_start
        eagle_monitor.mark("model_loaded")
        eagle_tokenizer = og.Tokenizer(eagle_model)

        warmup_tokens = min(
            16,
            min(case["output_token_budget"] for case in execution_cases),
        )
        for warmup_index in range(args.warmups):
            result = run_generation(
                og,
                eagle_model,
                encoded_prompts[0],
                warmup_tokens,
                eagle=True,
            )
            if warmup_reference is None:
                raise RuntimeError("Target warmup reference is unavailable")
            comparison = compare_tokens(warmup_reference, result["tail"])
            document["warmups"].append(
                {
                    "decoder": "eagle",
                    "warmup_index": warmup_index,
                    "output_token_budget": warmup_tokens,
                    "output_token_sha256": token_sequence_sha256(result["tail"]),
                    **comparison,
                }
            )
            if not comparison["exact_match"] and not args.continue_on_mismatch:
                write_checkpoint(document, rows, json_path, csv_path)
                raise CorrectnessError(
                    "EAGLE warmup output differs from target-only greedy output"
                )
        eagle_monitor.mark("warmup_complete")

        for case in execution_cases:
            prompt_index = case["prompt_index"]
            configured_budget = case["configured_output_token_budget"]
            output_budget = case["output_token_budget"]
            item = prompt_items[prompt_index]
            prompt_ids = encoded_prompts[prompt_index]
            reference_record = references[(output_budget, prompt_index)]
            expected_tail: list[int] = list(reference_record["tail"])
            baseline_quality = reference_record["quality"]
            if not isinstance(baseline_quality, dict):
                raise RuntimeError("Target quality result is unavailable")
            baseline = baseline_rates[(output_budget, prompt_index)]
            output_record = output_records[(output_budget, prompt_index)]
            prompt_rows = []
            for repetition in range(args.repetitions):
                result = run_generation(
                    og,
                    eagle_model,
                    prompt_ids,
                    output_budget,
                    eagle=True,
                )
                comparison = compare_tokens(expected_tail, result["tail"])
                quality = score_prompt_quality(
                    item,
                    result["tail"],
                    args.code_execution_timeout,
                    quality_cache,
                    eagle_tokenizer.decode,
                )
                quality_transition = suite_utils.classify_quality_transition(
                    baseline_quality,
                    quality,
                    "eagle",
                )
                row = _base_row(
                    document["run"]["run_id"],
                    "eagle",
                    item,
                    prompt_index,
                    configured_budget,
                    output_budget,
                    repetition,
                    len(prompt_ids),
                    result,
                    expected_tail,
                )
                row["baseline_decode_tokens_per_s"] = baseline["decode"]
                row["baseline_end_to_end_tokens_per_s"] = baseline["end_to_end"]
                row["decode_speedup"] = safe_ratio(
                    row["decode_tokens_per_s"],
                    baseline["decode"],
                )
                row["end_to_end_speedup"] = safe_ratio(
                    row["end_to_end_tokens_per_s"],
                    baseline["end_to_end"],
                )
                row.update(quality)
                row["baseline_quality_score"] = baseline_quality[
                    "quality_score"
                ]
                row["quality_score_delta"] = (
                    float(quality["quality_score"])
                    - float(baseline_quality["quality_score"])
                    if quality["quality_metric"]
                    else None
                )
                row["quality_transition"] = quality_transition
                stats = result["stats"]
                if stats is None:
                    raise RuntimeError("EAGLE generation returned no statistics")
                row.update(stats)
                row.update(tree_metrics(stats, result["decode_s"]))
                rows.append(row)
                prompt_rows.append(row)
                repetition_record = {
                    "repetition": repetition,
                    "output_token_sha256": row["output_token_sha256"],
                    "quality": quality,
                    "quality_transition": quality_transition,
                    **comparison,
                }
                if not comparison["exact_match"]:
                    repetition_record["actual_token_ids"] = result["tail"]
                output_record["eagle_repetitions"].append(repetition_record)
                _checkpoint_or_raise(
                    document,
                    rows,
                    json_path,
                    csv_path,
                    comparison,
                    args.continue_on_mismatch,
                    "EAGLE output differs from target-only greedy output",
                )

            if args.depth_telemetry:
                telemetry = run_depth_telemetry(
                    og,
                    eagle_model,
                    prompt_ids,
                    output_budget,
                )
                comparison = compare_tokens(expected_tail, telemetry["tail"])
                telemetry_record = {
                    "configured_output_token_budget": configured_budget,
                    "output_token_budget": output_budget,
                    "prompt_index": prompt_index,
                    "task": item["task"],
                    "subcategory": item["subcategory"],
                    "category": item["category"],
                    "question_id": item["question_id"],
                    **{
                        key: value
                        for key, value in telemetry.items()
                        if key != "tail"
                    },
                    **comparison,
                }
                if not comparison["exact_match"]:
                    telemetry_record["actual_token_ids"] = telemetry["tail"]
                document["acceptance_depth_telemetry"].append(telemetry_record)
                output_record["depth_telemetry"] = telemetry_record
                _checkpoint_or_raise(
                    document,
                    rows,
                    json_path,
                    csv_path,
                    comparison,
                    args.continue_on_mismatch,
                    "Untimed EAGLE telemetry output differs from target-only output",
                )

            median_tps = statistics.median(
                row["decode_tokens_per_s"] for row in prompt_rows
            )
            median_speedup = statistics.median(
                row["decode_speedup"] for row in prompt_rows
            )
            representative = prompt_rows[0]
            quality_text = suite_utils.format_quality_with_reference(
                representative
            )
            print(
                f"  [eagle]  budget={output_budget} "
                f"{item['task']}/{item['question_id']}: "
                f"{median_tps:.2f} decode tok/s, "
                f"{median_speedup:.3f}x, quality={quality_text}, "
                f"transition={representative['quality_transition'] or 'not_scored'}",
                flush=True,
            )
            write_checkpoint(document, rows, json_path, csv_path)
    except BaseException as error:
        eagle_phase_error = error
    finally:
        eagle_tokenizer = None
        eagle_model = None
        gc.collect()
        eagle_phase = eagle_monitor.stop()
        eagle_phase["model_load_s"] = time.perf_counter() - eagle_load_start
        if eagle_load_s is not None:
            eagle_phase["model_load_s"] = eagle_load_s
        eagle_phase["cuda_memory_evidence"] = _phase_memory_evidence(eagle_phase)
        document["resources"]["eagle"] = eagle_phase
        _fill_phase_memory(rows, "eagle", eagle_phase)
        write_checkpoint(document, rows, json_path, csv_path)
    if eagle_phase_error is not None:
        raise eagle_phase_error


def _manifest_verification(
    manifest_path: Path,
    bundle: Path,
    config: dict[str, Any],
    artifact_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    bundle_root = bundle.resolve()
    manifest = _load_json(manifest_path)
    expected_tree = {
        "total_tokens": TREE_TOTAL_TOKENS,
        "selected_draft_nodes": TREE_DRAFT_NODES,
        "depth": TREE_DEPTH,
        "top_k": TREE_TOP_K,
        "scored_candidates": TREE_SCORED_CANDIDATES,
        "target_verifications_per_round": 1,
        "eagle_calls_per_round": EAGLE_CALLS_PER_FULL_ROUND,
    }
    manifest_tree = manifest.get("tree")
    if not isinstance(manifest_tree, dict):
        raise ValueError("Manifest is missing tree")
    mismatches = [
        f"{name}={manifest_tree.get(name)!r} (expected {expected!r})"
        for name, expected in expected_tree.items()
        if manifest_tree.get(name) != expected
    ]
    if mismatches:
        raise ValueError("Manifest tree contract mismatch: " + "; ".join(mismatches))

    expected_runtime = {
        "batch_size": 1,
        "num_beams": 1,
        "greedy_only": True,
        "max_draft_tokens": MAX_DRAFT_TOKENS,
        "past_present_share_buffer": False,
        "graph_capture": False,
    }
    manifest_runtime = manifest.get("runtime_contract")
    if not isinstance(manifest_runtime, dict):
        raise ValueError("Manifest is missing runtime_contract")
    runtime_mismatches = [
        f"{name}={manifest_runtime.get(name)!r} (expected {expected!r})"
        for name, expected in expected_runtime.items()
        if manifest_runtime.get(name) != expected
    ]
    if runtime_mismatches:
        raise ValueError(
            "Manifest runtime contract mismatch: "
            + "; ".join(runtime_mismatches)
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Manifest is missing artifacts")
    decoder_files = decoder_artifacts(bundle_root, config)
    eagle_file = eagle_graph_path(bundle_root, config)
    mapping = {
        "bf16_target_onnx_sha256": artifact_records[
            str(decoder_files[0].relative_to(bundle_root))
        ]["sha256"],
        "bf16_eagle_sha256": artifact_records[
            str(eagle_file.relative_to(bundle_root))
        ]["sha256"],
        "bf16_runtime_config_sha256": artifact_records[CONFIG_FILENAME]["sha256"],
    }
    external_data = [
        path for path in decoder_files[1:] if path.name.endswith(".data")
    ]
    if external_data:
        mapping["bf16_target_data_sha256"] = artifact_records[
            str(external_data[0].relative_to(bundle_root))
        ]["sha256"]
    hash_results = {}
    for key, actual in mapping.items():
        expected = artifacts.get(key)
        if not isinstance(expected, str):
            raise ValueError(f"Manifest is missing {key}")
        matches = actual.lower() == expected.lower()
        hash_results[key] = {
            "expected": expected,
            "actual": actual,
            "match": matches,
        }
        if not matches:
            raise ValueError(f"Manifest hash mismatch for {key}")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "tree_contract_match": True,
        "runtime_contract_match": True,
        "artifact_hashes": hash_results,
    }


def _collect_artifacts(
    bundle: Path,
    config: dict[str, Any],
    script_path: Path,
    dataset: Path | None,
) -> dict[str, Any]:
    bundle_root = bundle.resolve()
    paths = [
        bundle_root / CONFIG_FILENAME,
        *decoder_artifacts(bundle_root, config),
        eagle_graph_path(bundle_root, config),
    ]
    records: dict[str, Any] = {}
    for path in paths:
        relative = str(path.relative_to(bundle_root))
        print(f"Hashing {relative} ({path.stat().st_size / 1024**3:.3f} GiB) ...")
        records[relative] = artifact_record(path, bundle_root)
    records["benchmark_script"] = artifact_record(script_path)
    if dataset is not None:
        records["prompt_dataset"] = artifact_record(dataset)
    return records


def _prepare_output_prefix(value: Path | None, script_path: Path) -> Path:
    if value is not None:
        prefix = value.resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = script_path.parent / "results" / f"eagle_{timestamp}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    return prefix


def _output_path(prefix: Path, extension: str) -> Path:
    return Path(f"{prefix}{extension}")


def _run_benchmark_cli(args: argparse.Namespace, script_path: Path) -> None:
    if args.eagle_model is None:
        raise ValueError("--eagle-model is required unless --self-test is used")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.limit_per_task < 0:
        raise ValueError("--limit-per-task must be non-negative")
    if args.limit_per_category is not None and args.limit_per_category < 0:
        raise ValueError("--limit-per-category must be non-negative")
    if args.max_prompts < 0:
        raise ValueError("--max-prompts must be non-negative")
    if args.monitor_interval <= 0:
        raise ValueError("--monitor-interval must be positive")
    if args.code_execution_timeout <= 0:
        raise ValueError("--code-execution-timeout must be positive")
    count_options = {
        "--gsm8k-problems": args.gsm8k_problems,
        "--ifeval-problems": args.ifeval_problems,
        "--math500-problems": args.math500_problems,
        "--longbench-problems-per-task": args.longbench_problems_per_task,
        "--humaneval-problems": args.humaneval_problems,
        "--humanevalplus-problems": args.humanevalplus_problems,
        "--mbppplus-problems": args.mbppplus_problems,
        "--livecodebench-problems": args.livecodebench_problems,
    }
    for option, value in count_options.items():
        if value < 0:
            raise ValueError(f"{option} must be non-negative")
    if args.longbench_max_input_tokens < 512:
        raise ValueError("--longbench-max-input-tokens must be at least 512")
    if args.builtin and args.categories:
        raise ValueError("--categories cannot be combined with --builtin")
    if args.prompt and args.categories:
        raise ValueError("--categories cannot be combined with --prompt")
    if args.by_category and (args.builtin or args.prompt):
        raise ValueError("--by-category requires the Spec-Bench dataset")
    if args.limit_per_category is not None and (args.builtin or args.prompt):
        raise ValueError("--limit-per-category requires the Spec-Bench dataset")

    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be positive")
        if args.output_lengths != "32":
            raise ValueError(
                "--max-new-tokens cannot be combined with --output-lengths"
            )
        output_lengths = [args.max_new_tokens]
    else:
        output_lengths = parse_output_lengths(args.output_lengths)
    suites = parse_suite_names(args.suites)
    args.longbench_tasks = parse_longbench_tasks(args.longbench_tasks)
    executable_suites = executable_suite_selection(
        suites,
        args.allow_code_execution,
    )
    if executable_suites:
        print(
            "WARNING: evaluator guards are not security sandboxes. "
            "Run generated code only on an isolated, disposable machine.",
            file=sys.stderr,
        )
    if "mtbench" not in suites and (
        args.builtin
        or args.prompt
        or args.tasks
        or args.categories
        or args.by_category
        or args.limit_per_category is not None
    ):
        raise ValueError(
            "Built-in/custom/Spec-Bench selection options require "
            "'mtbench' in --suites"
        )

    tasks = parse_categories(args.tasks)
    categories = parse_categories(args.categories)
    eagle_bundle = args.eagle_model.resolve()
    if not eagle_bundle.is_dir():
        raise FileNotFoundError(f"EAGLE model bundle does not exist: {eagle_bundle}")
    config_path = eagle_bundle / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"EAGLE configuration does not exist: {config_path}")
    eagle_config = _load_json(config_path)
    runtime_contract = validate_eagle_config(eagle_config)

    dataset = args.dataset.resolve()
    prompt_items = load_selected_prompts(
        args,
        suites,
        dataset,
        tasks,
        categories,
    )
    dataset_for_provenance = (
        dataset
        if "mtbench" in suites and not args.builtin and not args.prompt
        else None
    )
    task_counts = collections.Counter(item["task"] for item in prompt_items)

    prefix = _prepare_output_prefix(args.output, script_path)
    json_path = _output_path(prefix, ".json")
    csv_path = _output_path(prefix, ".csv")
    log_path = _output_path(prefix, ".log")
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    )
    document: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "status": "initializing",
            "started_utc": utc_now(),
            "completed_utc": None,
            "last_checkpoint_utc": None,
            "command_line": [sys.executable, *sys.argv],
            "artifact_paths": {
                "json": str(json_path),
                "csv": str(csv_path),
                "log": str(log_path),
            },
        },
        "configuration": {
            "eagle_model": str(eagle_bundle),
            "target_model_argument": (
                str(args.target_model.resolve()) if args.target_model else None
            ),
            "output_lengths": output_lengths,
            "suites": suites,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "raw_prompts": args.raw_prompts,
            "think": args.think,
            "depth_telemetry": args.depth_telemetry,
            "continue_on_mismatch": args.continue_on_mismatch,
            "limit_per_task": args.limit_per_task,
            "limit_per_category": args.limit_per_category,
            "by_category": (
                args.by_category
                or args.limit_per_category is not None
                or categories is not None
            ),
            "tasks": tasks,
            "categories": categories,
            "task_counts": dict(task_counts),
            "quality_evaluation": {
                "code_execution_timeout_s": args.code_execution_timeout,
                "allow_code_execution": args.allow_code_execution,
                "executable_suites": sorted(executable_suites),
                "task_specific_output_budgets": True,
                "task_specific_input_truncation": True,
                "gsm8k_path": str(args.gsm8k_path),
                "gsm8k_problems": args.gsm8k_problems,
                "ifeval_path": str(args.ifeval_path),
                "ifeval_problems": args.ifeval_problems,
                "math500_path": str(args.math500_path),
                "math500_problems": args.math500_problems,
                "longbench_tasks": args.longbench_tasks,
                "longbench_problems_per_task": (
                    args.longbench_problems_per_task
                ),
                "longbench_max_input_tokens": (
                    args.longbench_max_input_tokens
                ),
                "humaneval_problems": args.humaneval_problems,
                "humanevalplus_problems": args.humanevalplus_problems,
                "mbppplus_problems": args.mbppplus_problems,
                "livecodebench_problems": args.livecodebench_problems,
                "livecodebench_release": args.livecodebench_release,
            },
            "runtime_contract": runtime_contract,
            "eagle_genai_config": eagle_config,
            "target_genai_config": derive_target_config(eagle_config),
            "metric_semantics": {
                "authoritative_speedup": "measured wall-clock throughput",
                "generic_formula_authoritative": False,
                "native_acceptance_rate": (
                    "accepted selected-path tokens divided by 59 evaluated "
                    "tree nodes per verification"
                ),
                "selected_path_utilization": (
                    "accepted selected-path tokens divided by the eight-token "
                    "path capacity per verification"
                ),
                "tree_node_yield": (
                    "accepted selected-path tokens divided by 59 evaluated "
                    "tree nodes per verification"
                ),
                "unattributed_decode_ms": (
                    "wall-clock decode time minus native draft and target-tree "
                    "verification timers; it is not a precise component timer"
                ),
            },
        },
        "provenance": {
            "timestamp_utc": utc_now(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "operating_system": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
            "cuda_environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_PATH",
                    "CUDA_HOME",
                    "CUDA_MODULE_LOADING",
                    "CUDA_VISIBLE_DEVICES",
                    "ORT_CUDA_UNAVAILABLE",
                )
            },
        },
        "target_bundle": None,
        "artifacts": {},
        "manifest_verification": None,
        "prompts": [],
        "execution_cases": [],
        "warmups": [],
        "outputs": [],
        "acceptance_depth_telemetry": [],
        "resources": {},
        "results": [],
        "summary": {},
        "errors": [],
    }
    rows: list[dict[str, Any]] = []
    staged_target = False
    target_bundle: Path | None = None

    try:
        document["run"]["status"] = "preflight"
        repo_root = script_path.parents[2]
        document["provenance"]["git"] = git_metadata(
            repo_root,
            args.expected_commit,
        )
        og, packages = import_runtime(
            args.expected_ort_version,
            args.expected_genai_version,
        )
        document["provenance"]["packages"] = packages
        nvidia = NvidiaSmi()
        document["provenance"]["nvidia_smi"] = nvidia.metadata()

        if args.target_model is not None:
            supplied_target: Path = args.target_model
            resolved_target = supplied_target.resolve()
            if not resolved_target.is_dir():
                raise FileNotFoundError(
                    f"Target-only model bundle does not exist: {resolved_target}"
                )
            target_info = validate_target_bundle(
                eagle_bundle,
                resolved_target,
                eagle_config,
            )
            target_bundle = resolved_target
        else:
            target_bundle, target_info = stage_target_bundle(
                eagle_bundle,
                eagle_config,
                args.staging_root,
            )
            staged_target = True
        document["target_bundle"] = target_info

        document["artifacts"] = _collect_artifacts(
            eagle_bundle,
            eagle_config,
            script_path,
            dataset_for_provenance,
        )
        if args.manifest:
            manifest_path = args.manifest.resolve()
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
            document["manifest_verification"] = _manifest_verification(
                manifest_path,
                eagle_bundle,
                eagle_config,
                document["artifacts"],
            )

        print(f"onnxruntime_genai: {packages['onnxruntime_genai_module']}")
        print(f"EAGLE model:       {eagle_bundle}")
        print(f"Target-only model: {target_bundle}")
        print(
            f"Prompts={len(prompt_items)}, tasks={dict(task_counts)}, "
            f"suites={suites}, output_lengths={output_lengths}, "
            f"warmups={args.warmups}, repetitions={args.repetitions}"
        )
        print(f"JSON: {json_path}")
        print(f"CSV:  {csv_path}")
        print(f"Log:  {log_path}")
        benchmark_target = target_bundle
        document["run"]["status"] = "running"
        write_checkpoint(document, rows, json_path, csv_path)
        _benchmark(
            args,
            document,
            rows,
            json_path,
            csv_path,
            og,
            nvidia,
            benchmark_target,
            eagle_bundle,
            eagle_config,
            prompt_items,
            output_lengths,
        )
        document["run"]["status"] = "complete"
        document["run"]["completed_utc"] = utc_now()
    except BaseException as error:
        document["run"]["status"] = "failed"
        document["run"]["completed_utc"] = utc_now()
        document["errors"].append(
            {
                "timestamp_utc": utc_now(),
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        traceback.print_exc()
        raise
    finally:
        if staged_target and target_bundle is not None and not args.keep_staged_target:
            shutil.rmtree(target_bundle)
            if isinstance(document["target_bundle"], dict):
                document["target_bundle"]["cleaned"] = True
        write_checkpoint(document, rows, json_path, csv_path)

    overall = document["summary"]["overall"]
    print("\nBenchmark complete")
    print(
        "  weighted decode: "
        f"target={overall['target']['weighted_decode_tokens_per_s']:.2f} tok/s, "
        f"EAGLE={overall['eagle']['weighted_decode_tokens_per_s']:.2f} tok/s, "
        f"speedup={overall['weighted_decode_speedup']:.3f}x"
    )
    print(
        "  exact EAGLE runs: "
        f"{overall['correct_eagle_runs']}/{overall['eagle']['runs']}"
    )
    for task, quality in document["summary"]["quality"].items():
        print(
            f"  quality {task} ({quality['metric']}): "
            f"target={quality['target_score']:.1%}, "
            f"EAGLE={quality['eagle_score']:.1%}, "
            f"delta={quality['score_delta']:+.1%}"
        )


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eagle-model",
        type=Path,
        help="complete EAGLE model bundle containing genai_config.json",
    )
    parser.add_argument(
        "--target-model",
        type=Path,
        help=(
            "optional prebuilt target-only bundle; omitted creates a temporary "
            "hard-linked bundle"
        ),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        help="directory for the temporary target bundle (default: EAGLE bundle parent)",
    )
    parser.add_argument(
        "--keep-staged-target",
        action="store_true",
        help="do not remove an automatically staged target-only bundle",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="optional Phase 5 manifest whose BF16 hashes and tree contract must match",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--builtin",
        action="store_true",
        help="use the four fixed Phase 5 smoke-test prompts",
    )
    source.add_argument(
        "--prompt",
        action="append",
        help="custom prompt; repeat for multiple prompts",
    )
    parser.add_argument(
        "--suites",
        default="mtbench",
        help=(
            "comma-separated suites: mtbench,gsm8k,ifeval,math500,longbench,"
            "humaneval,humanevalplus,mbppplus,livecodebench"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=here / "question.jsonl",
        help="Spec-Bench question.jsonl (first turn of each record is used)",
    )
    parser.add_argument(
        "--tasks",
        help=(
            "comma-separated Spec-Bench tasks to include: "
            "mt_bench,translation,summarization,qa,math_reasoning,rag"
        ),
    )
    parser.add_argument(
        "--categories",
        help=(
            "comma-separated raw Spec-Bench categories; implies category-level "
            "filtering and is retained for EAGLE benchmark compatibility"
        ),
    )
    parser.add_argument(
        "--by-category",
        "--mt-bench-by-subcategory",
        dest="by_category",
        action="store_true",
        help="report all 13 raw Spec-Bench categories instead of collapsing MT-Bench",
    )
    parser.add_argument(
        "--limit-per-task",
        type=int,
        default=0,
        help="prompt cap per Spec-Bench task; 0 uses all",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=None,
        help=(
            "prompt cap per raw category; implies --by-category; "
            "0 uses the full corpus"
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="cap built-in or custom prompts; 0 uses all",
    )
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--output-lengths",
        default="32",
        help=(
            "comma-separated generated-token budgets, e.g. 32,128,512; "
            "quality suites use their pinned task budgets"
        ),
    )
    budget.add_argument(
        "--max-new-tokens",
        type=int,
        help="single-budget alias for --output-lengths",
    )
    parser.add_argument(
        "--gsm8k-path",
        type=Path,
        default=here / ".cache" / "gsm8k_test.jsonl",
        help="GSM8K test JSONL; downloaded from the pinned source if absent",
    )
    parser.add_argument(
        "--gsm8k-problems",
        type=int,
        default=200,
        help="number of GSM8K problems; 0 uses all 1,319",
    )
    parser.add_argument(
        "--ifeval-path",
        type=Path,
        default=here / ".cache" / "ifeval_input_data.jsonl",
        help="IFEval JSONL; downloaded from the pinned source if absent",
    )
    parser.add_argument(
        "--ifeval-problems",
        type=int,
        default=541,
        help="number of IFEval prompts; 0 uses all 541",
    )
    parser.add_argument(
        "--math500-path",
        type=Path,
        default=here / ".cache" / "math500_test.jsonl",
        help="MATH-500 JSONL; downloaded from the pinned source if absent",
    )
    parser.add_argument(
        "--math500-problems",
        type=int,
        default=500,
        help="number of MATH-500 problems; 0 uses all 500",
    )
    parser.add_argument(
        "--longbench-tasks",
        default="qasper,hotpotqa,gov_report,passage_retrieval_en",
        help="LongBench tasks: qasper,hotpotqa,gov_report,passage_retrieval_en",
    )
    parser.add_argument(
        "--longbench-problems-per-task",
        type=int,
        default=50,
        help="problems per LongBench task; 0 uses all",
    )
    parser.add_argument(
        "--longbench-max-input-tokens",
        type=int,
        default=16384,
        help="middle-truncate LongBench prompts above this token count",
    )
    parser.add_argument(
        "--humaneval-problems",
        type=int,
        default=164,
        help="number of HumanEval problems; 0 uses all 164",
    )
    parser.add_argument(
        "--humanevalplus-problems",
        type=int,
        default=164,
        help="number of HumanEval+ problems; 0 uses all 164",
    )
    parser.add_argument(
        "--mbppplus-problems",
        type=int,
        default=378,
        help="number of MBPP+ problems; 0 uses all 378",
    )
    parser.add_argument(
        "--livecodebench-problems",
        type=int,
        default=100,
        help="number of newest LiveCodeBench problems; 0 uses the full release",
    )
    parser.add_argument(
        "--livecodebench-release",
        default="release_v6",
        help="pinned LiveCodeBench release",
    )
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
        help=(
            "required for executable code suites; use only on an isolated, "
            "disposable machine"
        ),
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--raw-prompts",
        "--raw",
        action="store_true",
        help="encode raw text instead of applying the tokenizer chat template",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="do not append Qwen3's empty no-think block after the chat template",
    )
    parser.add_argument(
        "--depth-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run separate untimed per-token acceptance-depth telemetry",
    )
    parser.add_argument(
        "--continue-on-mismatch",
        action="store_true",
        help="diagnostic mode: record token mismatches instead of failing immediately",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=0.25,
        help="resource-monitor sampling interval in seconds",
    )
    parser.add_argument(
        "--expected-commit",
        help="required Git HEAD hash or unique prefix",
    )
    parser.add_argument(
        "--expected-ort-version",
        help="required installed onnxruntime-gpu version",
    )
    parser.add_argument(
        "--expected-genai-version",
        help="required installed onnxruntime-genai-cuda version",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output prefix; .json, .csv, and .log are appended",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run model-independent helper checks and exit",
    )
    return parser


def run_self_tests() -> None:
    assert parse_output_lengths("32,128,32") == [32, 128]
    assert parse_categories("writing, coding,writing") == ["writing", "coding"]
    assert parse_suite_names("mtbench,math-500,humaneval+") == [
        "mtbench",
        "math500",
        "humanevalplus",
    ]
    assert parse_longbench_tasks("qasper,hotpotqa,qasper") == [
        "qasper",
        "hotpotqa",
    ]
    try:
        executable_suite_selection(["humaneval"], False)
    except ValueError:
        pass
    else:
        raise AssertionError("Generated-code suite did not require explicit consent")
    assert executable_suite_selection(
        ["humaneval", "gsm8k"],
        True,
    ) == {"humaneval"}
    assert process_rss_mib() > 0

    eagle_config = {
        "model": {
            "decoder": {"filename": "model.onnx"},
            "eagle": {"filename": "eagle.onnx"},
        },
        "search": {},
        "speculative": {"max_draft_tokens": 8},
    }
    target = derive_target_config(eagle_config)
    assert "eagle" not in target["model"]
    assert "speculative" not in target
    assert "eagle" in eagle_config["model"]

    exact = compare_tokens([1, 2], [1, 2])
    changed = compare_tokens([1, 2, 3], [1, 9, 3])
    shorter = compare_tokens([1, 2, 3], [1, 2])
    assert exact["exact_match"] and exact["first_difference_index"] is None
    assert changed["first_difference_index"] == 1
    assert changed["expected_token_at_difference"] == 2
    assert changed["actual_token_at_difference"] == 9
    assert shorter["divergence_type"] == "actual_shorter"

    stats: dict[str, Any] = {name: 0 for name in COUNT_STATS}
    stats.update({name: 0.0 for name in FLOAT_STATS})
    stats["formula_supported"] = False
    stats.update(
        {
            "rounds": 2,
            "completed_rounds": 2,
            "draft_tokens_proposed": 118,
            "draft_tokens_evaluated": 118,
            "draft_tokens_accepted": 8,
            "tokens_emitted": 10,
            "draft_forward_passes": 16,
            "target_verify_forward_passes": 2,
            "partial_accept_rounds": 2,
        }
    )
    validate_eagle_stats(stats)
    metrics = tree_metrics(stats, 0.1)
    assert metrics["accepted_tokens_per_verification"] == 4
    assert metrics["selected_path_utilization"] == 0.5
    assert metrics["tree_node_yield"] == 8 / 118

    class FakeParams:
        def set_search_options(self, **_: Any) -> None:
            pass

        def set_speculative_options(self, **_: Any) -> None:
            pass

        def get_speculative_options(self) -> dict[str, int]:
            return {"max_draft_tokens": MAX_DRAFT_TOKENS}

    class FakeGenerator:
        def get_speculative_stats(self) -> dict[str, Any]:
            return stats

    fake_og = SimpleNamespace(
        Model=object,
        Tokenizer=object,
        GeneratorParams=FakeParams,
        Generator=FakeGenerator,
    )
    validate_python_api(fake_og)
    assert normalize_stats(stats)["rounds"] == 2
    try:
        validate_python_api(
            SimpleNamespace(
                Model=object,
                Tokenizer=object,
                GeneratorParams=FakeParams,
                Generator=object,
            )
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Missing EAGLE APIs were not rejected")
    try:
        normalize_stats({"rounds": 0})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Incomplete speculative statistics were not rejected")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        bundle = root / "eagle-bundle"
        bundle.mkdir()
        complete_config = {
            "model": {
                "type": "qwen3",
                "context_length": 128,
                "decoder": {
                    "filename": "model.onnx",
                    "session_options": {"provider_options": [{"cuda": {}}]},
                },
                "eagle": {
                    "filename": "eagle.onnx",
                    "session_options": {"provider_options": [{"cuda": {}}]},
                    "total_tokens": TREE_TOTAL_TOKENS,
                    "depth": TREE_DEPTH,
                    "top_k": TREE_TOP_K,
                },
            },
            "search": {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
                "past_present_share_buffer": False,
                "repetition_penalty": 1.0,
                "min_length": 0,
            },
            "speculative": {"max_draft_tokens": MAX_DRAFT_TOKENS},
        }
        validate_eagle_config(complete_config)
        (bundle / CONFIG_FILENAME).write_text(
            json.dumps(complete_config),
            encoding="utf-8",
        )
        (bundle / "model.onnx").write_bytes(b"target graph")
        (bundle / "model.onnx.data").write_bytes(b"target weights")
        (bundle / "eagle.onnx").write_bytes(b"eagle graph")
        (bundle / "tokenizer.json").write_text("{}", encoding="utf-8")
        staged, staging_info = stage_target_bundle(
            bundle,
            complete_config,
            root / "staging",
        )
        try:
            assert not (staged / "eagle.onnx").exists()
            assert (staged / "model.onnx.data").read_bytes() == b"target weights"
            assert _load_json(staged / CONFIG_FILENAME) == derive_target_config(
                complete_config
            )
            assert staging_info["linked_files"] + staging_info["copied_files"] == 3
            assert validate_target_bundle(
                bundle,
                staged,
                complete_config,
            )["config_match"]
        finally:
            shutil.rmtree(staged)
        try:
            stage_target_bundle(
                bundle,
                complete_config,
                bundle / "invalid-staging-root",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Staging inside the source bundle was not rejected")

        dataset = root / "question.jsonl"
        records = [
            {"question_id": "a1", "category": "a", "turns": ["A1"]},
            {"question_id": "a2", "category": "a", "turns": ["A2"]},
            {"question_id": "b1", "category": "b", "turns": ["B1"]},
            {"question_id": "b2", "category": "b", "turns": ["B2"]},
        ]
        dataset.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        prompts = load_prompt_items(
            dataset,
            builtin=False,
            custom_prompts=None,
            tasks=None,
            categories=None,
            limit_per_task=1,
            by_category=True,
            max_prompts=0,
        )
        assert [item["question_id"] for item in prompts] == ["a1", "b1"]

        gsm8k_path = root / "gsm8k.jsonl"
        gsm8k_path.write_text(
            json.dumps(
                {
                    "question": "How many?",
                    "answer": "Reasoning. #### 9",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ifeval_path = root / "ifeval.jsonl"
        ifeval_path.write_text(
            json.dumps(
                {
                    "key": 1,
                    "prompt": "Say hello.",
                    "instruction_id_list": [],
                    "kwargs": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        math500_path = root / "math500.jsonl"
        math500_path.write_text(
            json.dumps(
                {
                    "unique_id": "m1",
                    "subject": "Algebra",
                    "problem": "What is 1 + 1?",
                    "answer": "2",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        suite_args = SimpleNamespace(
            gsm8k_path=gsm8k_path,
            gsm8k_problems=1,
            ifeval_path=ifeval_path,
            ifeval_problems=1,
            math500_path=math500_path,
            math500_problems=1,
            longbench_tasks=["qasper"],
            longbench_problems_per_task=1,
            longbench_max_input_tokens=512,
            humaneval_problems=1,
            humanevalplus_problems=1,
            mbppplus_problems=1,
            livecodebench_problems=1,
            livecodebench_release="release_v6",
            limit_per_category=None,
            limit_per_task=0,
            by_category=False,
            max_prompts=0,
            builtin=False,
            prompt=None,
        )
        quality_prompts = load_selected_prompts(
            suite_args,
            ["gsm8k", "ifeval", "math500"],
            root / "unused.jsonl",
            None,
            None,
        )
        assert [item["task"] for item in quality_prompts] == [
            "gsm8k",
            "ifeval",
            "math500",
        ]
        assert all(
            item["max_output_tokens"] == 512
            for item in quality_prompts
        )

        cases = build_execution_cases(
            [
                prompts[0],
                {
                    **prompts[1],
                    "max_output_tokens": 512,
                },
            ],
            [32, 128],
        )
        assert cases == [
            {
                "prompt_index": 0,
                "configured_output_token_budget": 32,
                "output_token_budget": 32,
            },
            {
                "prompt_index": 0,
                "configured_output_token_budget": 128,
                "output_token_budget": 128,
            },
            {
                "prompt_index": 1,
                "configured_output_token_budget": 32,
                "output_token_budget": 512,
            },
        ]
        assert suite_utils.truncate_prompt_tokens(
            list(range(10)),
            {"max_input_tokens": 6},
        ) == [0, 1, 2, 7, 8, 9]
        gsm8k_item = {
            "task": "gsm8k",
            "question_id": 0,
            "quality_metric": "gsm8k_accuracy",
            "reference_answer": "9",
        }
        gsm8k_quality = score_prompt_quality(
            gsm8k_item,
            [1],
            1.0,
            {},
            lambda _: "The answer is 9. #### 9",
        )
        assert gsm8k_quality["quality_score"] is True

        target_result = {
            "prefill_s": 0.05,
            "first_decode_s": 0.01,
            "ttft_s": 0.06,
            "decode_s": 0.2,
            "end_to_end_s": 0.25,
            "generated_tokens": 2,
            "tail": [7, 8],
        }
        eagle_result = {
            **target_result,
            "decode_s": 0.1,
            "end_to_end_s": 0.15,
        }
        prompt = {
            "task": "gsm8k",
            "subcategory": "math_reasoning",
            "category": "math_reasoning",
            "question_id": "a1",
        }
        target_row = _base_row(
            "self-test",
            "target",
            prompt,
            0,
            2,
            2,
            0,
            3,
            target_result,
            [7, 8],
        )
        eagle_row = _base_row(
            "self-test",
            "eagle",
            prompt,
            0,
            2,
            2,
            0,
            3,
            eagle_result,
            [7, 8],
        )
        eagle_row.update(stats)
        eagle_row.update(tree_metrics(stats, eagle_result["decode_s"]))
        target_row.update(gsm8k_quality)
        target_row["baseline_quality_score"] = True
        target_row["quality_score_delta"] = 0.0
        target_row["quality_transition"] = ""
        eagle_row.update(gsm8k_quality)
        eagle_row["baseline_quality_score"] = True
        eagle_row["quality_score_delta"] = 0.0
        eagle_row["quality_transition"] = "both_correct"
        eagle_row["baseline_decode_tokens_per_s"] = target_row[
            "decode_tokens_per_s"
        ]
        eagle_row["baseline_end_to_end_tokens_per_s"] = target_row[
            "end_to_end_tokens_per_s"
        ]
        eagle_row["decode_speedup"] = 2.0
        eagle_row["end_to_end_speedup"] = safe_ratio(
            eagle_row["end_to_end_tokens_per_s"],
            target_row["end_to_end_tokens_per_s"],
        )
        document: dict[str, Any] = {
            "run": {"run_id": "self-test"},
            "acceptance_depth_telemetry": [
                {
                    "acceptance_depth_histogram": {
                        str(depth): 2 if depth == 4 else 0
                        for depth in range(MAX_DRAFT_TOKENS + 1)
                    }
                }
            ],
        }
        json_path = root / "result.json"
        csv_path = root / "result.csv"
        write_checkpoint(
            document,
            [target_row, eagle_row],
            json_path,
            csv_path,
        )
        checkpoint = json.loads(json_path.read_text(encoding="utf-8"))
        assert len(checkpoint["results"]) == 2
        assert checkpoint["summary"]["overall"]["weighted_decode_speedup"] == 2
        assert checkpoint["summary"]["quality"]["gsm8k"]["target_score"] == 1.0
        assert checkpoint["summary"]["quality"]["gsm8k"]["eagle_score"] == 1.0
        assert (
            checkpoint["summary"]["acceptance_depth_telemetry"][
                "mean_selected_depth"
            ]
            == 4
        )
        assert csv_path.read_text(encoding="utf-8").startswith("run_id,")
        previous_json = json_path.read_text(encoding="utf-8")
        document["not_json_serializable"] = {1, 2}
        try:
            write_checkpoint(
                document,
                [target_row, eagle_row],
                json_path,
                csv_path,
            )
        except TypeError:
            pass
        else:
            raise AssertionError("Invalid JSON unexpectedly serialized")
        assert json_path.read_text(encoding="utf-8") == previous_json

    parser = build_parser()
    parsed = parser.parse_args(
        ["--eagle-model", "bundle", "--output-lengths", "32,128"]
    )
    assert parsed.eagle_model == Path("bundle")
    assert parse_output_lengths(parsed.output_lengths) == [32, 128]
    all_suites = ",".join(sorted(suite_utils.SUPPORTED_SUITES))
    suite_args = parser.parse_args(
        [
            "--eagle-model",
            "bundle",
            "--suites",
            all_suites,
            "--allow-code-execution",
            "--gsm8k-problems",
            "1",
            "--ifeval-problems",
            "1",
            "--math500-problems",
            "1",
            "--longbench-problems-per-task",
            "1",
            "--humaneval-problems",
            "1",
            "--humanevalplus-problems",
            "1",
            "--mbppplus-problems",
            "1",
            "--livecodebench-problems",
            "1",
        ]
    )
    assert parse_suite_names(suite_args.suites) == sorted(
        suite_utils.SUPPORTED_SUITES
    )
    print("benchmark_eagle.py self-tests passed")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        run_self_tests()
        return

    script_path = Path(__file__).resolve()
    prefix = _prepare_output_prefix(args.output, script_path)
    args.output = prefix
    log_path = _output_path(prefix, ".log")
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log_file)):
            with contextlib.redirect_stderr(_Tee(sys.stderr, log_file)):
                _run_benchmark_cli(args, script_path)


if __name__ == "__main__":
    main()

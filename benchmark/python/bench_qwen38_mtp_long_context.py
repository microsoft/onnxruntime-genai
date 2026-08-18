#!/usr/bin/env python3
"""Benchmark one exact Qwen3.8 MTP context length with NVML-v2 memory sampling.

Run each context length in a separate process because the CUDA arena retains its
high-water allocation:

    CUDA_VISIBLE_DEVICES=0 python bench_qwen38_mtp_long_context.py \
            --model-dir <rc1-model> --context-length 2048 \
      --generate-length 1024 --spec 3 --json-out result.json
"""

import argparse
import hashlib
import json
import math
import os
import struct
import threading
import time
from pathlib import Path

import onnxruntime_genai as og
import pynvml

from mtp_model_utils import load_main_model, load_mtp_model


PROMPT_SEED = (
    "Explain how transformer inference processes token embeddings, attention, recurrent state, "
    "and feed-forward layers. Include implementation details and verify each conclusion. "
)

SOURCE_EXTENSIONS = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py"}
SOURCE_EXCLUDED_PARTS = {
    ".git",
    "build",
    "cmake",
    "external",
    "node_modules",
    "packages",
    "testdata",
    "third_party",
}


def exact_prompt_ids(tokenizer, target_tokens):
    prefix = list(tokenizer.encode("<|im_start|>user\n"))
    suffix = list(tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n"))
    seed = list(tokenizer.encode(PROMPT_SEED))
    body_tokens = target_tokens - len(prefix) - len(suffix)
    if body_tokens < 0:
        raise ValueError(f"target_tokens={target_tokens} is too small for the chat framing")
    repeats = math.ceil(body_tokens / len(seed))
    result = prefix + (seed * repeats)[:body_tokens] + suffix
    if len(result) != target_tokens:
        raise AssertionError(f"constructed {len(result)} prompt tokens, expected {target_tokens}")
    return result


def source_files(source_roots):
    files = []
    for root_index, root_text in enumerate(source_roots):
        root = Path(root_text).resolve()
        if not root.is_dir():
            raise ValueError(f"source root is not a directory: {root}")
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if (
                path.is_file()
                and path.suffix.lower() in SOURCE_EXTENSIONS
                and not SOURCE_EXCLUDED_PARTS.intersection(relative.parts)
                and path.stat().st_size <= 512 * 1024
            ):
                files.append((root_index, root.name, relative.as_posix(), path))
    return sorted(files, key=lambda item: (item[0], item[2]))


def exact_source_prompt_ids(tokenizer, target_tokens, source_roots):
    prefix_text = (
        "<|im_start|>user\n"
        "Review the following ONNX Runtime and ONNX Runtime GenAI source code. Identify the most "
        "important correctness, performance, and maintainability improvements. Cite file paths and "
        "explain concrete changes.\n\n"
    )
    suffix_text = (
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n"
    )
    prefix = list(tokenizer.encode(prefix_text))
    suffix = list(tokenizer.encode(suffix_text))
    body_budget = target_tokens - len(prefix) - len(suffix)
    if body_budget <= 0:
        raise ValueError(f"target_tokens={target_tokens} is too small for the source-review framing")

    body = []
    used_files = []
    source_bytes = 0
    corpus_hash = hashlib.sha256()
    for _, root_name, relative, path in source_files(source_roots):
        raw = path.read_bytes()
        if b"\x00" in raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        section = f"\n===== {root_name}/{relative} =====\n{text}"
        section_tokens = list(tokenizer.encode(section))
        remaining = body_budget - len(body)
        if remaining <= 0:
            break
        taken = section_tokens[:remaining]
        body.extend(taken)
        used_files.append(f"{root_name}/{relative}")
        source_bytes += len(raw)
        corpus_hash.update(section.encode("utf-8"))
        if len(taken) < len(section_tokens):
            break

    if len(body) != body_budget:
        raise RuntimeError(
            f"source roots supplied only {len(body)} of {body_budget} required body tokens"
        )
    result = prefix + body + suffix
    token_hash = hashlib.sha256()
    for token in result:
        token_hash.update(struct.pack("<I", int(token)))
    metadata = {
        "prompt_source": "source-code",
        "source_roots": [str(Path(root).resolve()) for root in source_roots],
        "source_file_count": len(used_files),
        "source_first_file": used_files[0],
        "source_last_file": used_files[-1],
        "source_bytes_read": source_bytes,
        "source_corpus_sha256": corpus_hash.hexdigest(),
        "prompt_token_sha256": token_hash.hexdigest(),
    }
    return result, metadata


class MemorySampler:
    def __init__(self, interval_seconds=0.002):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        if visible.isdigit():
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible))
        else:
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(visible)
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def used_bytes(self):
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle, version=pynvml.nvmlMemory_v2)
        return int(info.used)

    def start(self):
        self.peak_bytes = self.used_bytes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self):
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, self.used_bytes())

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.used_bytes())
        return self.peak_bytes


def generate(model, head, prompt_ids, generated_tokens, spec):
    prompt_length = len(prompt_ids)
    total_length = prompt_length + generated_tokens
    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        max_length=total_length,
        min_length=total_length,
        batch_size=1,
    )
    params.set_speculative_options(max_draft_tokens=spec)
    generator = og.MtpGenerator(model, head, params)

    wall_start = time.perf_counter()
    generator.append_tokens(prompt_ids)
    generator.generate_next_token()
    ttft_seconds = time.perf_counter() - wall_start

    decode_start = time.perf_counter()
    while not generator.is_done() and len(generator.get_sequence()) < total_length:
        generator.generate_next_token()
    decode_seconds = time.perf_counter() - decode_start
    wall_seconds = time.perf_counter() - wall_start
    sequence = generator.get_sequence()
    actual_generated = len(sequence) - prompt_length
    generated_hash = hashlib.sha256()
    for token in sequence[prompt_length:]:
        generated_hash.update(struct.pack("<I", int(token)))
    stats = dict(generator.get_stats())
    del generator
    if actual_generated != generated_tokens:
        raise RuntimeError(f"generated {actual_generated} tokens, expected exactly {generated_tokens}")
    return ttft_seconds, decode_seconds, wall_seconds, stats, generated_hash.hexdigest()


def mib(value):
    return value / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--head-dir",
        default="",
        help="legacy standalone head directory; default derives model.mtp from --model-dir",
    )
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--generate-length", type=int, default=1024)
    parser.add_argument("--spec", type=int, default=3)
    parser.add_argument("--mtp-prefill-chunk", type=int, default=1024)
    parser.add_argument("--prompt-source", choices=("synthetic", "source-code"), default="synthetic")
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        help="source tree used by --prompt-source=source-code; may be repeated",
    )
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    os.environ["ORT_MTP_NUM_SPECULATIVE_TOKENS"] = str(args.spec)
    os.environ["ORT_MTP_PREFILL_CHUNK"] = str(args.mtp_prefill_chunk)

    pynvml.nvmlInit()
    sampler = MemorySampler()
    baseline_bytes = sampler.used_bytes()
    sampler.start()

    load_start = time.perf_counter()
    model = load_main_model(og, args.model_dir)
    after_main_load_bytes = sampler.used_bytes()
    head = load_mtp_model(og, args.model_dir, args.head_dir)
    load_seconds = time.perf_counter() - load_start
    tokenizer = og.Tokenizer(model)
    after_head_load_bytes = sampler.used_bytes()

    if not args.skip_warmup:
        warmup_ids = exact_prompt_ids(tokenizer, 128)
        generate(model, head, warmup_ids, 32, args.spec)
    after_warmup_bytes = sampler.used_bytes()

    prompt_metadata = {"prompt_source": args.prompt_source}
    if args.prompt_source == "source-code":
        if not args.source_root:
            parser.error("--prompt-source=source-code requires at least one --source-root")
        prompt_ids, prompt_metadata = exact_source_prompt_ids(
            tokenizer, args.context_length, args.source_root
        )
    else:
        prompt_ids = exact_prompt_ids(tokenizer, args.context_length)
    ttft_seconds, decode_seconds, wall_seconds, stats, generated_token_sha256 = generate(
        model, head, prompt_ids, args.generate_length, args.spec
    )
    peak_bytes = sampler.stop()

    decode_tokens = args.generate_length - 1
    accepts = int(stats.get("accepts", 0))
    trials = int(stats.get("trials", 0))
    forwards = int(stats.get("forwards", 0))
    result = {
        "context_length": args.context_length,
        "generate_length": args.generate_length,
        "speculative_tokens": args.spec,
        "mtp_prefill_chunk": args.mtp_prefill_chunk,
        "warmup": not args.skip_warmup,
        "load_seconds": load_seconds,
        "ttft_ms": ttft_seconds * 1000,
        "prefill_tps": args.context_length / ttft_seconds,
        "decode_tps": decode_tokens / decode_seconds,
        "decode_ms_per_token": decode_seconds * 1000 / decode_tokens,
        "wall_seconds": wall_seconds,
        "wall_tps": args.generate_length / wall_seconds,
        "mtp_forwards": forwards,
        "mtp_accepts": accepts,
        "mtp_trials": trials,
        "mtp_acceptance": accepts / trials if trials else None,
        "tokens_per_target_forward": args.generate_length / forwards if forwards else None,
        "generated_token_sha256": generated_token_sha256,
        "gpu_baseline_mib": mib(baseline_bytes),
        "gpu_after_main_load_mib": mib(after_main_load_bytes),
        "gpu_after_head_load_mib": mib(after_head_load_bytes),
        "gpu_after_warmup_mib": mib(after_warmup_bytes),
        "gpu_peak_mib": mib(peak_bytes),
        "gpu_main_load_delta_mib": mib(after_main_load_bytes - baseline_bytes),
        "gpu_head_load_delta_mib": mib(after_head_load_bytes - after_main_load_bytes),
        "gpu_total_peak_delta_mib": mib(peak_bytes - baseline_bytes),
        **prompt_metadata,
    }
    print("RESULT " + json.dumps(result), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
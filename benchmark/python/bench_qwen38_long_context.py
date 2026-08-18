#!/usr/bin/env python3
"""Benchmark one exact Qwen3.8 context length with NVML peak-memory sampling."""

import argparse
import json
import math
import os
import threading
import time

import onnxruntime_genai as og
import pynvml


PROMPT_SEED = (
    "Explain how transformer inference processes token embeddings, attention, recurrent state, "
    "and feed-forward layers. Include implementation details and verify each conclusion. "
)


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


def generate(model, prompt_ids, generated_tokens, chunk_size):
    prompt_length = len(prompt_ids)
    total_length = prompt_length + generated_tokens
    params = og.GeneratorParams(model)
    search_options = dict(
        do_sample=False,
        max_length=total_length,
        min_length=total_length,
        batch_size=1,
    )
    if chunk_size:
        search_options["chunk_size"] = chunk_size
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)

    wall_start = time.perf_counter()
    generator.append_tokens(prompt_ids)
    generator.generate_next_token()
    ttft_seconds = time.perf_counter() - wall_start

    decode_start = time.perf_counter()
    while not generator.is_done():
        generator.generate_next_token()
    decode_seconds = time.perf_counter() - decode_start
    wall_seconds = time.perf_counter() - wall_start
    actual_generated = generator.token_count() - prompt_length
    del generator
    if actual_generated != generated_tokens:
        raise RuntimeError(f"generated {actual_generated} tokens, expected exactly {generated_tokens}")
    return ttft_seconds, decode_seconds, wall_seconds


def mib(value):
    return value / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--generate-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    pynvml.nvmlInit()
    sampler = MemorySampler()
    baseline_bytes = sampler.used_bytes()

    load_start = time.perf_counter()
    model = og.Model(args.model_dir)
    load_seconds = time.perf_counter() - load_start
    tokenizer = og.Tokenizer(model)
    after_load_bytes = sampler.used_bytes()

    if not args.skip_warmup:
        warmup_ids = exact_prompt_ids(tokenizer, 128)
        generate(model, warmup_ids, 32, args.chunk_size)
    after_warmup_bytes = sampler.used_bytes()

    prompt_ids = exact_prompt_ids(tokenizer, args.context_length)
    sampler.start()
    ttft_seconds, decode_seconds, wall_seconds = generate(
        model, prompt_ids, args.generate_length, args.chunk_size
    )
    peak_bytes = sampler.stop()

    decode_tokens = args.generate_length - 1
    result = {
        "context_length": args.context_length,
        "generate_length": args.generate_length,
        "prefill_chunk_size": args.chunk_size,
        "warmup": not args.skip_warmup,
        "load_seconds": load_seconds,
        "ttft_ms": ttft_seconds * 1000,
        "prefill_tps": args.context_length / ttft_seconds,
        "decode_tps": decode_tokens / decode_seconds,
        "decode_ms_per_token": decode_seconds * 1000 / decode_tokens,
        "wall_seconds": wall_seconds,
        "wall_tps": args.generate_length / wall_seconds,
        "gpu_baseline_mib": mib(baseline_bytes),
        "gpu_after_load_mib": mib(after_load_bytes),
        "gpu_after_warmup_mib": mib(after_warmup_bytes),
        "gpu_peak_mib": mib(peak_bytes),
        "gpu_model_load_delta_mib": mib(after_load_bytes - baseline_bytes),
        "gpu_workload_peak_delta_mib": mib(peak_bytes - after_load_bytes),
        "gpu_total_peak_delta_mib": mib(peak_bytes - baseline_bytes),
    }
    print("RESULT " + json.dumps(result), flush=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
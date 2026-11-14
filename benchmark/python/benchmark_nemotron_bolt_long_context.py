# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Continuous-batching long-context benchmark for native Nemotron-Bolt (`og.Engine`).

Evaluates `255K` (`261,120`-token) prompt prefill + `1K` (`1,024`-token) speculative
generation (`262,144` total context) across `runtime_profiles` for `max_batch_size`
`1, 2, 4, 8` (`gpu-256k-b1`, `gpu-256k-b2`, `gpu-256k-b4`, `gpu-256k-b8`), running
each profile in a fresh subprocess for isolated CUDA allocator and NVML VRAM telemetry.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import onnxruntime_genai as og


class _NvmlMemoryInfo(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class _NvmlProcessInfo(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, object]]] = [
        ("pid", ctypes.c_uint),
        ("used_gpu_memory", ctypes.c_ulonglong),
        ("gpu_instance_id", ctypes.c_uint),
        ("compute_instance_id", ctypes.c_uint),
    ]


class NvmlSampler:
    """Polls NVML per-process and device-wide VRAM usage in a background thread."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.peak_process_bytes = 0
        self.peak_device_bytes = 0
        self.total_device_bytes = 0
        self._lib = None
        self._handles: list[ctypes.c_void_p] = []
        self._pid = os.getpid()
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError:
            return
        if lib.nvmlInit_v2() != 0:
            return
        count = ctypes.c_uint(0)
        if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value == 0:
            lib.nvmlShutdown()
            return
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible.strip() != "":
            indices = [int(tok.strip()) for tok in visible.split(",") if tok.strip().isdigit()]
        else:
            indices = list(range(count.value))
        handles: list[ctypes.c_void_p] = []
        for idx in indices:
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(idx), ctypes.byref(handle)) == 0:
                handles.append(handle)
        if not handles:
            lib.nvmlShutdown()
            return
        self._lib = lib
        self._handles = handles
        self.sample_once()

    def sample_once(self) -> None:
        if self._lib is None:
            return
        proc_total = 0
        dev_used_total = 0
        dev_cap_total = 0
        not_avail = ctypes.c_ulonglong(-1).value
        for handle in self._handles:
            mem = _NvmlMemoryInfo()
            if self._lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) == 0:
                dev_used_total += int(mem.used)
                dev_cap_total += int(mem.total)
            proc_count = ctypes.c_uint(64)
            infos = (_NvmlProcessInfo * 64)()
            rc = self._lib.nvmlDeviceGetComputeRunningProcesses_v3(handle, ctypes.byref(proc_count), infos)
            if rc == 0:
                for idx in range(proc_count.value):
                    used = int(infos[idx].used_gpu_memory)
                    if int(infos[idx].pid) == self._pid and used != not_avail:
                        proc_total += used
        with self._lock:
            self.total_device_bytes = max(self.total_device_bytes, dev_cap_total)
            self.peak_device_bytes = max(self.peak_device_bytes, dev_used_total)
            self.peak_process_bytes = max(self.peak_process_bytes, proc_total)

    def start(self) -> None:
        if self._lib is None or self._thread is not None:
            return

        def _loop() -> None:
            while not self._stop.wait(self._interval_s):
                self.sample_once()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.sample_once()
        if self._lib is not None:
            self._lib.nvmlShutdown()
            self._lib = None


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    w = pos - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def _build_prompt_tokens(tokenizer, prompt_tokens: int, batch_size: int) -> list[np.ndarray]:
    passages = [
        (
            "In distributed systems and modern GPU inference engines, paged key-value caches decouple "
            "logical sequence positions from physical memory blocks, enabling continuous batching across "
            "heterogeneous request lengths without external fragmentation or quadratic memory waste. "
        ),
        (
            "Hybrid state-space and sparse mixture-of-experts architectures combine linear-time Mamba-2 "
            "recurrent layers with periodic grouped-query attention anchors and top-k routed experts to "
            "sustain sub-linear KV memory footprints at 256K context lengths. "
        ),
        (
            "Speculative block verification amortizes routed expert weight traffic across multiple draft "
            "positions per decode step while preserving exact greedy output parity with autoregressive "
            "target-only execution at every acceptance boundary. "
        ),
        (
            "def schedule_step(decode_slots, prefill_queue, free_blocks, chunk_budget):\n"
            "    admitted = [slot for slot in decode_slots if slot.needed_blocks <= free_blocks]\n"
            "    return admitted, max(0, chunk_budget - sum(s.tokens for s in admitted))\n"
        ),
    ]
    encoded_blocks = [np.asarray(tokenizer.encode(text), dtype=np.int32) for text in passages]
    prompts: list[np.ndarray] = []
    for req_idx in range(batch_size):
        header = np.asarray(
            tokenizer.encode(f"Document analysis request #{req_idx + 1}: summarize technical architecture.\n"),
            dtype=np.int32,
        )
        rotated = encoded_blocks[req_idx % len(encoded_blocks) :] + encoded_blocks[: req_idx % len(encoded_blocks)]
        unit = np.concatenate(rotated)
        body_needed = max(0, prompt_tokens - int(header.size))
        repeats = (body_needed + int(unit.size) - 1) // int(unit.size)
        full = np.concatenate([header, np.tile(unit, max(1, repeats))])[:prompt_tokens]
        if full.size != prompt_tokens:
            raise RuntimeError(f"Expected {prompt_tokens} prompt tokens, built {full.size}")
        prompts.append(np.ascontiguousarray(full, dtype=np.int32))
    return prompts


def _run_worker(args: argparse.Namespace) -> dict:
    profile_id = args.worker_profile
    batch_size = int(args.worker_batch_size)
    if profile_id:
        os.environ["ORTGENAI_ENABLE_RUNTIME_PROFILE"] = profile_id

    ep_lib = args.cuda_ep_library
    if ep_lib is None and os.environ.get("ORT_LIB_PATH"):
        candidate = Path(os.environ["ORT_LIB_PATH"]).resolve().parent / "libonnxruntime_providers_cuda.so"
        if candidate.is_file():
            ep_lib = candidate
    if ep_lib is not None:
        og.register_execution_provider_library("CUDAExecutionProvider", str(ep_lib))

    model_dir = Path(args.model_path).resolve(strict=True)
    raw_cfg = json.loads((model_dir / "genai_config.json").read_text())
    vocab_size = int(raw_cfg["model"]["vocab_size"])
    profile_meta = next((p for p in raw_cfg.get("runtime_profiles", []) if p.get("id") == profile_id), None)

    sampler = NvmlSampler(interval_s=0.05)
    sampler.start()
    try:
        t_load0 = time.perf_counter()
        cfg = og.Config(str(model_dir))
        if not args.allow_eos:
            cfg.overlay(json.dumps({"model": {"eos_token_id": vocab_size - 1}}))
        model = og.Model(cfg)
        tokenizer = og.Tokenizer(model)
        engine = og.Engine(model)

        caps = engine.get_capabilities()
        if int(caps.configured_max_batch_size) < batch_size:
            raise RuntimeError(
                f"Profile {profile_id!r} configured max_batch_size={caps.configured_max_batch_size} < {batch_size}"
            )
        if int(caps.max_request_length) < args.prompt_tokens + args.generation_tokens:
            raise RuntimeError(
                f"Engine max_request_length={caps.max_request_length} < {args.prompt_tokens + args.generation_tokens}"
            )

        event_buf = engine.create_event_buffer(max(128, batch_size * 16))

        if args.warmup_prompt_tokens > 0 and args.warmup_generation_tokens > 0:
            warm_prompts = _build_prompt_tokens(tokenizer, args.warmup_prompt_tokens, batch_size)
            warm_reqs = []
            for i in range(batch_size):
                ro = og.RequestOptions()
                ro.set_max_session_tokens(args.warmup_prompt_tokens + args.warmup_generation_tokens)
                req = engine.create_request(options=ro)
                to = og.TurnOptions(req)
                to.set_do_sample(False)
                to.set_max_generated_tokens(args.warmup_generation_tokens)
                req.begin_turn(warm_prompts[i], to)
                warm_reqs.append(req)
            while engine.has_pending_requests():
                for ev in engine.run(event_buf):
                    if ev.flags & og.EngineEventFlags.FAILED:
                        raise RuntimeError(f"Warmup failed with error_code={ev.error_code}")
                    if ev.flags & og.EngineEventFlags.TURN_FINISHED and ev.request is not None:
                        ev.request.close()
            del warm_reqs
        t_load1 = time.perf_counter()

        prompts = _build_prompt_tokens(tokenizer, args.prompt_tokens, batch_size)
        stats_before = engine.get_speculative_stats()

        requests = []
        req_id_to_idx: dict[int, int] = {}
        emitted_counts = [0] * batch_size
        first_token_times = [-1.0] * batch_size
        last_token_times = [0.0] * batch_size
        itl_samples: list[float] = []
        active_finished = [False] * batch_size

        t_start = time.perf_counter()
        for i in range(batch_size):
            ro = og.RequestOptions()
            ro.set_max_session_tokens(args.prompt_tokens + args.generation_tokens)
            req = engine.create_request(options=ro)
            to = og.TurnOptions(req)
            to.set_do_sample(False)
            to.set_max_generated_tokens(args.generation_tokens)
            req.begin_turn(prompts[i], to)
            requests.append(req)
            req_id_to_idx[id(req)] = i

        all_prefilled_at: float | None = None
        tokens_at_all_prefilled = 0
        pure_decode_steps = 0
        pure_decode_full_b_steps = 0
        pure_decode_full_b_tokens = 0
        pure_decode_full_b_time_s = 0.0
        pure_decode_by_b: dict[int, list[float]] = {}
        total_engine_steps = 0

        while engine.has_pending_requests():
            was_all_prefilled = all_prefilled_at is not None
            active_before = sum(0 if done else 1 for done in active_finished)
            step_t0 = time.perf_counter()
            engine.run(event_buf)
            step_t1 = time.perf_counter()
            total_engine_steps += 1
            step_elapsed_s = step_t1 - step_t0
            elapsed_from_start = step_t1 - t_start

            step_tokens = 0
            for idx_ev in range(len(event_buf)):
                ev = event_buf[idx_ev]
                if ev.flags & og.EngineEventFlags.FAILED:
                    raise RuntimeError(f"Benchmark turn failed: error_code={ev.error_code}")
                ev_req = ev.request
                if ev_req is None:
                    continue
                req_idx = req_id_to_idx.get(id(ev_req))
                if req_idx is None:
                    raise RuntimeError("Unrecognized request in EngineEvent")

                if ev.flags & og.EngineEventFlags.TOKEN:
                    emitted_counts[req_idx] += 1
                    step_tokens += 1
                    if first_token_times[req_idx] < 0.0:
                        first_token_times[req_idx] = elapsed_from_start
                    else:
                        itl_samples.append((step_t1 - last_token_times[req_idx]) * 1000.0)
                    last_token_times[req_idx] = step_t1

                if ev.flags & og.EngineEventFlags.TURN_FINISHED:
                    active_finished[req_idx] = True
                    ev.request.close()

            if all_prefilled_at is None and all(t >= 0.0 for t in first_token_times):
                all_prefilled_at = elapsed_from_start
                tokens_at_all_prefilled = sum(emitted_counts)
            elif was_all_prefilled:
                pure_decode_steps += 1
                if active_before > 0:
                    bucket = pure_decode_by_b.setdefault(active_before, [0.0, 0.0, 0.0])
                    bucket[0] += 1.0
                    bucket[1] += float(step_tokens)
                    bucket[2] += step_elapsed_s
                if active_before == batch_size:
                    pure_decode_full_b_steps += 1
                    pure_decode_full_b_tokens += step_tokens
                    pure_decode_full_b_time_s += step_elapsed_s

        t_end = time.perf_counter()
        stats_after = engine.get_speculative_stats()
        sampler.sample_once()
    finally:
        sampler.stop()

    for i, cnt in enumerate(emitted_counts):
        if cnt != args.generation_tokens:
            raise RuntimeError(f"Request {i} generated {cnt} tokens, expected {args.generation_tokens}")

    e2e_s = t_end - t_start
    total_prompt_tokens = batch_size * args.prompt_tokens
    total_gen_tokens = sum(emitted_counts)
    prefill_phase_s = all_prefilled_at if all_prefilled_at is not None else e2e_s
    pure_decode_s = max(0.0, e2e_s - prefill_phase_s)
    pure_decode_tokens = total_gen_tokens - tokens_at_all_prefilled
    mixed_inflight_decode_tokens = max(0, tokens_at_all_prefilled - batch_size)

    delta_rounds = int(stats_after["rounds"]) - int(stats_before["rounds"])
    delta_eval = int(stats_after["draft_tokens_evaluated"]) - int(stats_before["draft_tokens_evaluated"])
    delta_acc = int(stats_after["draft_tokens_accepted"]) - int(stats_before["draft_tokens_accepted"])
    delta_target_passes = int(stats_after["target_forward_passes"]) - int(stats_before["target_forward_passes"])
    delta_draft_passes = int(stats_after["draft_forward_passes"]) - int(stats_before["draft_forward_passes"])

    elig = profile_meta.get("eligibility", {}) if profile_meta else {}
    overlay_db = profile_meta.get("overlay", {}).get("engine", {}).get("dynamic_batching", {}) if profile_meta else {}

    return {
        "profile_id": profile_id,
        "batch_size": batch_size,
        "prompt_tokens_per_req": args.prompt_tokens,
        "generation_tokens_per_req": args.generation_tokens,
        "total_context_per_req": args.prompt_tokens + args.generation_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_generated_tokens": total_gen_tokens,
        "configured_max_batch_size": int(caps.configured_max_batch_size),
        "configured_max_scheduled_tokens": int(caps.max_scheduled_tokens),
        "configured_max_request_length": int(caps.max_request_length),
        "profile_num_blocks": overlay_db.get("num_blocks"),
        "profile_min_device_memory_bytes": elig.get("minimum_total_device_memory_bytes"),
        "profile_max_device_memory_bytes": elig.get("maximum_total_device_memory_bytes"),
        "model_load_and_graph_warmup_s": round(t_load1 - t_load0, 3),
        "e2e_wall_s": round(e2e_s, 3),
        "prefill_phase_wall_s": round(prefill_phase_s, 3),
        "pure_decode_phase_wall_s": round(pure_decode_s, 3),
        "effective_prefill_tok_s": round(total_prompt_tokens / max(prefill_phase_s, 1e-6), 2),
        "pure_decode_phase_tok_s": (
            round(pure_decode_tokens / pure_decode_s, 2) if pure_decode_s > 1e-6 and pure_decode_tokens > 0 else 0.0
        ),
        "pure_decode_full_batch_steps": pure_decode_full_b_steps,
        "pure_decode_full_batch_tok_s": (
            round(pure_decode_full_b_tokens / pure_decode_full_b_time_s, 2)
            if pure_decode_full_b_time_s > 1e-6 and pure_decode_full_b_tokens > 0
            else 0.0
        ),
        "pure_decode_full_batch_ms_per_step": (
            round(1000.0 * pure_decode_full_b_time_s / pure_decode_full_b_steps, 3)
            if pure_decode_full_b_steps > 0
            else 0.0
        ),
        "pure_decode_by_active_batch": {
            str(b_key): {
                "steps": int(vals[0]),
                "tokens": int(vals[1]),
                "time_s": round(vals[2], 4),
                "tok_s": round(vals[1] / vals[2], 2) if vals[2] > 1e-6 else 0.0,
                "ms_per_step": round(1000.0 * vals[2] / vals[0], 3) if vals[0] > 0 else 0.0,
            }
            for b_key, vals in sorted(pure_decode_by_b.items(), reverse=True)
        },
        "e2e_output_tok_s": round(total_gen_tokens / max(e2e_s, 1e-6), 2),
        "e2e_total_tok_s": round((total_prompt_tokens + total_gen_tokens) / max(e2e_s, 1e-6), 2),
        "mixed_inflight_decode_tokens": mixed_inflight_decode_tokens,
        "pure_decode_tokens": pure_decode_tokens,
        "ttft_s": {
            "min": round(min(first_token_times), 3),
            "mean": round(statistics.fmean(first_token_times), 3),
            "p50": round(_percentile(first_token_times, 0.50), 3),
            "max": round(max(first_token_times), 3),
            "per_request": [round(x, 3) for x in first_token_times],
        },
        "inter_token_latency_ms": {
            "p50": round(_percentile(itl_samples, 0.50), 3),
            "p95": round(_percentile(itl_samples, 0.95), 3),
        },
        "speculative": {
            "rounds": delta_rounds,
            "target_forward_passes": delta_target_passes,
            "draft_forward_passes": delta_draft_passes,
            "acceptance_rate": round(delta_acc / delta_eval, 4) if delta_eval > 0 else 0.0,
            "effective_k": round((total_gen_tokens - batch_size) / delta_rounds, 3) if delta_rounds > 0 else 0.0,
        },
        "engine_steps": {
            "total": total_engine_steps,
            "pure_decode": pure_decode_steps,
        },
        "vram": {
            "peak_process_mib": round(sampler.peak_process_bytes / (1024**2), 1),
            "peak_process_gib": round(sampler.peak_process_bytes / (1024**3), 3),
            "peak_device_mib": round(sampler.peak_device_bytes / (1024**2), 1),
            "peak_device_gib": round(sampler.peak_device_bytes / (1024**3), 3),
            "total_device_gib": round(sampler.total_device_bytes / (1024**3), 3),
        },
    }


def _parse_profile_specs(spec_str: str) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    for raw in spec_str.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Expected '<profile_id>:<batch_size>' in --profiles, got {entry!r}")
        pid, b_str = entry.split(":", 1)
        items.append((pid.strip(), int(b_str.strip())))
    if not items:
        raise ValueError("--profiles must specify at least one '<profile_id>:<batch_size>' pair")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Path to PagedAttention Bolt package")
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=255 * 1024,
        help="Prompt length in tokens per request (default: 261120 = 255K)",
    )
    parser.add_argument(
        "--generation-tokens",
        type=int,
        default=1024,
        help="Generated tokens per request (default: 1024 = 1K)",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="gpu-256k-b1:1,gpu-256k-b2:2,gpu-256k-b4:4,gpu-256k-b8:8",
        help="Comma-separated '<profile_id>:<batch_size>' matrix",
    )
    parser.add_argument("--warmup-prompt-tokens", type=int, default=512)
    parser.add_argument("--warmup-generation-tokens", type=int, default=32)
    parser.add_argument("--cuda-ep-library", type=Path, default=None, help="Optional CUDA plugin EP .so path")
    parser.add_argument("--allow-eos", action="store_true", help="Do not mask EOS tokens during benchmark")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--worker-profile", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-batch-size", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_batch_size is not None:
        result = _run_worker(args)
        print("BOLT_BENCH_RESULT_JSON:" + json.dumps(result), flush=True)
        return

    specs = _parse_profile_specs(args.profiles)
    results: list[dict] = []
    for profile_id, batch_size in specs:
        print(
            f"\n=== Running profile={profile_id} | B={batch_size} | "
            f"prompt={args.prompt_tokens} | gen={args.generation_tokens} ===",
            flush=True,
        )
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--model-path",
            str(args.model_path),
            "--prompt-tokens",
            str(args.prompt_tokens),
            "--generation-tokens",
            str(args.generation_tokens),
            "--warmup-prompt-tokens",
            str(args.warmup_prompt_tokens),
            "--warmup-generation-tokens",
            str(args.warmup_generation_tokens),
            "--worker-profile",
            profile_id,
            "--worker-batch-size",
            str(batch_size),
        ]
        if args.cuda_ep_library is not None:
            cmd.extend(["--cuda-ep-library", str(args.cuda_ep_library)])
        if args.allow_eos:
            cmd.append("--allow-eos")
        child_env = os.environ.copy()
        if profile_id:
            child_env["ORTGENAI_ENABLE_RUNTIME_PROFILE"] = profile_id
        proc = subprocess.run(cmd, env=child_env, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"Worker for {profile_id} (B={batch_size}) exited with code {proc.returncode}")
        marker = "BOLT_BENCH_RESULT_JSON:"
        payload_line = next(
            (line[len(marker) :] for line in reversed(proc.stdout.splitlines()) if line.startswith(marker)),
            None,
        )
        if payload_line is None:
            sys.stderr.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"Worker for {profile_id} (B={batch_size}) produced no JSON result")
        row = json.loads(payload_line)
        results.append(row)
        print(
            f"  [OK] {profile_id} (B={batch_size}, blocks={row['profile_num_blocks']}): "
            f"prefill={row['effective_prefill_tok_s']:.1f} tok/s ({row['prefill_phase_wall_s']:.2f}s, "
            f"TTFT min/max={row['ttft_s']['min']:.2f}/{row['ttft_s']['max']:.2f}s) | "
            f"pure_decode={row['pure_decode_phase_tok_s']:.2f} tok/s "
            f"(full-B={row['pure_decode_full_batch_tok_s']:.2f} tok/s) | "
            f"e2e_wall={row['e2e_wall_s']:.2f}s | eff_k={row['speculative']['effective_k']:.2f} | "
            f"VRAM proc={row['vram']['peak_process_gib']:.3f} GiB (dev={row['vram']['peak_device_gib']:.3f} GiB)",
            flush=True,
        )

    summary = {
        "model_path": str(args.model_path.resolve()),
        "prompt_tokens_per_req": args.prompt_tokens,
        "generation_tokens_per_req": args.generation_tokens,
        "results": results,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nSaved benchmark summary to {args.output_json}", flush=True)


if __name__ == "__main__":
    main()

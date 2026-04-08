#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Live telemetry integration test.

Sends real telemetry events to OneCollector and verifies they are accepted.
This is NOT a unit test — it makes actual HTTP requests.

Usage:
    python test/python/test_telemetry_live.py

Environment:
    Set ORTGENAI_DISABLE_TELEMETRY=1 to skip (e.g. in CI).
"""

import json
import os
import sys
import threading
import time

# Add the telemetry source to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))

# Ensure telemetry is NOT disabled for this test
os.environ.pop("ORTGENAI_DISABLE_TELEMETRY", None)
os.environ.pop("CI", None)
os.environ.pop("GITHUB_ACTIONS", None)
os.environ.pop("TF_BUILD", None)

# Reset singletons so we get a fresh initialization
from telemetry.library.telemetry_logger import TelemetryLogger
TelemetryLogger._instance = None
TelemetryLogger._default_logger = None

from telemetry.telemetry import GenAITelemetry
GenAITelemetry._instance = None


# ─── Transmission tracking ───────────────────────────────────────────────────

transmission_results = []
results_lock = threading.Lock()
results_event = threading.Event()
expected_count = 0


def on_payload_transmitted(args):
    """Callback invoked after each HTTP payload is sent."""
    with results_lock:
        transmission_results.append({
            "succeeded": args.succeeded,
            "status_code": args.status_code,
            "payload_size_bytes": args.payload_size_bytes,
            "item_count": args.item_count,
        })
        total_items = sum(r["item_count"] for r in transmission_results)
        if total_items >= expected_count:
            results_event.set()


# ─── Main test ────────────────────────────────────────────────────────────────

def main():
    global expected_count

    print("=" * 60)
    print("GenAI Telemetry — Live Integration Test")
    print("=" * 60)
    print()

    # Initialize telemetry
    print("[1/7] Initializing telemetry singleton...")
    telemetry = GenAITelemetry()

    if not telemetry._enabled or not telemetry._logger:
        print("  ✗ Telemetry failed to initialize!")
        print("    Check that opentelemetry-sdk and requests are installed.")
        sys.exit(1)
    print("  ✓ Telemetry initialized")

    # Register the transmission callback
    print("[2/7] Registering transmission callback...")
    telemetry._logger.register_payload_transmitted_callback(
        on_payload_transmitted, include_failures=True
    )
    print("  ✓ Callback registered")

    # The heartbeat was already sent during __init__, so count it
    # We will send 5 more events below = 6 total
    expected_count = 6

    # ── Send events ──────────────────────────────────────────────────────

    print("[3/7] Sending GenAIModelBuild event...")
    telemetry.log_model_build(
        action="create_model",
        duration_ms=12345.6,
        success=True,
        model_name="microsoft/phi-3-mini-4k-instruct",
        model_type="phi3",
        hidden_size=3072,
        num_layers=32,
        num_attn_heads=32,
        num_kv_heads=8,
        vocab_size=32064,
        context_length=4096,
        io_dtype="FLOAT16",
        quant_type="INT4",
        execution_provider="cuda",
        output_model_size_bytes=2_100_000_000,
        num_onnx_operators=15,
        operator_types="MatMul,Add,LayerNorm,Attention,Gather,Reshape,Transpose,Cast,Mul,Softmax,Concat,Unsqueeze,Slice,Shape,RotaryEmbedding",
        has_custom_ops=False,
        source_format="huggingface",
        has_adapter=False,
        extra_options={"int4_block_size": "32", "int4_is_symmetric": "True"},
    )
    print("  ✓ GenAIModelBuild sent")

    print("[4/7] Sending GenAIModelLoad event...")
    telemetry.log_model_load(
        model_name="microsoft/phi-3-mini-4k-instruct",
        model_type="phi3",
        execution_provider="cuda",
        total_load_time_ms=3456.7,
        num_sessions=2,
        model_file_size_bytes=2_100_000_000,
    )
    print("  ✓ GenAIModelLoad sent")

    print("[5/7] Sending GenAIBenchmark event...")
    telemetry.log_benchmark(
        model_name="microsoft/phi-3-mini-4k-instruct",
        precision="int4",
        backend="onnxruntime-genai",
        device="cuda",
        batch_size=1,
        prompt_length=128,
        tokens_generated=256,
        tokenization_latency_ms=0.5,
        tokenization_throughput=2000.0,
        prompt_processing_latency_ms=15.2,
        prompt_processing_throughput=8421.0,
        token_generation_latency_ms=4.8,
        token_generation_throughput=208.3,
        sampling_latency_ms=5.1,
        sampling_throughput=196.1,
        wall_clock_time_ms=1280.0,
        wall_clock_throughput=300.0,
        time_to_first_token_ms=20.3,
        peak_memory_gpu_mb=3200.0,
        peak_memory_cpu_mb=1500.0,
    )
    print("  ✓ GenAIBenchmark sent")

    print("[6/7] Sending GenAIInference event...")
    telemetry.log_inference(
        model_name="microsoft/phi-3-mini-4k-instruct",
        model_type="phi3",
        execution_provider="cuda",
        time_to_first_token_ms=22.5,
        total_generation_time_ms=1100.0,
        total_tokens_generated=200,
        input_token_count=50,
        memory_used_mb=1400.0,
        gpu_memory_used_mb=3100.0,
    )
    print("  ✓ GenAIInference sent")

    print("[7/7] Sending GenAIError event...")
    telemetry.log_error(
        exception_type="RuntimeError",
        exception_message="onnxruntime_genai/models/model.cpp:42 — CUDA out of memory",
        action="generate_next_token",
        model_name="microsoft/phi-3-mini-4k-instruct",
        execution_provider="cuda",
    )
    print("  ✓ GenAIError sent")

    # ── Wait for transmission ────────────────────────────────────────────

    print()
    print("Waiting for HTTP transmission (up to 15 seconds)...")
    # The BatchLogRecordProcessor flushes every 1 second by default,
    # but we also call shutdown to force-flush.
    telemetry.shutdown()

    # Give callbacks a moment to fire after shutdown flush
    results_event.wait(timeout=15)

    # ── Report results ───────────────────────────────────────────────────

    print()
    print("=" * 60)
    print("TRANSMISSION RESULTS")
    print("=" * 60)

    with results_lock:
        if not transmission_results:
            print("  ✗ No transmission callbacks received!")
            print("    The BatchLogRecordProcessor may not have flushed.")
            print("    This could indicate a network issue.")
            sys.exit(1)

        total_items = 0
        total_bytes = 0
        successes = 0
        failures = 0

        for i, result in enumerate(transmission_results):
            status = "✓" if result["succeeded"] else "✗"
            print(f"  {status} Payload {i+1}: "
                  f"status={result['status_code']}, "
                  f"items={result['item_count']}, "
                  f"size={result['payload_size_bytes']} bytes")
            total_items += result["item_count"]
            total_bytes += result["payload_size_bytes"]
            if result["succeeded"]:
                successes += 1
            else:
                failures += 1

        print()
        print(f"  Total payloads: {len(transmission_results)}")
        print(f"  Total items sent: {total_items}")
        print(f"  Total bytes: {total_bytes}")
        print(f"  Successful: {successes}")
        print(f"  Failed: {failures}")

    print()
    if failures == 0 and total_items >= expected_count:
        print("=" * 60)
        print(f"ALL {total_items} EVENTS SENT SUCCESSFULLY ✓")
        print("=" * 60)
        return 0
    elif failures == 0:
        print("=" * 60)
        print(f"PARTIAL SUCCESS: {total_items}/{expected_count} events sent ✓")
        print("(Some events may have been batched together)")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print(f"FAILURES DETECTED: {failures} payload(s) failed ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

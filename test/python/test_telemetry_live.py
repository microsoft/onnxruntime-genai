#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Live telemetry integration test.

Logs real events, lets the SQLite-backed uploader drain them to OneCollector,
and verifies the store empties (every event accepted, HTTP 2xx). This makes
actual HTTP requests; it is NOT a unit test.

Usage:
    python test/python/test_telemetry_live.py
"""

import os
import sys
import threading
import time

# Add the telemetry source to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))

transmission_results = []
results_lock = threading.Lock()


def on_payload_transmitted(args):
    with results_lock:
        transmission_results.append(
            {
                "succeeded": args.succeeded,
                "status_code": args.status_code,
                "payload_size_bytes": args.payload_size_bytes,
                "item_count": args.item_count,
            }
        )


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.environ.pop("ORTGENAI_DISABLE_TELEMETRY", None)
    os.environ.pop("CI", None)
    os.environ.pop("GITHUB_ACTIONS", None)
    os.environ.pop("TF_BUILD", None)

    from telemetry.telemetry import GenAITelemetry
    GenAITelemetry._instance = None

    print("=" * 60)
    print("GenAI Telemetry - Live Integration Test (SQLite uploader)")
    print("=" * 60)
    print()

    print("[1/8] Initializing telemetry singleton...")
    telemetry = GenAITelemetry()
    if not telemetry._enabled or telemetry._store is None or telemetry._uploader is None:
        print("  FAIL: telemetry did not initialize")
        return 1
    tenant = telemetry._instrumentation_key
    print("  OK: store + uploader initialized")

    # Observe HTTP transmissions on the uploader's transport.
    telemetry._uploader._transport.register_payload_transmitted_callback(
        on_payload_transmitted, include_failures=True
    )

    print("[2/8] Heartbeat already enqueued during init.")

    print("[3/8] log_model_build ...")
    telemetry.log_model_build(
        action="create_model", duration_ms=12345.6, success=True,
        model_name="microsoft/phi-3-mini-4k-instruct", model_type="phi3",
        hidden_size=3072, num_layers=32, num_attn_heads=32, num_kv_heads=8,
        vocab_size=32064, context_length=4096, io_dtype="FLOAT16", quant_type="INT4",
        execution_provider="cuda", output_model_size_bytes=2_100_000_000,
        num_onnx_operators=15, operator_types="MatMul,Add,Attention,RotaryEmbedding",
        has_custom_ops=False, source_format="huggingface", has_adapter=False,
        extra_options={"int4_block_size": "32"},
    )

    print("[4/8] log_model_load ...")
    telemetry.log_model_load(
        model_name="phi-3-mini-int4-cuda", model_type="phi3",
        execution_provider="cuda", total_load_time_ms=842.17, num_sessions=2,
    )

    print("[5/8] log_benchmark ...")
    telemetry.log_benchmark(
        model_name="phi-3-mini-int4-cuda", precision="int4", backend="onnxruntime-genai",
        device="cuda", batch_size=1, prompt_length=128, tokens_generated=256,
        token_generation_latency_ms=4.8, token_generation_throughput=208.3,
        time_to_first_token_ms=20.3, peak_memory_gpu_mb=3200.0,
    )

    print("[6/8] log_inference ...")
    telemetry.log_inference(
        model_name="phi-3-mini-int4-cuda", time_to_first_token_ms=22.5,
        total_generation_time_ms=1100.0, total_tokens_generated=200, input_token_count=50,
    )

    print("[7/8] log_error ...")
    telemetry.log_error(
        exception_type="RuntimeError",
        exception_message="model.cpp:42 CUDA out of memory",
        action="generate_next_token", model_name="phi-3-mini-int4-cuda",
        execution_provider="cuda",
    )

    pending_before = telemetry._store.count()
    print(f"  Enqueued; pending in store before flush: {pending_before}")

    print("[8/8] Flushing uploader (drain store) ...")
    telemetry.shutdown(flush_seconds=20.0)

    # Re-open the store read-only to confirm it drained (uploader deletes on 2xx).
    from telemetry.deviceid import get_telemetry_base_dir
    from telemetry.offline_store import OfflineEventStore
    db_path = os.path.join(get_telemetry_base_dir(), "genai_telemetry.db")
    remaining = OfflineEventStore(db_path).count()

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    with results_lock:
        for i, r in enumerate(transmission_results):
            ok = "OK " if r["succeeded"] else "ERR"
            print(f"  {ok} payload {i+1}: status={r['status_code']} items={r['item_count']} size={r['payload_size_bytes']}B")
        failures = sum(1 for r in transmission_results if not r["succeeded"])
        sent_items = sum(r["item_count"] for r in transmission_results if r["succeeded"])
    print(f"  pending before flush: {pending_before}")
    print(f"  remaining in store after flush (this tenant): {remaining}")
    print(f"  HTTP payloads: {len(transmission_results)}, failures: {failures}, items delivered: {sent_items}")
    print()

    if remaining == 0 and failures == 0 and pending_before >= 6:
        print("=" * 60)
        print(f"ALL {pending_before} EVENTS DELIVERED (store drained, HTTP 2xx)")
        print("=" * 60)
        return 0
    print("=" * 60)
    print("FAILURE: store did not fully drain or a send failed")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())

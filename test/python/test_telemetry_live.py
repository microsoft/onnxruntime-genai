#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Live telemetry integration test.

Logs real events and verifies they reach OneCollector (HTTP 2xx):

- The device-id heartbeat and five detailed events are persisted to the SQLite
  store and shipped by the background uploader, then the store is confirmed empty.

Every outgoing HTTP send is recorded by wrapping the transport, so uploader
batches are observed. This makes actual HTTP requests; it is NOT a unit test.

Usage:
    python test/python/test_telemetry_live.py
"""

import os
import sys
import threading

transmission_results = []
results_lock = threading.Lock()


def main():
    telemetry_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))
    if telemetry_root not in sys.path:
        sys.path.insert(0, telemetry_root)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    os.environ.pop("ORT_DISABLE_TELEMETRY", None)
    os.environ.pop("CI", None)
    os.environ.pop("GITHUB_ACTIONS", None)
    os.environ.pop("TF_BUILD", None)

    # Record every HTTP send (heartbeat + uploader batches) by wrapping the
    # transport before any telemetry object is created.
    import telemetry.library.transport as transport_mod

    original_send = transport_mod.HttpJsonPostTransport.send

    def recording_send(self, payload, timeout_sec, item_count=1):
        succeeded, status_code = original_send(self, payload, timeout_sec, item_count)
        with results_lock:
            transmission_results.append(
                {
                    "succeeded": succeeded,
                    "status_code": status_code,
                    "payload_size_bytes": len(payload),
                    "item_count": item_count,
                    "has_heartbeat": b"GenAIHeartbeat" in bytes(payload),
                }
            )
        return succeeded, status_code

    transport_mod.HttpJsonPostTransport.send = recording_send

    from telemetry.telemetry import GenAITelemetry

    GenAITelemetry._instance = None

    print("=" * 60)
    print("GenAI Telemetry - Live Integration Test (SQLite uploader)")
    print("=" * 60)
    print()

    try:
        print("[1/8] Initializing telemetry singleton...")
        telemetry = GenAITelemetry()
        if not telemetry._enabled or telemetry._store is None or telemetry._uploader is None:
            print("  FAIL: telemetry did not initialize")
            return 1
        print("  OK: store + uploader initialized")

        # The heartbeat is enqueued on a background thread (system-info
        # collection uses blocking subprocesses); wait for that thread so it is
        # in the store before we add the detailed events.
        print("[2/8] Letting the background heartbeat be collected/enqueued...")
        if telemetry._heartbeat_thread is not None:
            telemetry._heartbeat_thread.join(20)

        print("[3/8] log_model_build ...")
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
            operator_types="MatMul,Add,Attention,RotaryEmbedding",
            has_custom_ops=False,
            source_format="huggingface",
            has_adapter=False,
            extra_options={"int4_block_size": "32"},
        )

        print("[4/8] log_model_load ...")
        telemetry.log_model_load(
            model_name="phi-3-mini-int4-cuda",
            model_type="phi3",
            execution_provider="cuda",
            total_load_time_ms=842.17,
            num_sessions=2,
        )

        print("[5/8] log_benchmark ...")
        telemetry.log_benchmark(
            model_name="phi-3-mini-int4-cuda",
            precision="int4",
            backend="onnxruntime-genai",
            device="cuda",
            batch_size=1,
            prompt_length=128,
            tokens_generated=256,
            token_generation_latency_ms=4.8,
            token_generation_throughput=208.3,
            time_to_first_token_ms=20.3,
            peak_memory_gpu_mb=3200.0,
        )

        print("[6/8] log_inference ...")
        telemetry.log_inference(
            model_name="phi-3-mini-int4-cuda",
            time_to_first_token_ms=22.5,
            total_generation_time_ms=1100.0,
            total_tokens_generated=200,
            input_token_count=50,
        )

        print("[7/8] log_error ...")
        telemetry.log_error(
            exception_type="RuntimeError",
            exception_message="model.cpp:42 CUDA out of memory",
            action="generate_next_token",
            model_name="phi-3-mini-int4-cuda",
            execution_provider="cuda",
        )

        pending_before = telemetry._store.count()
        print(f"  Enqueued; pending detailed events in store before flush: {pending_before}")

        print("[8/8] Flushing uploader (drain store) ...")
        telemetry.shutdown(flush_seconds=20.0)

        # Re-open the store read-only to confirm it drained (uploader deletes on 2xx).
        from telemetry.deviceid import get_telemetry_base_dir
        from telemetry.offline_store import OfflineEventStore

        db_path = os.path.join(get_telemetry_base_dir(), "genai_telemetry.db")
        store = OfflineEventStore(db_path)
        try:
            remaining = store.count()
        finally:
            store.close()
    finally:
        transport_mod.HttpJsonPostTransport.send = original_send

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    with results_lock:
        for i, r in enumerate(transmission_results):
            ok = "OK " if r["succeeded"] else "ERR"
            print(
                f"  {ok} payload {i + 1}: status={r['status_code']} items={r['item_count']}"
                f" size={r['payload_size_bytes']}B"
            )
        failures = sum(1 for r in transmission_results if not r["succeeded"])
        sent_items = sum(r["item_count"] for r in transmission_results if r["succeeded"])
        heartbeat_sent = any(r["succeeded"] and r["has_heartbeat"] for r in transmission_results)
    print(f"  pending before flush: {pending_before}")
    print(f"  remaining in store after flush: {remaining}")
    print(f"  HTTP payloads: {len(transmission_results)}, failures: {failures}, items delivered: {sent_items}")
    print(f"  heartbeat delivered: {heartbeat_sent}")
    print()

    # 1 heartbeat + 5 detailed = 6 items, all through the durable uploader.
    if remaining == 0 and failures == 0 and sent_items >= 6 and heartbeat_sent:
        print("=" * 60)
        print(f"ALL EVENTS DELIVERED ({sent_items} items: heartbeat + 5 detailed, store drained, HTTP 2xx)")
        print("=" * 60)
        return 0
    print("=" * 60)
    print("FAILURE: store did not fully drain, heartbeat missing, or a send failed")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Source/wheel compatibility import for benchmark telemetry privacy helpers."""

import os
import sys
from contextlib import contextmanager

_SOURCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "python", "py"))


@contextmanager
def _source_path():
    path_added = _SOURCE_ROOT not in sys.path
    if path_added:
        sys.path.insert(0, _SOURCE_ROOT)
    try:
        yield
    finally:
        if path_added and _SOURCE_ROOT in sys.path:
            sys.path.remove(_SOURCE_ROOT)


try:
    from onnxruntime_genai.telemetry.path_utils import normalize_execution_provider, sanitize_model_identifier
except ImportError:
    with _source_path():
        from telemetry.path_utils import normalize_execution_provider, sanitize_model_identifier


def get_telemetry():
    """Create telemetry from either the installed wheel or repository source."""
    try:
        from onnxruntime_genai.telemetry import GenAITelemetry  # noqa: PLC0415
    except ImportError:
        with _source_path():
            from telemetry import GenAITelemetry  # noqa: PLC0415

    return GenAITelemetry()


def emit_benchmark_telemetry(
    *,
    model_name,
    precision,
    execution_provider,
    batch_size,
    prompt_length,
    tokens_generated,
    tokenization_latency_ms,
    tokenization_throughput,
    prompt_processing_latency_ms,
    prompt_processing_throughput,
    token_generation_latency_ms,
    token_generation_throughput,
    sampling_latency_ms,
    sampling_throughput,
    wall_clock_time_ms,
    wall_clock_throughput,
    time_to_first_token_ms,
    peak_memory_gpu_mb,
    peak_memory_cpu_mb,
    session_id,
):
    """Emit one benchmark event using already-aggregated benchmark metrics."""
    get_telemetry().log_benchmark(
        model_name=sanitize_model_identifier(model_name),
        precision=precision,
        backend="onnxruntime-genai",
        device=normalize_execution_provider(execution_provider),
        batch_size=batch_size,
        prompt_length=prompt_length,
        tokens_generated=tokens_generated,
        tokenization_latency_ms=tokenization_latency_ms,
        tokenization_throughput=tokenization_throughput,
        prompt_processing_latency_ms=prompt_processing_latency_ms,
        prompt_processing_throughput=prompt_processing_throughput,
        token_generation_latency_ms=token_generation_latency_ms,
        token_generation_throughput=token_generation_throughput,
        sampling_latency_ms=sampling_latency_ms,
        sampling_throughput=sampling_throughput,
        wall_clock_time_ms=wall_clock_time_ms,
        wall_clock_throughput=wall_clock_throughput,
        time_to_first_token_ms=time_to_first_token_ms,
        peak_memory_gpu_mb=peak_memory_gpu_mb,
        peak_memory_cpu_mb=peak_memory_cpu_mb,
        session_id=session_id,
    )


__all__ = [
    "emit_benchmark_telemetry",
    "get_telemetry",
    "normalize_execution_provider",
    "sanitize_model_identifier",
]

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""GenAI Telemetry singleton with OneCollector integration.

Provides high-level telemetry for:
- MAD/DAD tracking via heartbeat events with system metadata
- ModelBuilder instrumentation (model structure, architecture, precision, kernels)
- Benchmark instrumentation (latency, throughput, memory, TTFT)
- Model loading performance (session creation time, TTFT)
- Error/crash reporting
"""

from __future__ import annotations

import base64
import os
import threading
import time
import traceback
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

from .deviceid import get_hashed_device_id_and_status, get_telemetry_base_dir
from .library.options import OneCollectorExporterOptions
from .library.serialization import CommonSchemaJsonSerializationHelper
from .offline_store import OfflineEventStore
from .path_utils import scrub_error_message_for_telemetry, scrub_string_for_telemetry, scrub_value_for_telemetry
from .system_info import get_execution_provider_info, get_system_info
from .uploader import EventUploader

# Event names
HEARTBEAT_EVENT = "GenAIHeartbeat"
MODEL_BUILD_EVENT = "GenAIModelBuild"
BENCHMARK_EVENT = "GenAIBenchmark"
MODEL_LOAD_EVENT = "GenAIModelLoad"
INFERENCE_EVENT = "GenAIInference"
ACTION_EVENT = "GenAIAction"
ERROR_EVENT = "GenAIError"

# CI environment variables that auto-disable telemetry. Keep this aligned with
# src/telemetry/telemetry_environment.h.
_CI_ENV_VARS = {
    "CI",
    "TF_BUILD",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "CODEBUILD_BUILD_ID",
    "BUILDKITE",
    "TEAMCITY_VERSION",
    "APPVEYOR",
    "BITBUCKET_BUILD_NUMBER",
    "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
}
_UNIT_TEST_ENV_VAR = "ORT_RUNNING_UNIT_TESTS"
_HEARTBEAT_ENRICHMENT_GRACE_SECONDS = 60.0
_DISTRIBUTION_NAMES = (
    "onnxruntime-genai",
    "onnxruntime-genai-cuda",
    "onnxruntime-genai-directml",
    "onnxruntime-genai-trt-rtx",
    "onnxruntime-genai-winml",
)


def _is_environment_signal_truthy(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _is_ci_environment() -> bool:
    return any(_is_environment_signal_truthy(os.environ.get(var, "")) for var in _CI_ENV_VARS) or (
        _is_environment_signal_truthy(os.environ.get(_UNIT_TEST_ENV_VAR, ""))
    )


def _is_telemetry_disabled_by_environment() -> bool:
    return os.environ.get("ORT_DISABLE_TELEMETRY", "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _get_app_version() -> str:
    """Resolve the onnxruntime-genai version.

    Tries three sources in order:
    1. The installed onnxruntime_genai package (__version__ attribute)
    2. importlib.metadata (works even when native ext isn't loadable)
    3. VERSION_INFO file at the repo root
    """
    # 1. Try the package attribute (fastest when native ext is loaded)
    with suppress(Exception):
        import onnxruntime_genai  # noqa: PLC0415

        v = getattr(onnxruntime_genai, "__version__", None)
        if v:
            return v

    # 2. Resolve an installed base or provider-specific distribution.
    for distribution in _DISTRIBUTION_NAMES:
        with suppress(PackageNotFoundError):
            return distribution_version(distribution)

    # 3. Fall back to VERSION_INFO file at repository root
    with suppress(Exception):
        d = Path(__file__).resolve().parent
        for _ in range(10):
            candidate = d / "VERSION_INFO"
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
            d = d.parent

    return "unknown"


def _redact_paths(text: str) -> str:
    return scrub_string_for_telemetry(text)


def _redact_error_message(text: str) -> str:
    return scrub_error_message_for_telemetry(text)


def _format_exception_message(ex: BaseException, tb=None) -> str:
    """Format an exception and strip local paths for privacy.

    Each entry from ``traceback.format_exception`` is a multi-line string (the
    ``File "..."`` line plus the offending source line), so we process every
    physical line: filenames are replaced with ``[path]``, and any path that
    remains on a source or message line is redacted so a username embedded in it
    cannot leak.
    """
    formatted = traceback.format_exception(type(ex), ex, tb, limit=5)
    lines = []
    for chunk in formatted:
        for raw_line in chunk.splitlines():
            line_trunc = raw_line.strip()
            if line_trunc.startswith('File "'):
                path_end = line_trunc.find('"', len('File "'))
                if path_end != -1:
                    line_trunc = f'File "[path]"{line_trunc[path_end + 1 :]}'
            line_trunc = _redact_error_message(line_trunc)
            lines.append(line_trunc)
    return "\n".join(lines)


class GenAITelemetry:
    """Singleton telemetry manager for ONNX Runtime GenAI.

    Thread-safe singleton that sends telemetry to Microsoft OneCollector.
    CI and ORT_DISABLE_TELEMETRY suppression are latched for the process lifetime.
    """

    _instance: GenAITelemetry | None = None
    _lock = threading.RLock()
    _process_disabled = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    instance._telemetry_disabled = cls._process_disabled
                    instance._next_model_session_id = 1
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        with self._lock:
            if self._initialized:
                return

            self._initialized = True
            self._enabled = True
            self._store = None
            self._uploader = None
            self._instrumentation_key = ""
            self._envelope_ikey = ""
            self._app_version = "unknown"
            self._app_name = "onnxruntime-genai"
            self._heartbeat_thread: threading.Thread | None = None

            # Full suppression is process-wide and irreversible, including later
            # initialization attempts after shutdown or environment changes.
            if self._telemetry_disabled or _is_ci_environment() or _is_telemetry_disabled_by_environment():
                self._telemetry_disabled = True
                self._enabled = False
                return

            self._app_session_guid = str(uuid.uuid4())

            try:
                self._app_version = _get_app_version()
                connection_string = base64.b64decode(
                    "SW5zdHJ1bWVudGF0aW9uS2V5PTlkNWRkYWVjNjFlMjQ1NjdiNzg4YTIwYWVhMzI0NjMxLWQyMTZmODZmLTQ4NzQtNDU5Yi1hMzM1LWIzYTliODBhY2FkNi03MzI3"
                ).decode()
                options = OneCollectorExporterOptions(connection_string=connection_string)
                options.validate()
                self._instrumentation_key = options.instrumentation_key
                self._envelope_ikey = (
                    f"{CommonSchemaJsonSerializationHelper.ONE_COLLECTOR_TENANCY_SYMBOL}:{options.tenant_token}"
                )

                # Durable on-disk queue + uploader. Events survive process exit,
                # so there is no exit-time flush. The uploader retries until
                # delivery.
                db_path = os.path.join(get_telemetry_base_dir(), "genai_telemetry.db")
                self._store = OfflineEventStore(db_path)
                if not self._store.is_open:
                    self._store = None
                    self._enabled = False
                    self._initialized = False
                    return
                # Persist the counting-critical heartbeat before any background
                # work so even a short-lived builder process leaves it durable.
                heartbeat_id = self._persist(
                    HEARTBEAT_EVENT,
                    self._minimal_heartbeat_attributes(),
                    available_after_seconds=_HEARTBEAT_ENRICHMENT_GRACE_SECONDS,
                )

                self._uploader = EventUploader(self._store, instrumentation_key=self._instrumentation_key)
                self._uploader.start()

                # System enrichment can use blocking subprocesses. Replace the
                # queued heartbeat only if it has not already been delivered.
                self._heartbeat_thread = threading.Thread(
                    target=self._send_heartbeat,
                    args=(heartbeat_id,),
                    name="genai-telemetry-heartbeat",
                    daemon=True,
                )
                self._heartbeat_thread.start()
            except Exception:
                self._enabled = False
                self.shutdown(1.0)

    def _common_context(self) -> dict[str, str]:
        return {
            "appName": self._app_name,
            "LibraryVersion": self._app_version,
            "AppSessionGuid": self._app_session_guid,
        }

    def _serialize_event(self, event_name: str, attributes: dict[str, Any] | None = None) -> bytes:
        data = self._common_context()
        if attributes:
            scrubbed_attributes = scrub_value_for_telemetry(attributes)
            if isinstance(attributes.get("exceptionMessage"), str):
                scrubbed_attributes["exceptionMessage"] = scrub_error_message_for_telemetry(
                    attributes["exceptionMessage"]
                )
            data.update(scrubbed_attributes)
        envelope = CommonSchemaJsonSerializationHelper.create_event_envelope(
            event_name=event_name,
            timestamp=datetime.now(timezone.utc),
            ikey=self._envelope_ikey,
            data=data,
        )
        return CommonSchemaJsonSerializationHelper.serialize_to_json_bytes(envelope)

    def _persist(
        self,
        event_name: str,
        attributes: dict[str, Any] | None = None,
        available_after_seconds: float = 0.0,
    ) -> int | None:
        if not self._enabled or self._store is None:
            return None
        try:
            return self._store.store_with_id(
                self._serialize_event(event_name, attributes),
                available_after_seconds=available_after_seconds,
            )
        except Exception:
            return None

    def allocate_model_session_id(self) -> int:
        """Allocate a process-local, monotonic ID for one model lifecycle."""
        with self._lock:
            session_id = self._next_model_session_id
            self._next_model_session_id += 1
        return session_id

    def _emit(self, event_name: str, attributes: dict[str, Any] | None = None) -> None:
        """Serialize an event to a Common Schema envelope and persist it durably."""
        if not self._enabled or self._store is None:
            return
        try:
            row_id = self._persist(event_name, attributes)
            if row_id is not None and self._uploader is not None:
                self._uploader.request_drain()
        except Exception:
            return

    @property
    def accepts_detailed_events(self) -> bool:
        """Whether detailed events can currently be persisted."""
        return bool(self._enabled and self._store is not None and self._store.is_open)

    def _minimal_heartbeat_attributes(self) -> dict[str, Any]:
        device_id, id_status = get_hashed_device_id_and_status()
        return {
            "sessionId": 0,
            "deviceId": device_id,
            "deviceIdStatus": id_status.value,
        }

    def _build_heartbeat_attributes(self) -> dict[str, Any]:
        """Collect device-id + system info for the heartbeat event."""
        sys_info = get_system_info()
        ep_info = get_execution_provider_info()
        return {
            **self._minimal_heartbeat_attributes(),
            "os": sys_info.get("os", ""),
            "osVersion": sys_info.get("os_version", ""),
            "osRelease": sys_info.get("os_release", ""),
            "osArchitecture": sys_info.get("os_arch", ""),
            "processorCount": sys_info.get("processor_count", 0),
            "cpuModel": sys_info.get("cpu_model", ""),
            "totalMemoryMB": sys_info.get("total_memory_mb", 0),
            "gpuName": sys_info.get("gpu_name", ""),
            "gpuDriverVersion": sys_info.get("gpu_driver_version", ""),
            "gpuMemoryMB": sys_info.get("gpu_memory_mb", 0),
            "gpuCount": sys_info.get("gpu_count", 0),
            "deviceManufacturer": sys_info.get("device_manufacturer", ""),
            "deviceModel": sys_info.get("device_model", ""),
            "pythonVersion": sys_info.get("python_version", ""),
            "ortVersion": sys_info.get("ort_version", ""),
            "availableProviders": ",".join(ep_info.get("available_providers", [])),
        }

    def _send_heartbeat(self, row_id: int | None = None) -> None:
        """Best-effort enrichment of the already-durable heartbeat."""
        if not self._enabled or self._telemetry_disabled or self._store is None:
            return
        ready = False
        try:
            attributes = self._build_heartbeat_attributes()
            with self._lock:
                if not self._enabled or self._telemetry_disabled or self._store is None:
                    return
                if row_id is None:
                    self._emit(HEARTBEAT_EVENT, attributes)
                    return
                ready = self._store.replace(row_id, self._serialize_event(HEARTBEAT_EVENT, attributes))
        except Exception:
            pass
        finally:
            if row_id is not None and not ready:
                with self._lock:
                    if self._enabled and not self._telemetry_disabled and self._store is not None:
                        ready = self._store.make_available(row_id)
            if ready and self._enabled and not self._telemetry_disabled and self._uploader is not None:
                self._uploader.request_drain()

    def log(self, event_name: str, attributes: dict[str, Any] | None = None) -> None:
        """Log a generic telemetry event."""
        if not self._enabled or self._store is None:
            return
        with suppress(Exception):
            self._emit(event_name, attributes)

    def log_model_build(
        self,
        action: str,
        duration_ms: float,
        success: bool,
        model_name: str = "",
        model_type: str = "",
        hidden_size: int = 0,
        num_layers: int = 0,
        num_attn_heads: int = 0,
        num_kv_heads: int = 0,
        vocab_size: int = 0,
        context_length: int = 0,
        io_dtype: str = "",
        quant_type: str = "",
        execution_provider: str = "",
        output_model_size_bytes: int = 0,
        num_onnx_operators: int = 0,
        operator_types: str = "",
        has_custom_ops: bool = False,
        source_format: str = "",
        has_adapter: bool = False,
        extra_options: dict[str, Any] | None = None,
    ) -> None:
        """Log a ModelBuilder telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            attributes = {
                "action": action,
                "durationMs": duration_ms,
                "success": success,
                "modelName": _redact_paths(model_name),
                "modelType": model_type,
                "hiddenSize": hidden_size,
                "numLayers": num_layers,
                "numAttentionHeads": num_attn_heads,
                "numKeyValueHeads": num_kv_heads,
                "vocabSize": vocab_size,
                "contextLength": context_length,
                "ioDtype": io_dtype,
                "quantType": quant_type,
                "executionProvider": execution_provider,
                "outputModelSizeBytes": output_model_size_bytes,
                "numOnnxOperators": num_onnx_operators,
                "operatorTypes": operator_types,
                "hasCustomOps": has_custom_ops,
                "sourceFormat": source_format,
                "hasAdapter": has_adapter,
            }
            if extra_options:
                attributes["extraOptions"] = scrub_value_for_telemetry(extra_options)
            self._emit(MODEL_BUILD_EVENT, attributes)
        except Exception:
            return

    def log_benchmark(
        self,
        model_name: str = "",
        precision: str = "",
        backend: str = "",
        device: str = "",
        batch_size: int = 0,
        prompt_length: int = 0,
        tokens_generated: int = 0,
        tokenization_latency_ms: float = 0.0,
        tokenization_throughput: float = 0.0,
        prompt_processing_latency_ms: float = 0.0,
        prompt_processing_throughput: float = 0.0,
        token_generation_latency_ms: float = 0.0,
        token_generation_throughput: float = 0.0,
        sampling_latency_ms: float = 0.0,
        sampling_throughput: float = 0.0,
        wall_clock_time_ms: float = 0.0,
        wall_clock_throughput: float = 0.0,
        time_to_first_token_ms: float = 0.0,
        peak_memory_gpu_mb: float = 0.0,
        peak_memory_cpu_mb: float = 0.0,
        session_id: int | None = None,
    ) -> None:
        """Log benchmark telemetry; prompt/tokenization latency fields are total milliseconds per prompt."""
        if not self._enabled or self._store is None:
            return
        try:
            session_id = session_id if session_id is not None else self.allocate_model_session_id()
            attributes = {
                "sessionId": session_id,
                "modelName": _redact_paths(model_name),
                "precision": precision,
                "backend": backend,
                "device": device,
                "batchSize": batch_size,
                "promptLength": prompt_length,
                "tokensGenerated": tokens_generated,
                "tokenizationLatencyMs": tokenization_latency_ms,
                "tokenizationThroughput": tokenization_throughput,
                "promptProcessingLatencyMs": prompt_processing_latency_ms,
                "promptProcessingThroughput": prompt_processing_throughput,
                "tokenGenerationLatencyMs": token_generation_latency_ms,
                "tokenGenerationThroughput": token_generation_throughput,
                "samplingLatencyMs": sampling_latency_ms,
                "samplingThroughput": sampling_throughput,
                "wallClockTimeMs": wall_clock_time_ms,
                "wallClockThroughput": wall_clock_throughput,
                "timeToFirstTokenMs": time_to_first_token_ms,
                "peakGpuMemoryMB": peak_memory_gpu_mb,
                "peakCpuMemoryMB": peak_memory_cpu_mb,
            }
            self._emit(BENCHMARK_EVENT, attributes)
        except Exception:
            return

    def log_model_load(
        self,
        model_name: str = "",
        model_type: str = "",
        execution_provider: str = "",
        total_load_time_ms: float = 0.0,
        num_sessions: int = 0,
        model_file_size_bytes: int = 0,
        session_id: int | None = None,
    ) -> None:
        """Log a model loading telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            session_id = session_id if session_id is not None else self.allocate_model_session_id()
            attributes = {
                "sessionId": session_id,
                "modelName": _redact_paths(model_name),
                "modelType": model_type,
                "executionProvider": execution_provider,
                "totalLoadTimeMs": total_load_time_ms,
                "numSessions": num_sessions,
                "modelFileSizeBytes": model_file_size_bytes,
            }
            self._emit(MODEL_LOAD_EVENT, attributes)
        except Exception:
            return

    def log_inference(
        self,
        model_name: str = "",
        model_type: str = "",
        execution_provider: str = "",
        time_to_first_token_ms: float = 0.0,
        total_generation_time_ms: float = 0.0,
        total_tokens_generated: int = 0,
        input_token_count: int = 0,
        memory_used_mb: float = 0.0,
        gpu_memory_used_mb: float = 0.0,
        session_id: int | None = None,
    ) -> None:
        """Log an inference telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            session_id = session_id if session_id is not None else self.allocate_model_session_id()
            attributes = {
                "sessionId": session_id,
                "modelName": _redact_paths(model_name),
                "modelType": model_type,
                "executionProvider": execution_provider,
                "timeToFirstTokenMs": time_to_first_token_ms,
                "totalGenerationTimeMs": total_generation_time_ms,
                "totalTokensGenerated": total_tokens_generated,
                "inputTokenCount": input_token_count,
                "memoryUsedMB": memory_used_mb,
                "gpuMemoryUsedMB": gpu_memory_used_mb,
            }
            self._emit(INFERENCE_EVENT, attributes)
        except Exception:
            return

    def log_error(
        self,
        exception_type: str,
        exception_message: str,
        action: str = "",
        model_name: str = "",
        execution_provider: str = "",
        session_id: int | None = None,
    ) -> None:
        """Log an error/crash telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            attributes = {
                "exceptionType": exception_type,
                "exceptionMessage": _redact_error_message(exception_message),
                "action": action,
                "modelName": _redact_paths(model_name),
                "executionProvider": execution_provider,
            }
            if session_id is not None:
                attributes["sessionId"] = session_id
            self._emit(ERROR_EVENT, attributes)
        except Exception:
            return

    def disable_telemetry(self) -> None:
        """Disable telemetry irreversibly for the remainder of this process."""
        with self._lock:
            type(self)._process_disabled = True
            self._telemetry_disabled = True
            self._enabled = False
            if self._uploader is not None:
                # Signal the daemon thread to wind down without joining, so opting
                # out never blocks the caller. The thread releases the drain lock on
                # exit; an in-flight send may finish, while remaining queued events
                # stay on disk for the next run.
                self._uploader.signal_stop()
                if self._uploader.stop_loop(0):
                    self._uploader.close()
                    self._uploader = None

    @classmethod
    def _before_fork(cls) -> None:
        """Close SQLite in the parent so the child never inherits a live handle."""
        instance = cls._instance
        store = getattr(instance, "_store", None) if instance is not None else None
        if store is not None:
            with suppress(Exception):
                store.prepare_for_fork()

    @classmethod
    def _after_fork_child(cls) -> None:
        """Discard parent process resources inherited by a forked child."""
        instance = cls._instance
        telemetry_disabled = bool(instance is not None and getattr(instance, "_telemetry_disabled", False))
        cls._lock = threading.RLock()
        cls._instance = None
        if instance is None:
            return
        uploader = getattr(instance, "_uploader", None)
        store = getattr(instance, "_store", None)
        if uploader is not None:
            with suppress(Exception):
                uploader.discard_after_fork()
        if store is not None:
            with suppress(Exception):
                store.discard_after_fork()
        instance._enabled = False
        instance._initialized = False
        instance._uploader = None
        instance._store = None
        instance._heartbeat_thread = None
        if telemetry_disabled:
            cls._instance = instance

    def shutdown(self, flush_seconds: float = 5.0) -> None:
        """Best-effort shutdown within one overall time budget.

        Durability does not depend on this: any events not delivered remain in
        the on-disk store and are uploaded on the next run.
        """
        deadline = time.monotonic() + max(0.0, flush_seconds)

        def remaining_seconds() -> float:
            return max(0.0, deadline - time.monotonic())

        heartbeat_stopped = True
        if self._heartbeat_thread is not None and self._heartbeat_thread is not threading.current_thread():
            if self._heartbeat_thread.ident is not None:
                self._heartbeat_thread.join(remaining_seconds())
            heartbeat_stopped = not self._heartbeat_thread.is_alive()
            if heartbeat_stopped:
                self._heartbeat_thread = None

        uploader_stopped = True
        if self._uploader is not None:
            try:
                # Quiesce the background drainer first. Only drain synchronously
                # AND release the single-drainer lock if the thread actually
                # stopped. If it is still mid-send, leave the lock with the live
                # daemon thread (it releases on wind-down / at process exit);
                # force-releasing here would let another drainer re-send the
                # batch the thread is still processing.
                uploader_stopped = self._uploader.stop_loop(remaining_seconds())
                if uploader_stopped:
                    try:
                        self._uploader.flush(remaining_seconds())
                    finally:
                        self._uploader.close()
                        self._uploader = None
            except Exception:
                uploader_stopped = False
        if self._store is not None and uploader_stopped and heartbeat_stopped:
            self._store.close()
            self._store = None
        if self._heartbeat_thread is None and self._uploader is None and self._store is None:
            self._initialized = False


# Module-level convenience functions
def _get_telemetry() -> GenAITelemetry:
    """Get the telemetry singleton."""
    return GenAITelemetry()


def disable_telemetry() -> None:
    """Disable GenAI telemetry for the remainder of this process."""
    with GenAITelemetry._lock:
        GenAITelemetry._process_disabled = True
        instance = GenAITelemetry._instance
        if instance is not None:
            instance.disable_telemetry()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=GenAITelemetry._before_fork,
        after_in_child=GenAITelemetry._after_fork_child,
    )

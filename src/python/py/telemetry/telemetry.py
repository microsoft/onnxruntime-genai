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

import base64
import os
import platform
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .constants import CONNECTION_STRING
from .deviceid import get_encrypted_device_id_and_status, get_telemetry_base_dir
from .library.event_source import event_source
from .library.options import OneCollectorExporterOptions
from .library.serialization import CommonSchemaJsonSerializationHelper
from .offline_store import OfflineEventStore
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

# CI environment variables that auto-disable telemetry
_CI_ENV_VARS = {"CI", "TF_BUILD", "GITHUB_ACTIONS", "JENKINS_URL", "TRAVIS", "CIRCLECI", "GITLAB_CI", "BUILD_ID"}


def _is_ci_environment() -> bool:
    return any(os.environ.get(var) for var in _CI_ENV_VARS)


def _get_app_version() -> str:
    """Resolve the onnxruntime-genai version.

    Tries three sources in order:
    1. The installed onnxruntime_genai package (__version__ attribute)
    2. importlib.metadata (works even when native ext isn't loadable)
    3. VERSION_INFO file at the repo root
    """
    # 1. Try the package attribute (fastest when native ext is loaded)
    try:
        import onnxruntime_genai
        v = getattr(onnxruntime_genai, "__version__", None)
        if v:
            return v
    except ImportError:
        pass

    # 2. Try importlib.metadata (works for pip-installed packages)
    try:
        from importlib.metadata import version as pkg_version
        return pkg_version("onnxruntime-genai")
    except Exception:
        pass

    # 3. Fall back to VERSION_INFO file at repository root
    try:
        # Walk up from this file to find VERSION_INFO
        import pathlib
        d = pathlib.Path(__file__).resolve().parent
        for _ in range(10):
            candidate = d / "VERSION_INFO"
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip()
            d = d.parent
    except Exception:
        pass

    return "unknown"


def _redact_paths(text: str) -> str:
    """Replace absolute filesystem paths with their basename (drops usernames)."""
    import re

    # Windows drive paths (C:\Users\me\x) and POSIX absolute paths (/home/me/x).
    pattern = re.compile(r"(?:[A-Za-z]:\\[^\s\"']+)|(?:/[^\s\"':]+(?:/[^\s\"':]+)+)")

    def _basename(match: "re.Match") -> str:
        raw = match.group(0)
        return raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] or raw

    return pattern.sub(_basename, text)


def _format_exception_message(ex: BaseException, tb=None) -> str:
    """Format an exception and trim local paths for privacy."""
    formatted = traceback.format_exception(type(ex), ex, tb, limit=5)
    lines = []
    for line in formatted:
        line_trunc = line.strip()
        # Trim paths to relative paths within the package
        if line_trunc.startswith('File "') and "onnxruntime_genai" in line_trunc:
            idx = line_trunc.find("onnxruntime_genai")
            if idx != -1:
                line_trunc = line_trunc[idx:]
        elif line_trunc.startswith('File "'):
            idx = line_trunc[len('File "') :].find('"')
            line_trunc = line_trunc[idx + len('File "') :]
        else:
            # Exception message / other lines: redact any absolute paths.
            line_trunc = _redact_paths(line_trunc)
        lines.append(line_trunc)
    return "\n".join(lines)


class GenAITelemetry:
    """Singleton telemetry manager for ONNX Runtime GenAI.

    Thread-safe singleton that sends telemetry to Microsoft OneCollector.
    Auto-disabled in CI environments. Opt-out via ORTGENAI_DISABLE_TELEMETRY=1.
    """

    _instance: Optional["GenAITelemetry"] = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
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
            self._app_instance_id = uuid.uuid4().hex
            self._app_name = "onnxruntime-genai"

            # Check opt-out conditions
            if os.environ.get("ORTGENAI_DISABLE_TELEMETRY") == "1":
                self._enabled = False
            elif _is_ci_environment():
                self._enabled = False

            if not self._enabled:
                return

            try:
                self._app_version = _get_app_version()
                connection_string = base64.b64decode(CONNECTION_STRING).decode()
                options = OneCollectorExporterOptions(connection_string=connection_string)
                options.validate()
                self._instrumentation_key = options.instrumentation_key
                self._envelope_ikey = (
                    f"{CommonSchemaJsonSerializationHelper.ONE_COLLECTOR_TENANCY_SYMBOL}:{options.tenant_token}"
                )

                event_source.disable()

                # Durable on-disk queue: events survive process exit, so no
                # exit-time flush is required.
                db_path = os.path.join(get_telemetry_base_dir(), "genai_telemetry.db")
                self._store = OfflineEventStore(db_path)

                # The uploader drains the queue (including any rows left by a
                # previous run) in the background.
                self._uploader = EventUploader(
                    self._store, instrumentation_key=self._instrumentation_key
                )
                self._uploader.start()

                # The heartbeat collects system info via blocking subprocesses
                # (nvidia-smi/wmic/sysctl); run it off the host thread so the
                # first telemetry call never stalls the caller.
                threading.Thread(
                    target=self._log_heartbeat, name="genai-telemetry-heartbeat", daemon=True
                ).start()
            except Exception:
                self._store = None
                self._uploader = None
                self._enabled = False

    def _emit(self, event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Serialize an event to a Common Schema envelope and persist it durably."""
        if not self._enabled or self._store is None:
            return
        try:
            data = {
                "app_name": self._app_name,
                "app_version": self._app_version,
                "app_instance_id": self._app_instance_id,
            }
            if attributes:
                data.update(attributes)
            envelope = CommonSchemaJsonSerializationHelper.create_event_envelope(
                event_name=event_name,
                timestamp=datetime.now(timezone.utc),
                ikey=self._envelope_ikey,
                data=data,
            )
            payload = CommonSchemaJsonSerializationHelper.serialize_to_json_bytes(envelope)
            self._store.store(payload)
            if self._uploader is not None:
                self._uploader.request_drain()
        except Exception:
            pass

    def _log_heartbeat(self) -> None:
        """Log initial heartbeat with system info for MAD/DAD tracking."""
        if not self._enabled or self._store is None:
            return

        try:
            device_id, id_status = get_encrypted_device_id_and_status()
            sys_info = get_system_info()
            ep_info = get_execution_provider_info()

            attributes = {
                "device_id": device_id,
                "device_id_status": id_status.value,
                "os": sys_info.get("os", ""),
                "os_version": sys_info.get("os_version", ""),
                "os_release": sys_info.get("os_release", ""),
                "os_arch": sys_info.get("os_arch", ""),
                "processor_count": sys_info.get("processor_count", 0),
                "cpu_model": sys_info.get("cpu_model", ""),
                "total_memory_mb": sys_info.get("total_memory_mb", 0),
                "gpu_name": sys_info.get("gpu_name", ""),
                "gpu_driver_version": sys_info.get("gpu_driver_version", ""),
                "gpu_memory_mb": sys_info.get("gpu_memory_mb", 0),
                "gpu_count": sys_info.get("gpu_count", 0),
                "device_manufacturer": sys_info.get("device_manufacturer", ""),
                "device_model": sys_info.get("device_model", ""),
                "python_version": sys_info.get("python_version", ""),
                "ort_version": sys_info.get("ort_version", ""),
                "user_locale": sys_info.get("user_locale", ""),
                "user_timezone": sys_info.get("user_timezone", ""),
                "process_name": sys_info.get("process_name", ""),
                "available_providers": ",".join(ep_info.get("available_providers", [])),
            }
            self._emit(HEARTBEAT_EVENT, attributes)
        except Exception:
            pass

    def log(self, event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Log a generic telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            self._emit(event_name, attributes)
        except Exception:
            pass

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
        extra_options: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a ModelBuilder telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            attributes = {
                "action": action,
                "duration_ms": duration_ms,
                "success": success,
                "model_name": model_name,
                "model_type": model_type,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_attn_heads": num_attn_heads,
                "num_kv_heads": num_kv_heads,
                "vocab_size": vocab_size,
                "context_length": context_length,
                "io_dtype": io_dtype,
                "quant_type": quant_type,
                "execution_provider": execution_provider,
                "output_model_size_bytes": output_model_size_bytes,
                "num_onnx_operators": num_onnx_operators,
                "operator_types": operator_types,
                "has_custom_ops": has_custom_ops,
                "source_format": source_format,
                "has_adapter": has_adapter,
            }
            if extra_options:
                attributes["extra_options"] = str(extra_options)
            self._emit(MODEL_BUILD_EVENT, attributes)
        except Exception:
            pass

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
    ) -> None:
        """Log a benchmark telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            attributes = {
                "model_name": model_name,
                "precision": precision,
                "backend": backend,
                "device": device,
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "tokens_generated": tokens_generated,
                "tokenization_latency_ms": tokenization_latency_ms,
                "tokenization_throughput": tokenization_throughput,
                "prompt_processing_latency_ms": prompt_processing_latency_ms,
                "prompt_processing_throughput": prompt_processing_throughput,
                "token_generation_latency_ms": token_generation_latency_ms,
                "token_generation_throughput": token_generation_throughput,
                "sampling_latency_ms": sampling_latency_ms,
                "sampling_throughput": sampling_throughput,
                "wall_clock_time_ms": wall_clock_time_ms,
                "wall_clock_throughput": wall_clock_throughput,
                "time_to_first_token_ms": time_to_first_token_ms,
                "peak_memory_gpu_mb": peak_memory_gpu_mb,
                "peak_memory_cpu_mb": peak_memory_cpu_mb,
            }
            self._emit(BENCHMARK_EVENT, attributes)
        except Exception:
            pass

    def log_model_load(
        self,
        model_name: str = "",
        model_type: str = "",
        execution_provider: str = "",
        total_load_time_ms: float = 0.0,
        num_sessions: int = 0,
        model_file_size_bytes: int = 0,
    ) -> None:
        """Log a model loading telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            attributes = {
                "model_name": model_name,
                "model_type": model_type,
                "execution_provider": execution_provider,
                "total_load_time_ms": total_load_time_ms,
                "num_sessions": num_sessions,
                "model_file_size_bytes": model_file_size_bytes,
            }
            self._emit(MODEL_LOAD_EVENT, attributes)
        except Exception:
            pass

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
    ) -> None:
        """Log an inference telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            attributes = {
                "model_name": model_name,
                "model_type": model_type,
                "execution_provider": execution_provider,
                "time_to_first_token_ms": time_to_first_token_ms,
                "total_generation_time_ms": total_generation_time_ms,
                "total_tokens_generated": total_tokens_generated,
                "input_token_count": input_token_count,
                "memory_used_mb": memory_used_mb,
                "gpu_memory_used_mb": gpu_memory_used_mb,
            }
            self._emit(INFERENCE_EVENT, attributes)
        except Exception:
            pass

    def log_error(
        self,
        exception_type: str,
        exception_message: str,
        action: str = "",
        model_name: str = "",
        execution_provider: str = "",
    ) -> None:
        """Log an error/crash telemetry event."""
        if not self._enabled or self._store is None:
            return
        try:
            attributes = {
                "exception_type": exception_type,
                "exception_message": exception_message,
                "action": action,
                "model_name": model_name,
                "execution_provider": execution_provider,
            }
            self._emit(ERROR_EVENT, attributes)
        except Exception:
            pass

    def disable_telemetry(self) -> None:
        """Disable telemetry and stop the background uploader (non-blocking)."""
        self._enabled = False
        if self._uploader is not None:
            # Signal the daemon thread to wind down without joining, so opting
            # out never blocks the caller. The thread releases the drain lock on
            # exit; any already-stored events go out on the next run.
            self._uploader.signal_stop()
            self._uploader = None

    def enable_telemetry(self) -> None:
        """Enable telemetry (restarts the uploader if it was stopped)."""
        self._enabled = True
        if self._store is not None and self._uploader is None:
            try:
                self._uploader = EventUploader(
                    self._store, instrumentation_key=self._instrumentation_key
                )
                self._uploader.start()
            except Exception:
                self._uploader = None

    def shutdown(self, flush_seconds: float = 5.0) -> None:
        """Best-effort flush of pending events, then stop the uploader.

        Durability does not depend on this: any events not delivered remain in
        the on-disk store and are uploaded on the next run. Process exit is never
        blocked because the uploader runs on a daemon thread.
        """
        if self._uploader is not None:
            try:
                # Quiesce the background drainer first. Only drain synchronously
                # if it actually stopped, so we never double-send rows a still
                # in-flight send is processing.
                if self._uploader.stop_loop():
                    self._uploader.flush(flush_seconds)
            except Exception:
                pass
            finally:
                self._uploader.close()
                self._uploader = None
        if self._store is not None:
            self._store.close()


# Module-level convenience functions
def _get_telemetry() -> GenAITelemetry:
    """Get the telemetry singleton."""
    return GenAITelemetry()


def disable_telemetry() -> None:
    """Disable GenAI telemetry."""
    _get_telemetry().disable_telemetry()


def enable_telemetry() -> None:
    """Enable GenAI telemetry."""
    _get_telemetry().enable_telemetry()

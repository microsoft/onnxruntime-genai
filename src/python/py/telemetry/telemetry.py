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
import re
import threading
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .constants import CONNECTION_STRING
from .deviceid import get_encrypted_device_id_and_status, get_telemetry_base_dir
from .library.event_source import event_source
from .library.options import CompressionType, OneCollectorExporterOptions, OneCollectorTransportOptions
from .library.serialization import CommonSchemaJsonSerializationHelper
from .library.transport import HttpJsonPostTransport
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
    """Redact path-bearing tails without leaking space-containing user names."""
    pattern = re.compile(
        r"(?:[A-Za-z]:[\\/])"
        r"|(?:\\\\)"
        r"|(?:~[\\/])"
        r"|(?:(?<![:/])/(?:[^/\r\n]+/))"
        r"|(?:(?<![\\/\w])(?:[A-Za-z0-9_.-]+\\)[^\\/\r\n]+\\)",
        re.IGNORECASE,
    )
    redacted = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        match = pattern.search(body)
        redacted.append(body[: match.start()] + "<path>" + ending if match else line)
    return "".join(redacted)


def _format_exception_message(ex: BaseException, tb=None) -> str:
    """Format an exception and strip local paths for privacy.

    Each entry from ``traceback.format_exception`` is a multi-line string (the
    ``File "..."`` line plus the offending source line), so we process every
    physical line: filenames are trimmed to a package-relative form, and any
    absolute path that remains on a source or message line is redacted so a
    username embedded in it cannot leak.
    """
    formatted = traceback.format_exception(type(ex), ex, tb, limit=5)
    lines = []
    for chunk in formatted:
        for raw_line in chunk.splitlines():
            line_trunc = raw_line.strip()
            # Trim file paths to a relative path within the package.
            if line_trunc.startswith('File "') and "onnxruntime_genai" in line_trunc:
                idx = line_trunc.find("onnxruntime_genai")
                if idx != -1:
                    line_trunc = line_trunc[idx:]
            elif line_trunc.startswith('File "'):
                idx = line_trunc[len('File "') :].find('"')
                line_trunc = line_trunc[idx + len('File "') :]
            # Redact any absolute path that remains (source lines, message, and
            # the tail of File lines).
            line_trunc = _redact_paths(line_trunc)
            lines.append(line_trunc)
    return "\n".join(lines)


class GenAITelemetry:
    """Singleton telemetry manager for ONNX Runtime GenAI.

    Thread-safe singleton that sends telemetry to Microsoft OneCollector.
    Auto-disabled in CI environments. Opt-out via ORT_DISABLE_TELEMETRY=1.
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
            self._heartbeat_thread: Optional[threading.Thread] = None

            # CI / automated-testing: record and send nothing at all — not even
            # the device-id heartbeat — so pipelines don't pollute usage data.
            if _is_ci_environment():
                self._enabled = False
                return

            # User opt-out (ORT_DISABLE_TELEMETRY=1): detailed events are not
            # recorded, but a direct best-effort heartbeat still supports device
            # counting without draining prior detailed events.
            user_opt_out = os.environ.get("ORT_DISABLE_TELEMETRY") == "1"

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

                # Detailed events are recorded only when enabled.
                self._enabled = not user_opt_out

                # Opt-out sends only a direct best-effort heartbeat. Do not open
                # the durable queue, which may contain detailed events from a
                # prior enabled run.
                if user_opt_out:
                    self._heartbeat_thread = threading.Thread(
                        target=self._send_heartbeat,
                        args=(False,),
                        name="genai-telemetry-heartbeat",
                        daemon=True,
                    )
                    self._heartbeat_thread.start()
                    return

                # Durable on-disk queue + uploader. Events survive process exit,
                # so there is no exit-time flush. The uploader retries until
                # delivery.
                db_path = os.path.join(get_telemetry_base_dir(), "genai_telemetry.db")
                self._store = OfflineEventStore(db_path)
                self._uploader = EventUploader(
                    self._store, instrumentation_key=self._instrumentation_key
                )
                self._uploader.start()

                # The heartbeat collects system info via blocking subprocesses;
                # build and enqueue it off the host thread.
                self._heartbeat_thread = threading.Thread(
                    target=self._send_heartbeat,
                    args=(True,),
                    name="genai-telemetry-heartbeat",
                    daemon=True,
                )
                self._heartbeat_thread.start()
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

    def _build_heartbeat_attributes(self) -> dict[str, Any]:
        """Collect device-id + system info for the heartbeat event."""
        device_id, id_status = get_encrypted_device_id_and_status()
        sys_info = get_system_info()
        ep_info = get_execution_provider_info()
        return {
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
            "process_name": sys_info.get("process_name", ""),
            "available_providers": ",".join(ep_info.get("available_providers", [])),
        }

    def _send_heartbeat(self, durable: bool = True) -> None:
        """Send the heartbeat durably when enabled and directly on opt-out."""
        if durable and self._store is None:
            return
        try:
            data = {
                "app_name": self._app_name,
                "app_version": self._app_version,
                "app_instance_id": self._app_instance_id,
            }
            data.update(self._build_heartbeat_attributes())
            envelope = CommonSchemaJsonSerializationHelper.create_event_envelope(
                event_name=HEARTBEAT_EVENT,
                timestamp=datetime.now(timezone.utc),
                ikey=self._envelope_ikey,
                data=data,
            )
            payload = CommonSchemaJsonSerializationHelper.serialize_to_json_bytes(envelope)
            if durable:
                self._store.store(payload)
                if self._uploader is not None:
                    self._uploader.request_drain()
            else:
                transport = HttpJsonPostTransport(
                    endpoint=OneCollectorTransportOptions.DEFAULT_ENDPOINT,
                    ikey=self._instrumentation_key,
                    compression=CompressionType.DEFLATE,
                )
                transport.send(payload, timeout_sec=2.0, item_count=1)
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
        """Log benchmark telemetry; prompt/tokenization latency fields are total milliseconds per prompt."""
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
                "exception_message": _redact_paths(exception_message),
                "action": action,
                "model_name": model_name,
                "execution_provider": execution_provider,
            }
            self._emit(ERROR_EVENT, attributes)
        except Exception:
            pass

    def disable_telemetry(self) -> None:
        """Disable detailed telemetry and stop the uploader (non-blocking)."""
        with self._lock:
            self._enabled = False
            if self._uploader is not None:
                # Signal the daemon thread to wind down without joining, so opting
                # out never blocks the caller. The thread releases the drain lock on
                # exit; any already-stored events go out on the next run.
                self._uploader.signal_stop()

    def enable_telemetry(self) -> None:
        """Enable telemetry (creates/restarts the uploader if needed).

        Has no effect when telemetry was hard-disabled for CI/testing, in which
        case no instrumentation key was ever resolved.
        """
        with self._lock:
            if not self._instrumentation_key:
                return
            # The environment opt-out / CI is the user's master switch: a
            # programmatic enable must not override ORT_DISABLE_TELEMETRY.
            if os.environ.get("ORT_DISABLE_TELEMETRY") == "1" or _is_ci_environment():
                return
            if self._uploader is not None:
                old_uploader = self._uploader
                old_uploader.signal_stop()
                if old_uploader.stop_loop(0):
                    old_uploader.close()
                # If a send is still in flight, the old uploader retains the
                # process lock until it exits. The replacement waits on that
                # lock, so rows cannot be double-sent.
                self._uploader = None
            if self._store is None:
                try:
                    db_path = os.path.join(get_telemetry_base_dir(), "genai_telemetry.db")
                    self._store = OfflineEventStore(db_path)
                except Exception:
                    self._store = None
                    return
            try:
                self._uploader = EventUploader(
                    self._store, instrumentation_key=self._instrumentation_key
                )
                self._uploader.start()
                self._enabled = True
            except Exception:
                self._uploader = None
                self._enabled = False

    def shutdown(self, flush_seconds: float = 5.0) -> None:
        """Best-effort flush of pending events, then stop the uploader.

        Durability does not depend on this: any events not delivered remain in
        the on-disk store and are uploaded on the next run. Process exit is never
        blocked because the uploader runs on a daemon thread.
        """
        heartbeat_stopped = True
        if self._heartbeat_thread is not None and self._heartbeat_thread is not threading.current_thread():
            self._heartbeat_thread.join(max(0.0, flush_seconds))
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
                uploader_stopped = self._uploader.stop_loop(max(0.0, flush_seconds))
                if uploader_stopped:
                    try:
                        self._uploader.flush(flush_seconds)
                    finally:
                        self._uploader.close()
                        self._uploader = None
            except Exception:
                uploader_stopped = False
        if self._store is not None and uploader_stopped and heartbeat_stopped:
            self._store.close()
            self._store = None


# Module-level convenience functions
def _get_telemetry() -> GenAITelemetry:
    """Get the telemetry singleton."""
    return GenAITelemetry()


def disable_telemetry() -> None:
    """Disable detailed GenAI telemetry; the opt-out heartbeat remains enabled outside CI."""
    _get_telemetry().disable_telemetry()


def enable_telemetry() -> None:
    """Enable GenAI telemetry."""
    _get_telemetry().enable_telemetry()

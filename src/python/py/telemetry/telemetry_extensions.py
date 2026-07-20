# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Telemetry decorators and context managers for GenAI instrumentation."""

import functools
import inspect
import time
from types import TracebackType
from typing import Any, Callable, Optional, TypeVar

from .telemetry import (
    ACTION_EVENT,
    ERROR_EVENT,
    GenAITelemetry,
    _format_exception_message,
    _redact_paths,
)

_TFunc = TypeVar("_TFunc", bound=Callable[..., Any])


def _get_telemetry() -> GenAITelemetry:
    return GenAITelemetry()


def log_action(
    invoked_from: str,
    action_name: str,
    duration_ms: float,
    success: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log a telemetry action event."""
    telemetry = _get_telemetry()
    attributes = dict(metadata or {})
    attributes.update({
        "invoked_from": invoked_from,
        "action_name": action_name,
        "duration_ms": duration_ms,
        "success": success,
    })
    telemetry.log(ACTION_EVENT, attributes)


def log_error(
    exception_type: str,
    exception_message: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log a telemetry error event."""
    telemetry = _get_telemetry()
    attributes = dict(metadata or {})
    attributes.update({
        "exception_type": exception_type,
        "exception_message": _redact_paths(exception_message),
    })
    telemetry.log(ERROR_EVENT, attributes)


def _resolve_invoked_from(skip_frames: int = 0) -> str:
    """Resolve how GenAI was invoked by examining the call stack."""
    for frame_info in inspect.stack()[2 + skip_frames :]:
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            continue
        module_name = module.__name__
        if module_name.startswith("onnxruntime_genai."):
            continue
        if module_name == "__main__":
            return "Script"
        return module_name
    return "Interactive"


class ActionContext:
    """Context manager for recording telemetry around a block of work.

    Usage:
        with ActionContext("load_model") as ctx:
            ctx.add_metadata("model_type", "llama")
            do_work()
    """

    def __init__(
        self,
        action_name: str,
        invoked_from: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.action_name = action_name
        self.invoked_from = invoked_from if invoked_from is not None else _resolve_invoked_from()
        self.metadata = metadata or {}
        self._start_time: Optional[float] = None

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def __enter__(self) -> "ActionContext":
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        duration_ms = int((time.perf_counter() - (self._start_time or time.perf_counter())) * 1000)
        success = exc_type is None

        log_action(
            invoked_from=self.invoked_from,
            action_name=self.action_name,
            duration_ms=duration_ms,
            success=success,
            metadata=self.metadata,
        )

        if exc_type is not None and exc_val is not None:
            log_error(
                exception_type=exc_type.__name__,
                exception_message=_format_exception_message(exc_val, exc_tb),
                metadata=self.metadata,
            )

        return False


def action(func: _TFunc) -> _TFunc:
    """Decorator to record telemetry around a function call.

    Automatically logs action duration and success/failure. On exception,
    also logs an error event with the exception details.

    Usage:
        @action
        def create_model(...):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        # Resolve telemetry context defensively: instrumentation (including
        # inspect.stack()) must never propagate into the wrapped call.
        try:
            invoked_from = _resolve_invoked_from()
            action_name = func.__name__

            # Try to resolve class name for methods
            if args and hasattr(args[0], "__class__"):
                cls_name = args[0].__class__.__name__
                if cls_name != "type":
                    action_name = f"{cls_name}.{func.__name__}" if func.__name__ != "run" else cls_name
        except Exception:
            invoked_from = "unknown"
            action_name = getattr(func, "__name__", "unknown")

        start_time = time.perf_counter()
        success = True
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            success = False
            log_error(
                exception_type=type(exc).__name__,
                exception_message=_format_exception_message(exc, exc.__traceback__),
            )
            raise
        finally:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_action(
                invoked_from=invoked_from,
                action_name=action_name,
                duration_ms=duration_ms,
                success=success,
            )

    return wrapper  # type: ignore[return-value]

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""ONNX Runtime GenAI Telemetry.

Provides telemetry integration for ONNX Runtime GenAI via Microsoft OneCollector.
Set ORT_DISABLE_TELEMETRY=1 before initialization, or call disable_telemetry(),
to suppress detailed events. One best-effort Heartbeat may still be attempted.
CI and automated-test environments suppress all telemetry.

Usage:
    from onnxruntime_genai.telemetry import GenAITelemetry, action, ActionContext

    telemetry = GenAITelemetry()

    # Decorator pattern
    @action
    def my_function():
        ...

    # Context manager pattern
    with ActionContext("my_operation") as ctx:
        ctx.add_metadata("model", "phi-3")
        ...

    # Direct logging
    telemetry.log_model_build(...)
    telemetry.log_benchmark(...)

    # Shutdown
    telemetry.shutdown()
"""

from .telemetry import GenAITelemetry, disable_telemetry
from .telemetry_extensions import ActionContext, action, log_action, log_error

__all__ = [
    "ActionContext",
    "GenAITelemetry",
    "action",
    "disable_telemetry",
    "log_action",
    "log_error",
]

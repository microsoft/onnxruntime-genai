# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Optional telemetry integration for the standalone model builder."""

import importlib
import logging
import os
import sys
import time
from typing import Any


class ModelBuilderTelemetry:
    def __init__(self):
        self.start = time.perf_counter()
        self.get_telemetry()

    def load_telemetry_module(self, name=""):
        suffix = f".{name}" if name else ""
        try:
            return importlib.import_module(f"onnxruntime_genai.telemetry{suffix}")
        except Exception:
            # Source usage must not require the native onnxruntime_genai package.
            telemetry_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            path_added = telemetry_root not in sys.path
            if path_added:
                sys.path.insert(0, telemetry_root)
            try:
                return importlib.import_module(f"telemetry{suffix}")
            finally:
                if path_added and telemetry_root in sys.path:
                    sys.path.remove(telemetry_root)

    def get_telemetry(self):
        try:
            return self.load_telemetry_module().GenAITelemetry()
        except Exception:
            logging.getLogger(__name__).debug("Model-builder telemetry is unavailable", exc_info=True)
            return None

    def shutdown(self, max_seconds: float = 1.0) -> None:
        """Best-effort bounded delivery for the model-builder CLI."""
        try:
            telemetry = self.get_telemetry()
            if telemetry is not None:
                telemetry.shutdown(max_seconds)
        except Exception:
            logging.getLogger(__name__).debug("Model-builder telemetry shutdown failed", exc_info=True)

    def sanitize_extra_options(self, extra_options: dict[str, Any]) -> dict[str, Any]:
        """Exclude authentication/internal Hugging Face state and scrub user-facing options."""
        path_utils = self.load_telemetry_module("path_utils")
        return {
            key: path_utils.scrub_value_for_telemetry(value)
            for key, value in extra_options.items()
            if key not in {"hf_token", "hf_details"}
        }

    def emit(
        self,
        config,
        onnx_model,
        precision: str,
        execution_provider: str,
        output_dir: str,
        extra_options: dict[str, Any],
        input_path: str,
        model_name: str,
    ) -> None:
        try:
            duration_ms = (time.perf_counter() - self.start) * 1000
            telemetry = self.get_telemetry()
            if telemetry is None or not telemetry.accepts_detailed_events:
                return
            path_utils = self.load_telemetry_module("path_utils")

            model_type = getattr(onnx_model, "model_type", getattr(config, "model_type", ""))
            hidden_size = getattr(config, "hidden_size", 0)
            num_layers = getattr(config, "num_hidden_layers", 0)
            num_attn_heads = getattr(config, "num_attention_heads", 0)
            num_kv_heads = getattr(config, "num_key_value_heads", num_attn_heads)
            vocab_size = getattr(config, "vocab_size", 0)
            context_length = getattr(config, "max_position_embeddings", 0)

            output_model_size = 0
            if os.path.isdir(output_dir):
                for filename in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, filename)
                    if os.path.isfile(file_path) and filename.endswith((".onnx", ".onnx_data", ".onnx.data")):
                        output_model_size += os.path.getsize(file_path)

            num_ops = 0
            op_types = ""
            has_custom_ops = False
            # Saving can quantize a copy, so these counts may differ from the exported graph.
            if hasattr(onnx_model, "model") and onnx_model.model is not None:
                try:
                    graph = onnx_model.model.graph
                    if graph is not None:
                        op_type_set = set()
                        for node in graph:
                            num_ops += 1
                            op_type_set.add(node.op_type)
                            if node.domain and not node.domain.startswith("ai.onnx"):
                                has_custom_ops = True
                        op_types = ",".join(sorted(op_type_set))
                except Exception:
                    logging.getLogger(__name__).debug("Model-builder graph telemetry is unavailable", exc_info=True)

            io_dtype = str(getattr(onnx_model, "io_dtype", "")).replace("DataType.", "")
            quant_type = str(getattr(onnx_model, "onnx_dtype", precision)).replace("DataType.", "")

            telemetry.log_model_build(
                action="create_model",
                duration_ms=duration_ms,
                success=True,
                model_name=path_utils.sanitize_model_identifier(getattr(config, "_name_or_path", "") or model_name),
                model_type=str(model_type),
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attn_heads=num_attn_heads,
                num_kv_heads=num_kv_heads,
                vocab_size=vocab_size,
                context_length=context_length,
                io_dtype=io_dtype,
                quant_type=quant_type,
                execution_provider=path_utils.normalize_execution_provider(execution_provider),
                output_model_size_bytes=output_model_size,
                num_onnx_operators=num_ops,
                operator_types=op_types,
                has_custom_ops=has_custom_ops,
                source_format="gguf" if input_path and input_path.lower().endswith(".gguf") else "huggingface",
                has_adapter="adapter_path" in extra_options,
                extra_options=self.sanitize_extra_options(extra_options),
            )
        except Exception:
            logging.getLogger(__name__).debug("Model-builder telemetry emission failed", exc_info=True)

import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"


def _load_builder_entrypoint_module():
    builders_stub = types.ModuleType("builders")
    builders_stub.__file__ = str(MODELS_DIR / "builders" / "__init__.py")

    def _getattr(name):
        return type(name, (), {})

    builders_stub.__getattr__ = _getattr
    previous_builders = sys.modules.get("builders")
    had_previous_builders = "builders" in sys.modules
    try:
        sys.modules["builders"] = builders_stub
        spec = importlib.util.spec_from_file_location("models_builder_telemetry", MODELS_DIR / "builder.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if had_previous_builders:
            sys.modules["builders"] = previous_builders
        else:
            sys.modules.pop("builders", None)


builder_module = _load_builder_entrypoint_module()


@pytest.mark.parametrize(
    "value, expected",
    [
        (r"C:\Users\alice\models\model.onnx", "model.onnx"),
        (r"\\server\share\models\model.onnx", "model.onnx"),
        ("/home/alice/models/model.onnx", "model.onnx"),
        ("~/private/model.onnx", "model.onnx"),
        (r"..\private\model.onnx", "model.onnx"),
        ("../private/model.onnx", "model.onnx"),
        ("/", "<path>"),
        ("microsoft/phi-3-mini", "microsoft/phi-3-mini"),
    ],
)
def test_sanitize_path_value_is_platform_independent(value, expected):
    assert builder_module._sanitize_path_value(value) == expected


def test_telemetry_fallback_restores_source_path(monkeypatch):
    telemetry_stub = types.ModuleType("telemetry")

    class DisabledTelemetry:
        accepts_detailed_events = False

    telemetry_stub.GenAITelemetry = DisabledTelemetry
    source_root = str(MODELS_DIR.parent)
    source_index = sys.path.index(source_root) if source_root in sys.path else None
    if source_index is not None:
        sys.path.pop(source_index)
    try:
        before = list(sys.path)
        monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)
        monkeypatch.setitem(sys.modules, "onnxruntime_genai.telemetry", None)
        monkeypatch.setitem(sys.modules, "telemetry", telemetry_stub)
        builder_module._emit_model_build_telemetry(
            action_name="create_model",
            duration_ms=1.0,
            success=False,
            config=None,
            onnx_model=None,
            precision="fp16",
            execution_provider="cpu",
            output_dir="",
            extra_options={},
        )
        assert sys.path == before
    finally:
        if source_index is not None:
            sys.path.insert(source_index, source_root)


def test_minimal_failure_telemetry_uses_sanitized_fallback_model_name(monkeypatch):
    telemetry_stub = types.ModuleType("telemetry")
    captured = {}

    class RecordingTelemetry:
        accepts_detailed_events = True

        def log_model_build(self, **kwargs):
            captured.update(kwargs)

    telemetry_stub.GenAITelemetry = RecordingTelemetry
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", None)
    monkeypatch.setitem(sys.modules, "onnxruntime_genai.telemetry", None)
    monkeypatch.setitem(sys.modules, "telemetry", telemetry_stub)

    builder_module._emit_model_build_telemetry(
        action_name="create_model",
        duration_ms=1.0,
        success=False,
        config=None,
        onnx_model=None,
        precision="fp16",
        execution_provider="cpu",
        output_dir="",
        extra_options={},
        fallback_model_name=r"C:\Users\alice\models\model.onnx",
    )

    assert captured["model_name"] == "model.onnx"

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
        (r"C:\Users\alice\models\model.onnx", "[path]"),
        (r"\\server\share\models\model.onnx", "[path]"),
        ("/home/alice/models/model.onnx", "[path]"),
        ("~/private/model.onnx", "[path]"),
        (r"..\private\model.onnx", "[path]"),
        ("../private/model.onnx", "[path]"),
        ("/", "[path]"),
        ("invalid\0identifier", "invalid\0identifier"),
        ("microsoft/phi-3-mini", "microsoft/phi-3-mini"),
    ],
)
def test_sanitize_path_value_is_platform_independent(value, expected):
    assert builder_module._sanitize_path_value(value) == expected


def test_telemetry_execution_provider_normalizes_trt_rtx():
    assert builder_module._normalize_execution_provider_name("NvTensorRtRtx") == "trt-rtx"
    assert builder_module._normalize_execution_provider_name("cuda") == "cuda"


def test_telemetry_extra_options_exclude_hugging_face_context():
    sanitized = builder_module._sanitize_extra_options(
        {
            "batch_size": 4,
            "hf_details": {"hf_config": "internal"},
            "hf_token": "secret",
        }
    )

    assert sanitized == {"batch_size": "4"}


def test_cli_parse_failure_emits_model_build_failure(monkeypatch):
    args = types.SimpleNamespace(
        cache_dir="cache",
        disable_telemetry=False,
        execution_provider="cuda",
        extra_options=None,
        input=r"C:\Users\alice\models\model",
        model_name=None,
        output="output",
        precision="fp16",
    )
    captured = {}
    times = iter((10.0, 10.25))

    def fail_parse(*_args):
        raise ValueError("bad")

    monkeypatch.setattr(builder_module, "parse_extra_options", fail_parse)
    monkeypatch.setattr(builder_module, "_emit_model_build_telemetry", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(builder_module.time, "perf_counter", lambda: next(times))

    with pytest.raises(ValueError, match="bad"):
        builder_module._run_from_args(args)

    assert captured["success"] is False
    assert captured["duration_ms"] == 250
    assert captured["fallback_model_name"] == args.input
    assert captured["extra_options"] == {}


def test_cli_model_build_duration_starts_before_parsing(monkeypatch):
    args = types.SimpleNamespace(
        cache_dir="cache",
        disable_telemetry=False,
        execution_provider="cuda",
        extra_options=None,
        input="",
        model_name="microsoft/model",
        output="output",
        precision="fp16",
    )
    captured = {}

    monkeypatch.setattr(builder_module, "parse_extra_options", lambda *_args: {"config_only": True})

    def record_create(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(builder_module, "_create_model_with_telemetry", record_create)
    monkeypatch.setattr(builder_module.time, "perf_counter", lambda: 42.0)

    builder_module._run_from_args(args)

    assert captured["args"][6] == 42.0
    assert captured["kwargs"] == {"config_only": True}


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
        execution_provider="NvTensorRtRtx",
        output_dir="",
        extra_options={},
        fallback_model_name=r"C:\Users\alice\models\model.onnx",
    )

    assert captured["model_name"] == "[path]"
    assert captured["execution_provider"] == "trt-rtx"


def test_minimal_failure_telemetry_accepts_missing_model_identifier(monkeypatch):
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
        fallback_model_name=None,
    )

    assert captured["model_name"] == ""


def test_failed_build_ignores_existing_output_artifacts(monkeypatch, tmp_path):
    telemetry_stub = types.ModuleType("telemetry")
    captured = {}

    class RecordingTelemetry:
        accepts_detailed_events = True

        def log_model_build(self, **kwargs):
            captured.update(kwargs)

    (tmp_path / "stale.onnx").write_bytes(b"stale model")
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
        output_dir=str(tmp_path),
        extra_options={},
    )

    assert captured["output_model_size_bytes"] == 0

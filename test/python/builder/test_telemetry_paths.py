import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models"


def _load_builder_entrypoint_module():
    builders_stub = types.ModuleType("builders")
    builders_stub.__file__ = str(MODELS_DIR / "builders" / "__init__.py")
    builders_stub.__path__ = [str(MODELS_DIR / "builders")]
    builders_stub.__package__ = "builders"

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


def test_extra_options_redact_relative_pathlike_values():
    sanitized = builder_module._sanitize_extra_options(
        {
            "adapter_path": Path("private/adapter"),
            "nested": {"scale_path": Path("private/scales.json")},
            "batch_size": 4,
            "hf_token": "hf-secret",
            "hf_details": {
                "extra_kwargs": {"cache_dir": Path("private/cache")},
                "hf_name": "microsoft/model",
                "hf_config": object(),
            },
        }
    )

    assert "hf_token" not in sanitized
    assert "hf_details" not in sanitized
    assert sanitized["adapter_path"] == "[path]"
    assert sanitized["nested"]["scale_path"] == "[path]"
    assert sanitized["batch_size"] == 4


def test_builder_import_survives_telemetry_import_failure(monkeypatch):
    real_import = builtins.__import__

    def fail_telemetry_import(name, *args, **kwargs):
        if name == "telemetry.path_utils":
            raise OSError(126, "native telemetry dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_telemetry_import)

    module = _load_builder_entrypoint_module()

    assert module._normalize_execution_provider_name("NvTensorRtRtx") == "trt-rtx"
    assert module._sanitize_path_value("/private/model") == "[path]"
    assert module._sanitize_extra_options({"private": "/secret", "batch_size": 4}) == {
        "private": "[redacted]",
        "batch_size": 4,
    }


def test_builder_shutdown_uses_bounded_budget(monkeypatch):
    calls = []

    def shutdown(seconds):
        calls.append(seconds)

    telemetry = types.SimpleNamespace(shutdown=shutdown)
    monkeypatch.setattr(builder_module, "_get_model_builder_telemetry", lambda: telemetry)

    builder_module._shutdown_model_builder_telemetry()

    assert calls == [1.0]


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


def test_pathlike_input_is_normalized_for_success(monkeypatch):
    captured = {}

    def create_impl(*args, **kwargs):
        assert captured["telemetry_initialized"]
        captured["input_path"] = args[1]
        return "created"

    monkeypatch.setattr(
        builder_module,
        "_get_model_builder_telemetry",
        lambda: captured.update(telemetry_initialized=True),
    )
    monkeypatch.setattr(builder_module, "_create_model_impl", create_impl)

    assert (
        builder_module.create_model(
            "model",
            Path("model.gguf"),
            "output",
            "fp16",
            "cpu",
            "cache",
        )
        == "created"
    )
    assert captured["input_path"] == "model.gguf"


def test_pathlike_input_preserves_early_failure_telemetry(monkeypatch):
    captured = {}

    def fail_create(*args, **kwargs):
        raise RuntimeError("early failure")

    monkeypatch.setattr(builder_module, "_create_model_impl", fail_create)
    monkeypatch.setattr(builder_module, "_get_model_builder_telemetry", lambda: None)
    monkeypatch.setattr(
        builder_module,
        "_emit_model_build_telemetry",
        lambda **kwargs: captured.update(kwargs),
    )

    with pytest.raises(RuntimeError, match="early failure"):
        builder_module.create_model(
            "model",
            Path("model.gguf"),
            "output",
            "fp16",
            "cpu",
            "cache",
        )

    assert captured["source_format"] == "gguf"
    assert captured["fallback_model_name"] == "model.gguf"


def test_interrupted_build_is_not_reported_as_success(monkeypatch, tmp_path):
    captured = {}
    config = types.SimpleNamespace(architectures=["LlamaForCausalLM"])

    class InterruptedModel:
        def make_model(self, input_path):
            raise KeyboardInterrupt

    monkeypatch.setattr(builder_module, "set_io_dtype", lambda *args: object())
    monkeypatch.setattr(builder_module, "set_onnx_dtype", lambda *args: object())
    monkeypatch.setattr(builder_module, "LlamaModel", lambda *args: InterruptedModel())
    monkeypatch.setattr(
        builder_module,
        "_emit_model_build_telemetry",
        lambda **kwargs: captured.update(kwargs),
    )
    telemetry_state = {"emitted": False}

    with pytest.raises(KeyboardInterrupt):
        builder_module._create_model_impl(
            "model",
            "",
            str(tmp_path / "output"),
            "fp16",
            "cpu",
            str(tmp_path / "cache"),
            telemetry_state,
            hf_details={
                "extra_kwargs": {},
                "hf_name": "model",
                "hf_config": config,
            },
        )

    assert telemetry_state["emitted"]
    assert captured["success"] is False

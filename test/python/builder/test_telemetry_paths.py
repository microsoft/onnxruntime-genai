import builtins
import importlib.util
import json
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
    models_path_added = str(MODELS_DIR) not in sys.path
    try:
        if models_path_added:
            sys.path.insert(0, str(MODELS_DIR))
        sys.modules["builders"] = builders_stub
        spec = importlib.util.spec_from_file_location("models_builder_telemetry", MODELS_DIR / "builder.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if models_path_added:
            sys.path.remove(str(MODELS_DIR))
        if had_previous_builders:
            sys.modules["builders"] = previous_builders
        else:
            sys.modules.pop("builders", None)


telemetry_spec = importlib.util.spec_from_file_location(
    "model_builder_telemetry", MODELS_DIR / "model_builder_telemetry.py"
)
telemetry_module = importlib.util.module_from_spec(telemetry_spec)
telemetry_spec.loader.exec_module(telemetry_module)


@pytest.fixture
def builder_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "model_builder_telemetry", telemetry_module)
    return _load_builder_entrypoint_module()


@pytest.fixture
def builder_telemetry(monkeypatch):
    monkeypatch.setattr(telemetry_module.ModelBuilderTelemetry, "get_telemetry", lambda self: None)
    return telemetry_module.ModelBuilderTelemetry()


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
def test_sanitize_path_value_is_platform_independent(builder_telemetry, value, expected):
    assert builder_telemetry.load_telemetry_module("path_utils").sanitize_model_identifier(value) == expected


def test_telemetry_execution_provider_normalizes_trt_rtx(builder_telemetry):
    path_utils = builder_telemetry.load_telemetry_module("path_utils")
    assert path_utils.normalize_execution_provider("NvTensorRtRtx") == "trt-rtx"
    assert path_utils.normalize_execution_provider("cuda") == "cuda"


def test_extra_options_redact_relative_pathlike_values(builder_telemetry):
    sanitized = builder_telemetry.sanitize_extra_options(
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
        if name.startswith(("onnxruntime_genai", "telemetry")):
            raise OSError(126, "native telemetry dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_telemetry_import)

    module = _load_builder_entrypoint_module()

    assert module.ModelBuilderTelemetry is not None


def test_optional_telemetry_failure_restores_source_path(monkeypatch, caplog):
    real_import_module = importlib.import_module

    def fail_telemetry_module(name, *args, **kwargs):
        if name.startswith(("onnxruntime_genai", "telemetry")):
            raise OSError(126, "native telemetry dependency unavailable")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fail_telemetry_module)
    before = list(sys.path)
    with caplog.at_level("DEBUG", logger=telemetry_module.__name__):
        telemetry = telemetry_module.ModelBuilderTelemetry()
        assert telemetry.get_telemetry() is None
        telemetry.shutdown()
        telemetry.emit(None, None, "fp16", "cpu", "", {}, "", "model")
    assert sys.path == before
    assert "Model-builder telemetry is unavailable" in caplog.text


def test_builder_shutdown_uses_bounded_budget(monkeypatch, builder_telemetry):
    calls = []

    def shutdown(seconds):
        calls.append(seconds)

    telemetry = types.SimpleNamespace(shutdown=shutdown)
    monkeypatch.setattr(builder_telemetry, "get_telemetry", lambda: telemetry)

    builder_telemetry.shutdown()

    assert calls == [1.0]


def test_telemetry_loading_prefers_the_wheel_package(monkeypatch, builder_telemetry):
    calls = []
    module = types.SimpleNamespace()

    def import_module(name):
        calls.append(name)
        return module

    monkeypatch.setattr(importlib, "import_module", import_module)

    assert builder_telemetry.load_telemetry_module("path_utils") is module
    assert calls == ["onnxruntime_genai.telemetry.path_utils"]


def test_telemetry_fallback_restores_source_path(monkeypatch, builder_telemetry):
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
        assert builder_telemetry.load_telemetry_module().GenAITelemetry is DisabledTelemetry
        assert sys.path == before
    finally:
        if source_index is not None:
            sys.path.insert(source_index, source_root)


@pytest.mark.parametrize("telemetry", [None, types.SimpleNamespace(accepts_detailed_events=False)])
def test_disabled_or_unsampled_telemetry_does_not_collect_build_details(monkeypatch, builder_telemetry, telemetry):
    calls = []
    monkeypatch.setattr(builder_telemetry, "get_telemetry", lambda: telemetry)
    monkeypatch.setattr(builder_telemetry, "load_telemetry_module", calls.append)

    builder_telemetry.emit(None, None, "fp16", "cpu", "", {}, "", "model")

    assert calls == []


def test_build_telemetry_sanitizes_model_name_and_normalizes_provider(monkeypatch, builder_telemetry):
    captured = {}

    class RecordingTelemetry:
        accepts_detailed_events = True

        def log_model_build(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(builder_telemetry, "get_telemetry", RecordingTelemetry)
    onnx_model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            graph=[
                types.SimpleNamespace(op_type="MatMul", domain=""),
                types.SimpleNamespace(op_type="GroupQueryAttention", domain="com.microsoft"),
            ]
        )
    )

    builder_telemetry.emit(
        config=None,
        onnx_model=onnx_model,
        precision="fp16",
        execution_provider="NvTensorRtRtx",
        output_dir="",
        extra_options={},
        input_path="model.gguf",
        model_name=r"C:\Users\alice\models\model.onnx",
    )

    assert captured["model_name"] == "[path]"
    assert captured["execution_provider"] == "trt-rtx"
    assert captured["source_format"] == "gguf"
    assert captured["success"] is True
    assert captured["num_onnx_operators"] == 2
    assert captured["operator_types"] == "GroupQueryAttention,MatMul"
    assert captured["has_custom_ops"] is True
    assert captured["duration_ms"] >= 0


def test_build_telemetry_counts_exported_artifacts(monkeypatch, tmp_path, builder_telemetry):
    captured = {}

    class RecordingTelemetry:
        accepts_detailed_events = True

        def log_model_build(self, **kwargs):
            captured.update(kwargs)

    (tmp_path / "model.onnx").write_bytes(b"model")
    (tmp_path / "model.onnx.data").write_bytes(b"weights")
    monkeypatch.setattr(builder_telemetry, "get_telemetry", RecordingTelemetry)

    builder_telemetry.emit(
        config=None,
        onnx_model=None,
        precision="fp16",
        execution_provider="cpu",
        output_dir=str(tmp_path),
        extra_options={},
        input_path="",
        model_name="model",
    )

    assert captured["output_model_size_bytes"] == 12


def test_telemetry_failures_do_not_interrupt_the_caller(monkeypatch, caplog, builder_telemetry):
    class FailingTelemetry:
        accepts_detailed_events = True

        def log_model_build(self, **kwargs):
            raise RuntimeError("event delivery failed")

        def shutdown(self, seconds):
            raise RuntimeError("shutdown failed")

    monkeypatch.setattr(builder_telemetry, "get_telemetry", FailingTelemetry)
    with caplog.at_level("DEBUG", logger=telemetry_module.__name__):
        builder_telemetry.emit(None, None, "fp16", "cpu", "", {}, "", "model")
        builder_telemetry.shutdown()

    assert "Model-builder telemetry emission failed" in caplog.text
    assert "Model-builder telemetry shutdown failed" in caplog.text


def test_build_duration_covers_the_export(monkeypatch, builder_telemetry):
    captured = {}
    telemetry = types.SimpleNamespace(
        accepts_detailed_events=True,
        log_model_build=lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(builder_telemetry, "get_telemetry", lambda: telemetry)
    builder_telemetry.start = 10.0
    monkeypatch.setattr(telemetry_module.time, "perf_counter", lambda: 12.5)

    builder_telemetry.emit(None, None, "fp16", "cpu", "", {}, "", "model")

    assert captured["duration_ms"] == 2500


def test_builder_cli_retains_the_telemetry_optout(monkeypatch, builder_module):
    monkeypatch.setattr(
        sys,
        "argv",
        ["builder.py", "-m", "model", "-o", "output", "-p", "fp16", "-e", "cpu", "--disable_telemetry"],
    )

    assert builder_module.get_args().disable_telemetry is True


def test_early_failure_does_not_report_a_successful_build(monkeypatch, tmp_path, builder_telemetry, builder_module):
    captured = {}
    monkeypatch.setattr(
        builder_module.ModelBuilderTelemetry,
        "emit",
        lambda self, **kwargs: captured.update(kwargs),
    )

    with pytest.raises(Exception, match="Hugging Face details not found"):
        builder_module.create_model(
            "model",
            Path("model.gguf"),
            str(tmp_path / "output"),
            "fp16",
            "cpu",
            str(tmp_path / "cache"),
        )

    assert captured == {}


def test_interrupted_build_is_not_reported_as_success(monkeypatch, tmp_path, builder_telemetry, builder_module):
    captured = {}
    config = types.SimpleNamespace(architectures=["LlamaForCausalLM"])

    class InterruptedModel:
        def make_model(self, input_path):
            raise KeyboardInterrupt

    monkeypatch.setattr(builder_module, "set_io_dtype", lambda *args: object())
    monkeypatch.setattr(builder_module, "set_onnx_dtype", lambda *args: object())
    monkeypatch.setattr(builder_module, "LlamaModel", lambda *args: InterruptedModel())
    monkeypatch.setattr(
        builder_module.ModelBuilderTelemetry,
        "emit",
        lambda self, **kwargs: captured.update(kwargs),
    )
    with pytest.raises(KeyboardInterrupt):
        builder_module.create_model(
            "model",
            "",
            str(tmp_path / "output"),
            "fp16",
            "cpu",
            str(tmp_path / "cache"),
            hf_details={
                "extra_kwargs": {},
                "hf_name": "model",
                "hf_config": config,
            },
        )

    assert captured == {}


@pytest.mark.parametrize("execution_provider", ["cpu", "NvTensorRtRtx"])
def test_structured_runtime_config_is_applied_before_success_telemetry(
    monkeypatch, tmp_path, builder_telemetry, builder_module, execution_provider
):
    captured = {}
    config = types.SimpleNamespace(architectures=["LlamaForCausalLM"])
    effective = types.SimpleNamespace(precision="fp16", runtime_config={"search": {"max_length": 128}})

    class ConfigOnlyModel:
        def make_genai_config(self, config, extra_kwargs, output_dir):
            (Path(output_dir) / "genai_config.json").write_text('{"search": {}}', encoding="utf-8")

        def save_processing(self, hf_name, extra_kwargs, output_dir):
            captured["saved_config"] = json.loads((Path(output_dir) / "genai_config.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(builder_module, "set_io_dtype", lambda *args: object())
    monkeypatch.setattr(builder_module, "set_onnx_dtype", lambda *args: object())
    monkeypatch.setattr(builder_module, "validate_model_dependent_config", lambda *args: None)

    def make_model(config, io_dtype, onnx_dtype, provider, cache_dir, extra_options):
        captured["provider"] = provider
        captured["use_qdq"] = extra_options.get("use_qdq", False)
        return ConfigOnlyModel()

    monkeypatch.setattr(builder_module, "LlamaModel", make_model)
    monkeypatch.setattr(
        builder_module,
        "apply_runtime_config",
        lambda generated, runtime: {"search": runtime["search"]},
    )

    def capture_event(self, **kwargs):
        captured.update(kwargs)
        captured["emitted_after_processing"] = "saved_config" in captured

    monkeypatch.setattr(builder_module.ModelBuilderTelemetry, "emit", capture_event)

    builder_module.create_model(
        "model",
        Path("model.gguf"),
        str(tmp_path / "output"),
        "int4",
        execution_provider,
        str(tmp_path / "cache"),
        config_only=True,
        _effective_builder_config=effective,
        hf_details={"extra_kwargs": {}, "hf_name": "model", "hf_config": config},
    )

    assert captured["saved_config"] == {"search": {"max_length": 128}}
    assert captured["emitted_after_processing"] is True
    assert captured["precision"] == "fp16"
    assert captured["input_path"] == "model.gguf"
    assert captured["provider"] == ("trt-rtx" if execution_provider == "NvTensorRtRtx" else "cpu")
    assert captured["use_qdq"] is (execution_provider == "NvTensorRtRtx")

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

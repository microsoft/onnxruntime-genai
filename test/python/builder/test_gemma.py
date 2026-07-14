from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

gemma_module = _load_builder_module("gemma")
Gemma2Model = gemma_module.Gemma2Model


def _make_gemma_model(ep: str):
    model = Gemma2Model.__new__(Gemma2Model)
    model.ep = ep
    model.values = {
        "root": types.SimpleNamespace(shape=["batch_size", "sequence_length", 128]),
        "skip": types.SimpleNamespace(shape=["batch_size", "sequence_length", 128]),
    }
    calls = []

    def _make_node(op_type, inputs, outputs, name, **kwargs):
        calls.append({"op_type": op_type, "inputs": inputs, "outputs": outputs, "name": name, "kwargs": kwargs})

    model.make_node = _make_node
    model.make_value = lambda *args, **kwargs: None
    return model, calls


def test_cpu_skip_simplified_layernorm_uses_fused_op():
    model, calls = _make_gemma_model("cpu")

    Gemma2Model.make_layernorm_op(
        model,
        "/model/layers.0/pre_feedforward_layernorm/SkipLayerNorm",
        "SkipSimplifiedLayerNormalization",
        ["root", "skip", "weight"],
        ["output_0", "", "", "output_3"],
        True,
        ir.DataType.FLOAT,
        epsilon=1e-6,
    )

    assert [call["op_type"] for call in calls] == ["SkipSimplifiedLayerNormalization"]
    assert calls[0]["kwargs"]["domain"] == "com.microsoft"


def test_non_cpu_skip_simplified_layernorm_uses_fused_op():
    model, calls = _make_gemma_model("cuda")

    Gemma2Model.make_layernorm_op(
        model,
        "/model/layers.0/pre_feedforward_layernorm/SkipLayerNorm",
        "SkipSimplifiedLayerNormalization",
        ["root", "skip", "weight"],
        ["output_0", "", "", "output_3"],
        True,
        ir.DataType.FLOAT,
        epsilon=1e-6,
    )

    assert [call["op_type"] for call in calls] == ["SkipSimplifiedLayerNormalization"]
    assert calls[0]["kwargs"]["domain"] == "com.microsoft"

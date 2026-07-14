from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir
import pytest

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

base_module = _load_builder_module("base")
gemma_module = _load_builder_module("gemma")
Gemma2Model = gemma_module.Gemma2Model
Model = base_module.Model


def _make_model_with_type(model_type: str) -> Model:
    """Create a minimal Model stub with the given HuggingFace architecture name."""
    model = Model.__new__(Model)
    model.model_type = model_type
    return model


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


# ---------------------------------------------------------------------------
# Root-cause regression tests
#
# Bug: when a VLM like Gemma3 was exported as a text-decoder, genai_config.json
# contained `"type": "gemma3"` (the full-VLM type).  The C++ runtime would then
# try to run the model as a VLM, leading to garbage output.
#
# Fix: make_genai_model_type() maps ForConditionalGeneration architectures to
# "<prefix>_text" so the runtime takes the correct text-decoder code path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "architecture, expected_type",
    [
        # VLM architectures: text-decoder export must use the *_text type so the
        # C++ runtime selects the text-decoder path (not the VLM path).
        ("Gemma3ForConditionalGeneration", "gemma3_text"),
        ("Mistral3ForConditionalGeneration", "mistral3_text"),
        ("Qwen25VLForConditionalGeneration", "qwen25vl_text"),
        ("Qwen3VLForConditionalGeneration", "qwen3vl_text"),
        # Pure text architectures: prefix is used as-is.
        ("Gemma3ForCausalLM", "gemma3"),
        ("LlamaForCausalLM", "llama"),
        ("MistralForCausalLM", "mistral"),
        # Already-resolved type strings (set explicitly by builder.py):
        ("gemma3_text", "gemma3_text"),
        ("qwen3", "qwen3"),
    ],
)
def test_make_genai_model_type(architecture, expected_type):
    model = _make_model_with_type(architecture)
    assert model.make_genai_model_type() == expected_type


def test_vlm_architecture_was_previously_mapped_to_wrong_type():
    """Demonstrates the pre-fix behavior that caused garbage output.

    Before make_genai_model_type(), the type was computed by stripping
    everything from 'For' onwards and lower-casing, which maps
    'Gemma3ForConditionalGeneration' -> 'gemma3' (the VLM type).
    The C++ runtime then used the VLM code path for a text-decoder-only
    model, producing garbage output.
    """
    arch = "Gemma3ForConditionalGeneration"
    old_type = arch[: arch.find("For")].lower()  # pre-fix behaviour
    assert old_type == "gemma3"  # the VLM type — wrong for a text-decoder export

    model = _make_model_with_type(arch)
    new_type = model.make_genai_model_type()  # post-fix behaviour
    assert new_type == "gemma3_text"  # the text-decoder type — correct
    assert new_type != old_type  # fix changes the behaviour


# ---------------------------------------------------------------------------
# Layernorm tests
# ---------------------------------------------------------------------------


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

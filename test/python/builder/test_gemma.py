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
Gemma3Model = gemma_module.Gemma3Model
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


# ---------------------------------------------------------------------------
# is_local tests
#
# Root cause: Gemma2Model.is_local previously returned `layer_id % 2 == 1`
# (odd=local), which is the OPPOSITE of HuggingFace Gemma-2 where even layers
# (0, 2, 4, ...) use sliding-window (local) attention and odd layers (1, 3, 5,
# ...) use full (global) attention.  This caused every layer to use the wrong
# attention type, producing garbage output.
#
# Fix: is_local now reads config.layer_types (populated by HuggingFace
# __post_init__) when available, falling back to the correct formula
# (even=local) otherwise.
# ---------------------------------------------------------------------------


def _gemma2_model_with_layer_types(layer_types):
    """Return a minimal Gemma2Model stub with the given layer_types list."""
    model = Gemma2Model.__new__(Gemma2Model)
    model.layer_types = layer_types
    return model


def _gemma3_model_with_layer_types(layer_types):
    """Return a minimal Gemma3Model stub with the given layer_types list."""
    model = Gemma3Model.__new__(Gemma3Model)
    model.layer_types = layer_types
    return model


class TestGemma2IsLocal:
    def test_reads_layer_types_from_config(self):
        """is_local uses layer_types when it is set."""
        layer_types = ["sliding_attention", "full_attention"] * 14  # 28 layers
        model = _gemma2_model_with_layer_types(layer_types)
        # Even layers (0, 2, ...) are "sliding_attention" -> local
        assert model.is_local(0) is True
        assert model.is_local(2) is True
        # Odd layers (1, 3, ...) are "full_attention" -> not local
        assert model.is_local(1) is False
        assert model.is_local(3) is False

    def test_fallback_without_layer_types(self):
        """is_local uses the correct fallback when layer_types is None."""
        model = _gemma2_model_with_layer_types(None)
        # HuggingFace default: even=local, odd=global
        assert model.is_local(0) is True
        assert model.is_local(1) is False
        assert model.is_local(2) is True
        assert model.is_local(3) is False

    def test_previous_formula_was_inverted(self):
        """Demonstrates that the old formula `layer_id % 2 == 1` was wrong.

        HuggingFace Gemma-2 sets layer_types[i] = "sliding_attention" when
        bool((i + 1) % 2) is True, which is True for even i (0, 2, 4, ...).
        The old formula returned True for ODD layers -- the opposite.
        """
        old_is_local = lambda layer_id: layer_id % 2 == 1  # noqa: E731 (old, wrong formula)
        hf_is_local = lambda layer_id: bool((layer_id + 1) % 2)  # noqa: E731 (HuggingFace formula)

        for layer_id in range(8):
            # The old formula and HuggingFace are opposite for every layer
            assert old_is_local(layer_id) != hf_is_local(layer_id), (
                f"layer {layer_id}: old={old_is_local(layer_id)}, hf={hf_is_local(layer_id)}"
            )


class TestGemma3IsLocal:
    def test_reads_layer_types_from_config(self):
        """Gemma3 is_local uses layer_types when available."""
        # 26-layer model: 5 local + 1 global per 6 layers
        layer_types = [
            "sliding_attention" if bool((i + 1) % 6) else "full_attention"
            for i in range(26)
        ]
        model = _gemma3_model_with_layer_types(layer_types)
        # Layers 0-4 are local, layer 5 is global
        for i in range(5):
            assert model.is_local(i) is True
        assert model.is_local(5) is False

    def test_fallback_uses_6_pattern(self):
        """Gemma3 is_local fallback uses the 5-local-1-global-per-6 pattern."""
        model = _gemma3_model_with_layer_types(None)
        for i in range(5):
            assert model.is_local(i) is True
        assert model.is_local(5) is False

    def test_custom_sliding_window_pattern_via_layer_types(self):
        """Non-default sliding_window_pattern is handled correctly via layer_types."""
        # sliding_window_pattern = 4: 3 local + 1 global per 4 layers
        layer_types = [
            "sliding_attention" if bool((i + 1) % 4) else "full_attention"
            for i in range(16)
        ]
        model = _gemma3_model_with_layer_types(layer_types)
        for i in range(3):
            assert model.is_local(i) is True
        assert model.is_local(3) is False
        for i in range(4, 7):
            assert model.is_local(i) is True
        assert model.is_local(7) is False

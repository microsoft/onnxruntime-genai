from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

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
Model = base_module.Model


class _RecordingOps:
    """Stand-in for ``Model.op`` that records the ops a builder method creates."""

    def __init__(self, model):
        self._model = model

    def __getattr__(self, op_type):
        def record(*inputs, _name=None, _outputs=None, _domain="", **attributes):
            self._model.nodes.append({"inputs": list(inputs), "attributes": attributes})

        return record


class _FakeGQAModel:
    is_fused_qk_norm_gqa_supported = Model.is_fused_qk_norm_gqa_supported
    make_group_query_attention = Model.make_group_query_attention
    make_paged_attention = Model.make_paged_attention
    get_qk_norm_weight_names = Model.get_qk_norm_weight_names
    get_kv_cache_scale_names = Model.get_kv_cache_scale_names
    extend_with_optional_inputs = Model.extend_with_optional_inputs
    get_qk_norm_weight_inputs = Model.get_qk_norm_weight_inputs
    get_kv_cache_scale_inputs = Model.get_kv_cache_scale_inputs
    get_attention_op_attributes = Model.get_attention_op_attributes

    def __init__(self, ep="cpu", fuse_qk_norm_gqa=True):
        self.ep = ep
        self.extra_options = {"fuse_qk_norm_gqa": fuse_qk_norm_gqa}
        self.kv_cache_quant_type = "none"
        self.num_attn_heads = 8
        self.num_kv_heads = 2
        self.head_size = 16
        self.window_size = -1
        self.attention_attrs = {
            "op_type": "GroupQueryAttention",
            "scale": 0.125,
            "softcap": 0.0,
            "rope": True,
            "use_rope_in_attn": True,
            "qk_norm_epsilon": 1e-6,
        }
        self.rope_attrs = {"interleaved": 0}
        self.io_dtype = None
        self.nodes = []
        self.op = _RecordingOps(self)

    def make_value(self, *args, **kwargs):
        pass


def test_cpu_does_not_enable_fused_qk_norm_gqa_by_default():
    assert not _FakeGQAModel("cpu").is_fused_qk_norm_gqa_supported()
    assert _FakeGQAModel("cuda").is_fused_qk_norm_gqa_supported()


def test_plain_gqa_omits_qk_norm_epsilon_attribute():
    model = _FakeGQAModel()

    model.make_group_query_attention("/gqa", q_path="q", k_path="k", v_path="v")

    assert "qk_norm_epsilon" not in model.nodes[-1]["attributes"]


def test_fused_qk_norm_gqa_emits_qk_norm_epsilon_attribute():
    model = _FakeGQAModel("cuda")

    model.make_group_query_attention(
        "/gqa",
        q_path="q",
        k_path="k",
        v_path="v",
        q_norm_weight="q_norm_weight",
        k_norm_weight="k_norm_weight",
    )

    assert model.nodes[-1]["attributes"]["qk_norm_epsilon"] == 1e-6


def test_paged_attention_preserves_sliding_window_size():
    model = _FakeGQAModel("cuda")
    model.window_size = 4096

    model.make_paged_attention(
        "/paged",
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_lengths="cumulative_sequence_lengths",
        past_sequence_lengths="past_sequence_lengths",
        block_table="block_table",
    )

    assert model.nodes[-1]["attributes"]["local_window_size"] == 4096


def test_quantized_gqa_emits_scale_inputs_and_attributes():
    model = _FakeGQAModel("cuda")
    model.kv_cache_quant_type = "int4_per_channel"
    model.kv_quant_type = "PER_CHANNEL"
    model.kv_cache_bit_width = 4

    model.make_group_query_attention("/gqa", layer_id=3, q_path="q", k_path="k", v_path="v")

    node = model.nodes[-1]
    assert node["inputs"][12:14] == [
        "model.layers.3.attn.k_scale",
        "model.layers.3.attn.v_scale",
    ]
    assert node["attributes"]["k_quant_type"] == "PER_CHANNEL"
    assert node["attributes"]["v_quant_type"] == "PER_CHANNEL"
    assert node["attributes"]["kv_cache_bit_width"] == 4


def test_quantized_paged_attention_emits_scale_inputs_and_attributes():
    model = _FakeGQAModel("cuda")
    model.kv_cache_quant_type = "int8_per_channel"
    model.kv_quant_type = "PER_CHANNEL"
    model.kv_cache_bit_width = 8

    model.make_paged_attention(
        "/paged",
        layer_id=3,
        q_path="q",
        k_path="k",
        v_path="v",
        cumulative_sequence_lengths="cumulative_sequence_lengths",
        past_sequence_lengths="past_sequence_lengths",
        block_table="block_table",
    )

    node = model.nodes[-1]
    assert node["inputs"][14:16] == [
        "model.layers.3.attn.k_scale",
        "model.layers.3.attn.v_scale",
    ]
    assert node["attributes"]["k_quant_type"] == "PER_CHANNEL"
    assert node["attributes"]["v_quant_type"] == "PER_CHANNEL"
    # PagedAttention derives the cache element type from the tensor, so it has no bit-width attribute.
    assert "kv_cache_bit_width" not in node["attributes"]


def test_quantized_paged_attention_requires_layer_id():
    model = _FakeGQAModel("cuda")
    model.kv_cache_quant_type = "int8_per_tensor"
    model.kv_quant_type = "PER_TENSOR"

    with pytest.raises(ValueError, match="layer_id is required"):
        model.make_paged_attention(
            "/paged",
            q_path="q",
            k_path="k",
            v_path="v",
            cumulative_sequence_lengths="cumulative_sequence_lengths",
            past_sequence_lengths="past_sequence_lengths",
            block_table="block_table",
        )


@pytest.mark.parametrize("provided", ["q_norm_weight", "k_norm_weight"])
def test_paged_attention_rejects_unpaired_qk_norm_weights(provided):
    model = _FakeGQAModel("cuda")

    with pytest.raises(ValueError, match="must be provided together"):
        model.make_paged_attention(
            "/paged",
            q_path="q",
            k_path="k",
            v_path="v",
            cumulative_sequence_lengths="cumulative_sequence_lengths",
            past_sequence_lengths="past_sequence_lengths",
            block_table="block_table",
            **{provided: "norm"},
        )


def test_get_qk_norm_weight_names_follows_convention():
    # The fused GQA path derives the Q/K norm weight names by convention (like `sinks`)
    # instead of storing them in attention_attrs; the initializer creation and the GQA
    # input must agree on these names.
    model = _FakeGQAModel("cuda")

    assert model.get_qk_norm_weight_names(3) == (
        "model.layers.3.attn.q_norm.layernorm.weight",
        "model.layers.3.attn.k_norm.layernorm.weight",
    )

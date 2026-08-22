# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the windowed KV cache plumbing in the model builder.

Sliding-window layers can keep a KV cache that is smaller than ``max_length`` on
execution providers that evict entries themselves:

* ``trt-rtx`` evicts inside the EP.
* ``cuda`` evicts inside ``GroupQueryAttention`` via the ``sliding_window_cache``
  attribute.

These tests cover the three builder-side pieces that make that work, standalone
(no model download):

* the ``sliding_window_cache`` attribute on the GQA node,
* the distinct symbolic ``sliding`` sequence dim on the windowed layers' cache,
* the ``model.decoder.sliding_window`` block (including ``cache_slack``) written
  to ``genai_config.json``,
* the ``windowed_kv_cache`` opt-out, which builds a full-length-KV baseline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parent))


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
resolve_windowed_kv_cache_eps = base_module.resolve_windowed_kv_cache_eps


# ===========================================================================
# resolve_windowed_kv_cache_eps: the windowed_kv_cache opt-out
# ===========================================================================


def test_windowed_kv_cache_is_on_by_default():
    assert resolve_windowed_kv_cache_eps({}) == {"trt-rtx", "cuda", "cpu"}


@pytest.mark.parametrize("extra_options", [{"windowed_kv_cache": False}, {"use_paged_attention": True}])
def test_windowed_kv_cache_can_be_turned_off(extra_options):
    # Explicit opt-out builds a full-length-KV baseline; PagedAttention has no windowed-cache
    # mode, so it never qualifies either way.
    assert resolve_windowed_kv_cache_eps(extra_options) == set()


def test_windowed_kv_cache_eps_are_not_shared_between_models():
    # The caller mutates nothing, but two models must not alias the same set.
    first = resolve_windowed_kv_cache_eps({})
    first.discard("cuda")
    assert "cuda" in resolve_windowed_kv_cache_eps({})


# ===========================================================================
# make_group_query_attention: the sliding_window_cache attribute
# ===========================================================================


def _make_gqa_model(ep, window_size):
    model = Model.__new__(Model)
    model.ep = ep
    model.extra_options = {}
    model.eps_with_windowed_kv_cache = {"trt-rtx", "cuda", "cpu"}
    model.kv_cache_attrs = {"quant_type": "none", "quant_mode": "PER_TENSOR", "bit_width": 0}
    model.num_attn_heads = 8
    model.num_kv_heads = 2
    model.head_size = 16
    # Builders for alternating-attention models (gemma, gpt-oss, smollm) temporarily set
    # window_size to -1 around the global layers, so window_size > 0 here means "this
    # layer is a sliding-window layer".
    model.window_size = window_size
    model.attention_attrs = {
        "op_type": "GroupQueryAttention",
        "scale": 0.125,
        "softcap": 0.0,
        "use_rope_in_attn": True,
        "qk_norm_epsilon": 1e-6,
    }
    model.rope_attrs = {"interleaved": 0}
    model.io_dtype = None
    model.nodes = []

    def make_node(op_type, inputs, outputs, name, domain="", **attributes):
        model.nodes.append({"op_type": op_type, "inputs": inputs, "attributes": attributes})

    model.make_node = make_node
    model.make_value = lambda *args, **kwargs: None
    return model


def test_cuda_sliding_window_layer_enables_sliding_window_cache():
    model = _make_gqa_model("cuda", window_size=128)

    model.make_group_query_attention("/gqa", layer_id=0, q_path="q", k_path="k", v_path="v")

    attributes = model.nodes[-1]["attributes"]
    assert attributes["sliding_window_cache"] == 1
    assert attributes["local_window_size"] == 128


@pytest.mark.parametrize("window_size", [-1, 0, None])
def test_cuda_global_layer_omits_sliding_window_cache(window_size):
    # Global (full-attention) layers keep a max_length-sized cache, so the kernel must
    # index it in absolute coordinates.
    model = _make_gqa_model("cuda", window_size=window_size)

    model.make_group_query_attention("/gqa", layer_id=1, q_path="q", k_path="k", v_path="v")

    assert "sliding_window_cache" not in model.nodes[-1]["attributes"]


def test_cpu_sliding_window_layer_enables_sliding_window_cache():
    model = _make_gqa_model("cpu", window_size=128)

    model.make_group_query_attention("/gqa", layer_id=0, q_path="q", k_path="k", v_path="v")

    attributes = model.nodes[-1]["attributes"]
    assert attributes["sliding_window_cache"] == 1
    assert attributes["local_window_size"] == 128


@pytest.mark.parametrize("ep", ["trt-rtx", "dml", "webgpu"])
def test_non_cuda_ep_omits_sliding_window_cache(ep):
    # trt-rtx gets a windowed cache but the EP evicts internally (no attribute).
    # dml and webgpu have no windowed cache at all.
    model = _make_gqa_model(ep, window_size=128)

    model.make_group_query_attention("/gqa", layer_id=0, q_path="q", k_path="k", v_path="v")

    assert "sliding_window_cache" not in model.nodes[-1]["attributes"]


@pytest.mark.parametrize("ep", ["cuda", "cpu"])
def test_windowed_kv_cache_opt_out_omits_sliding_window_cache(ep):
    # With the opt-out the layer's cache is max_length-sized, so the kernel must index it in
    # absolute coordinates like a global layer does.
    model = _make_gqa_model(ep, window_size=128)
    model.eps_with_windowed_kv_cache = set()

    model.make_group_query_attention("/gqa", layer_id=0, q_path="q", k_path="k", v_path="v")

    attributes = model.nodes[-1]["attributes"]
    assert "sliding_window_cache" not in attributes
    # The window itself still has to be masked, only the cache layout changes.
    assert attributes["local_window_size"] == 128


# ===========================================================================
# make_key_value_cache_shape: the distinct symbolic dim for windowed layers
# ===========================================================================


_CACHE_SHAPE = ["batch_size", 2, "past_sequence_length", 16]


def _make_shape_model(ep, local_layers=()):
    model = Model.__new__(Model)
    model.ep = ep
    model.eps_with_windowed_kv_cache = {"trt-rtx", "cuda", "cpu"}
    model.use_windowed_paged_kv_cache = False  # the paged ring is covered by its own test module
    if local_layers is not None:
        model.is_local = lambda layer_id: layer_id in local_layers
    return model


@pytest.mark.parametrize("ep", ["cuda", "trt-rtx", "cpu"])
def test_windowed_ep_renames_sequence_dim_on_sliding_layers(ep):
    model = _make_shape_model(ep, local_layers=(0,))

    assert model.make_key_value_cache_shape(0, list(_CACHE_SHAPE)) == [
        "batch_size",
        2,
        "past_sliding_length",
        16,
    ]


@pytest.mark.parametrize("ep", ["cuda", "trt-rtx", "cpu"])
def test_windowed_ep_keeps_sequence_dim_on_global_layers(ep):
    # Global layers keep the shared 'sequence' dim so they stay unified at max_length.
    model = _make_shape_model(ep, local_layers=(0,))

    assert model.make_key_value_cache_shape(1, list(_CACHE_SHAPE)) == _CACHE_SHAPE


@pytest.mark.parametrize("ep", ["dml", "webgpu"])
def test_non_windowed_ep_keeps_sequence_dim(ep):
    model = _make_shape_model(ep, local_layers=(0,))

    assert model.make_key_value_cache_shape(0, list(_CACHE_SHAPE)) == _CACHE_SHAPE


def test_model_without_alternating_attention_keeps_sequence_dim():
    # Models that never define is_local have a uniform attention pattern.
    model = _make_shape_model("cuda", local_layers=None)

    assert model.make_key_value_cache_shape(0, list(_CACHE_SHAPE)) == _CACHE_SHAPE


@pytest.mark.parametrize("ep", ["cuda", "trt-rtx", "cpu"])
def test_windowed_kv_cache_opt_out_keeps_sequence_dim(ep):
    # With the opt-out every layer's cache is allocated at max_length, so they share one dim.
    model = _make_shape_model(ep, local_layers=(0,))
    model.eps_with_windowed_kv_cache = set()

    assert model.make_key_value_cache_shape(0, list(_CACHE_SHAPE)) == _CACHE_SHAPE


# ===========================================================================
# make_genai_config: the model.decoder.sliding_window block
# ===========================================================================


class _NoGenerationConfig:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise FileNotFoundError("no generation_config.json")


def _write_genai_config(monkeypatch, out_dir, ep, window_size, num_layers=4, eps_with_windowed_kv_cache=None):
    hf_config = SimpleNamespace(bos_token_id=None, eos_token_id=[2], pad_token_id=None)
    monkeypatch.setattr(base_module, "GenerationConfig", _NoGenerationConfig)

    model = Model.__new__(Model)
    model.hf_token = None
    model.hf_remote = False
    model.ep = ep
    model.ep_attrs = {ep: {}}
    model.extra_options = {}
    model.use_paged_attention = False
    model.use_windowed_paged_kv_cache = False  # the paged ring is covered by its own test module
    model.past_present_share_buffer = True
    model.context_length = 1024
    model.filename = "model.onnx"
    model.head_size = 16
    model.hidden_size = 128
    model.num_attn_heads = 8
    model.num_kv_heads = 2
    model.num_layers = num_layers
    model.model_type = "TestForCausalLM"
    model.vocab_size = 32
    model.window_size = window_size
    model.eps_with_windowed_kv_cache = (
        {"trt-rtx", "cuda", "cpu"} if eps_with_windowed_kv_cache is None else eps_with_windowed_kv_cache
    )
    model.window_kv_cache_slack = 0  # let runtime apply EP defaults
    # Alternating attention: even layers are sliding-window layers.
    model.is_local = lambda layer_id: layer_id % 2 == 0
    model.input_names = {"input_ids": "input_ids", "past_key_values.key": [], "past_key_values.value": []}
    model.output_names = {"logits": "logits", "present.key": [], "present.value": []}

    model.make_genai_config(hf_config, {}, str(out_dir))
    return json.loads((Path(out_dir) / "genai_config.json").read_text())


@pytest.mark.parametrize("ep", ["cuda", "trt-rtx"])
def test_genai_config_emits_sliding_window_with_cache_slack(monkeypatch, tmp_path, ep):
    config = _write_genai_config(monkeypatch, tmp_path, ep, window_size=128)

    sliding_window = config["model"]["decoder"]["sliding_window"]
    assert sliding_window == {
        "window_size": 128,
        "slide_key_value_cache": False,
        "slide_inputs": False,
        "layers": [0, 2],
        "cache_slack": 0,  # 0 = use EP default at runtime (0 for CUDA, 16 for CPU)
    }


def test_genai_config_emits_sliding_window_for_cpu(monkeypatch, tmp_path):
    config = _write_genai_config(monkeypatch, tmp_path, "cpu", window_size=128)

    sliding_window = config["model"]["decoder"]["sliding_window"]
    assert sliding_window == {
        "window_size": 128,
        "slide_key_value_cache": False,
        "slide_inputs": False,
        "layers": [0, 2],
        "cache_slack": 0,  # runtime applies CPU default of 16
    }


@pytest.mark.parametrize("ep", ["dml", "webgpu"])
def test_genai_config_omits_sliding_window_for_other_eps(monkeypatch, tmp_path, ep):
    config = _write_genai_config(monkeypatch, tmp_path, ep, window_size=128)

    assert "sliding_window" not in config["model"]["decoder"]


@pytest.mark.parametrize("window_size", [-1, 0, None])
def test_genai_config_omits_sliding_window_without_a_window(monkeypatch, tmp_path, window_size):
    config = _write_genai_config(monkeypatch, tmp_path, "cuda", window_size=window_size)

    assert "sliding_window" not in config["model"]["decoder"]


@pytest.mark.parametrize("ep", ["cuda", "trt-rtx", "cpu"])
def test_genai_config_omits_sliding_window_when_opted_out(monkeypatch, tmp_path, ep):
    config = _write_genai_config(monkeypatch, tmp_path, ep, window_size=128, eps_with_windowed_kv_cache=set())

    assert "sliding_window" not in config["model"]["decoder"]

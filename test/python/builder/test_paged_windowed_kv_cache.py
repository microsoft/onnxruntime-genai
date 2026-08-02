# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the paged windowed KV cache plumbing in the model builder.

With PagedAttention a sliding-window layer never reads further back than
``window_size`` positions, so it does not need one block per position. It is instead
given a short *ring* of blocks that the runtime repeats across the block table, so
position ``p`` lands in slot ``p mod (ring_blocks * block_size)``. The operator itself
needs no change, because it already masks reads to ``[kv_end - window_size, kv_end]``.

These tests cover the four builder-side pieces that make that work, standalone
(no model download):

* ``resolve_windowed_paged_kv_cache``, which decides whether the model gets a ring,
* the extra ``block_table_windowed`` graph input,
* the ``num_blocks_windowed`` symbolic dim on the windowed layers' cache,
* the ``sliding_window`` block and the ``search.chunk_size`` default written to
  ``genai_config.json``.
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
resolve_windowed_paged_kv_cache = base_module.resolve_windowed_paged_kv_cache


# ===========================================================================
# resolve_windowed_paged_kv_cache: when the ring applies
# ===========================================================================


def test_ring_needs_paged_attention():
    # Without PagedAttention the sliding-window layers use the GQA windowed cache instead.
    assert resolve_windowed_paged_kv_cache({}, 128) is False


def test_ring_is_on_for_paged_sliding_window_models():
    assert resolve_windowed_paged_kv_cache({"use_paged_attention": True}, 128) is True


@pytest.mark.parametrize("window_size", [-1, 0, None])
def test_ring_needs_a_window(window_size):
    # A model with no sliding-window layers has nothing to shorten.
    assert resolve_windowed_paged_kv_cache({"use_paged_attention": True}, window_size) is False


def test_ring_honours_the_windowed_kv_cache_opt_out():
    # windowed_kv_cache=false builds a full-length-KV baseline, paged or not.
    extra_options = {"use_paged_attention": True, "windowed_kv_cache": False}
    assert resolve_windowed_paged_kv_cache(extra_options, 128) is False


# ===========================================================================
# uses_windowed_paged_cache: which layers read the ring
# ===========================================================================


def _make_layer_model(use_ring, local_layers=(0, 2)):
    model = Model.__new__(Model)
    model.use_windowed_paged_kv_cache = use_ring
    if local_layers is not None:
        model.is_local = lambda layer_id: layer_id in local_layers
    return model


def test_only_sliding_window_layers_use_the_ring():
    model = _make_layer_model(use_ring=True)

    assert [model.uses_windowed_paged_cache(i) for i in range(4)] == [True, False, True, False]


def test_no_layer_uses_the_ring_when_it_is_off():
    model = _make_layer_model(use_ring=False)

    assert [model.uses_windowed_paged_cache(i) for i in range(4)] == [False] * 4


def test_model_without_alternating_attention_uses_no_ring():
    # Models with a uniform attention pattern never define is_local.
    model = _make_layer_model(use_ring=True, local_layers=None)

    assert model.uses_windowed_paged_cache(0) is False


# ===========================================================================
# has_windowed_paged_layers: whether the ring is emitted at all
# ===========================================================================


def _make_layer_count_model(use_ring, local_layers=(0, 2), num_layers=4):
    model = _make_layer_model(use_ring, local_layers)
    model.num_layers = num_layers
    return model


def test_ring_is_emitted_when_some_layer_is_local():
    assert _make_layer_count_model(use_ring=True).has_windowed_paged_layers() is True


def test_ring_is_not_emitted_without_local_layers():
    # A paged model whose config carries sliding_window but whose builder cannot say which layers
    # it applies to: nothing reads the ring, so nothing extra may be emitted for it.
    assert _make_layer_count_model(use_ring=True, local_layers=None).has_windowed_paged_layers() is False


def test_ring_is_not_emitted_when_it_is_off():
    assert _make_layer_count_model(use_ring=False).has_windowed_paged_layers() is False


# ===========================================================================
# make_key_value_cache_shape: the num_blocks_windowed dim
# ===========================================================================


_PAGED_CACHE_SHAPE = ["num_blocks", 256, 2, 16]


def _make_shape_model(use_ring, local_layers=(0,)):
    model = _make_layer_model(use_ring, local_layers)
    model.ep = "cuda"
    # PagedAttention never qualifies for the GQA-style windowed cache.
    model.eps_with_windowed_kv_cache = set()
    return model


def test_ring_layer_gets_its_own_block_count():
    model = _make_shape_model(use_ring=True)

    # Only the block count shrinks; block_size, heads and head_size are untouched.
    assert model.make_key_value_cache_shape(0, list(_PAGED_CACHE_SHAPE)) == ["num_blocks_windowed", 256, 2, 16]


def test_global_layer_keeps_the_shared_block_count():
    model = _make_shape_model(use_ring=True)

    assert model.make_key_value_cache_shape(1, list(_PAGED_CACHE_SHAPE)) == _PAGED_CACHE_SHAPE


def test_every_layer_keeps_the_shared_block_count_without_the_ring():
    model = _make_shape_model(use_ring=False)

    assert model.make_key_value_cache_shape(0, list(_PAGED_CACHE_SHAPE)) == _PAGED_CACHE_SHAPE


# ===========================================================================
# make_inputs_init: the block_table_windowed graph input
# ===========================================================================


def _make_inputs_model(use_paged_attention, use_ring):
    model = Model.__new__(Model)
    model.extra_options = {}
    model.use_paged_attention = use_paged_attention
    model.use_windowed_paged_kv_cache = use_ring
    model.num_kv_heads = 2
    model.head_size = 16
    model.input_names = {
        "input_ids": "input_ids",
        "inputs_embeds": "inputs_embeds",
        "attention_mask": "attention_mask",
        "block_table": "block_table",
        "block_table_windowed": "block_table_windowed",
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
        "past_sequence_lengths": "past_sequence_lengths",
        "attention_metadata": "attention_metadata",
    }
    model.input_shapes = {
        "input_ids": ["batch_size", "sequence_length"],
        "past_key_values.key": [],
        "past_key_values.value": [],
    }
    return model


def test_ring_model_keeps_the_windowed_block_table():
    model = _make_inputs_model(use_paged_attention=True, use_ring=True)

    model.make_inputs_init()

    assert "block_table_windowed" in model.input_names


def test_paged_model_without_a_ring_drops_the_windowed_block_table():
    model = _make_inputs_model(use_paged_attention=True, use_ring=False)

    model.make_inputs_init()

    assert "block_table_windowed" not in model.input_names
    assert "block_table" in model.input_names


def test_non_paged_model_drops_every_paged_input():
    model = _make_inputs_model(use_paged_attention=False, use_ring=False)

    model.make_inputs_init()

    for name in ["block_table", "block_table_windowed", "cumulative_sequence_lengths", "attention_metadata"]:
        assert name not in model.input_names


# ===========================================================================
# make_inputs_and_outputs: no dangling graph input when no layer is local
# ===========================================================================


def _make_graph_model(use_ring, local_layers=(0,), num_layers=2):
    model = _make_layer_count_model(use_ring, local_layers, num_layers)
    model.model = SimpleNamespace(graph=SimpleNamespace(inputs=[], outputs=[]))
    model.make_value = lambda name, dtype=None, shape=None: name
    model.input_names = {
        "input_ids": "input_ids",
        "block_table": "block_table",
        "block_table_windowed": "block_table_windowed",
    }
    model.input_types = dict.fromkeys(model.input_names)
    model.input_shapes = {name: [] for name in model.input_names}
    model.output_names = {"logits": "logits"}
    model.output_types = {"logits": None}
    model.output_shapes = {"logits": []}
    return model


def test_graph_keeps_the_windowed_block_table_when_a_layer_reads_it():
    model = _make_graph_model(use_ring=True)

    model.make_inputs_and_outputs()

    assert "block_table_windowed" in model.model.graph.inputs


def test_graph_drops_the_windowed_block_table_when_no_layer_reads_it():
    # is_local is assigned after Model.__init__, so make_inputs_init cannot make this call.
    model = _make_graph_model(use_ring=True, local_layers=None)

    model.make_inputs_and_outputs()

    assert "block_table_windowed" not in model.model.graph.inputs
    assert "block_table" in model.model.graph.inputs


# ===========================================================================
# make_attention_op: which block table each layer reads
# ===========================================================================


def _make_attention_model(use_ring, local_layers=(0,)):
    model = _make_layer_model(use_ring, local_layers)
    model.attention_attrs = {"op_type": "PagedAttention"}
    model.input_names = {
        "block_table": "block_table",
        "block_table_windowed": "block_table_windowed",
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
        "past_sequence_lengths": "past_sequence_lengths",
        "attention_metadata": "attention_metadata",
    }
    model.paged_attention_calls = []
    model.make_paged_attention = lambda name, **kwargs: model.paged_attention_calls.append(kwargs)
    return model


def test_sliding_window_layer_reads_the_windowed_block_table():
    model = _make_attention_model(use_ring=True)

    model.make_attention_op("/attn", layer_id=0)

    assert model.paged_attention_calls[-1]["block_table"] == "block_table_windowed"


def test_global_layer_reads_the_growing_block_table():
    model = _make_attention_model(use_ring=True)

    model.make_attention_op("/attn", layer_id=1)

    assert model.paged_attention_calls[-1]["block_table"] == "block_table"


def test_every_layer_reads_the_growing_block_table_without_the_ring():
    model = _make_attention_model(use_ring=False)

    model.make_attention_op("/attn", layer_id=0)

    assert model.paged_attention_calls[-1]["block_table"] == "block_table"


# ===========================================================================
# make_genai_config: the sliding_window block and the chunk_size default
# ===========================================================================


class _NoGenerationConfig:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        raise FileNotFoundError("no generation_config.json")


def _write_genai_config(
    monkeypatch, out_dir, window_size, extra_options=None, num_layers=4, use_ring=True, has_local_layers=True
):
    hf_config = SimpleNamespace(eos_token_id=[2])
    monkeypatch.setattr(base_module, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: hf_config))
    monkeypatch.setattr(base_module, "GenerationConfig", _NoGenerationConfig)

    model = Model.__new__(Model)
    model.hf_token = None
    model.hf_remote = False
    model.ep = "cuda"
    model.ep_attrs = {"cuda": {}}
    model.extra_options = dict(extra_options or {})
    model.attention_attrs = {"paged_block_size": 256}
    model.use_paged_attention = True
    model.use_windowed_paged_kv_cache = use_ring
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
    # PagedAttention never qualifies for the GQA-style windowed cache.
    model.eps_with_windowed_kv_cache = set()
    model.window_kv_cache_slack = 0
    # Alternating attention: even layers are sliding-window layers.
    if has_local_layers:
        model.is_local = lambda layer_id: layer_id % 2 == 0
    model.input_names = {
        "input_ids": "input_ids",
        "block_table": "block_table",
        "block_table_windowed": "block_table_windowed",
        "cumulative_sequence_lengths": "cumulative_sequence_lengths",
        "past_sequence_lengths": "past_sequence_lengths",
        "attention_metadata": "attention_metadata",
        "past_key_values.key": [],
        "past_key_values.value": [],
    }
    model.output_names = {"logits": "logits", "present.key": [], "present.value": []}

    model.make_genai_config("model_name_or_path", {}, str(out_dir))
    return json.loads((Path(out_dir) / "genai_config.json").read_text())


def test_genai_config_lists_the_ring_layers(monkeypatch, tmp_path):
    config = _write_genai_config(monkeypatch, tmp_path, window_size=128)

    assert config["model"]["decoder"]["sliding_window"] == {
        "window_size": 128,
        "slide_key_value_cache": False,
        "slide_inputs": False,
        "layers": [0, 2],
        # The runtime sizes the ring from window_size and chunk_size, so it keeps no slack.
        "cache_slack": 0,
    }


def test_genai_config_names_the_windowed_block_table(monkeypatch, tmp_path):
    config = _write_genai_config(monkeypatch, tmp_path, window_size=128)

    inputs = config["model"]["decoder"]["inputs"]
    assert inputs["block_table_windowed"] == "block_table_windowed"
    assert inputs["block_table"] == "block_table"


def test_genai_config_defaults_chunk_size_to_the_block_size(monkeypatch, tmp_path):
    # The ring only holds chunk_size + window_size - 1 positions, so a one-shot prefill would
    # overwrite positions it still had to attend to. Chunking is therefore not optional here.
    config = _write_genai_config(monkeypatch, tmp_path, window_size=128)

    assert config["search"]["chunk_size"] == 256


def test_genai_config_honours_paged_chunk_size(monkeypatch, tmp_path):
    config = _write_genai_config(monkeypatch, tmp_path, window_size=128, extra_options={"paged_chunk_size": "64"})

    assert config["search"]["chunk_size"] == 64


def test_genai_config_omits_the_ring_when_it_is_off(monkeypatch, tmp_path):
    config = _write_genai_config(monkeypatch, tmp_path, window_size=128, use_ring=False)

    assert "sliding_window" not in config["model"]["decoder"]
    assert "chunk_size" not in config["search"]
    assert "block_table_windowed" not in config["model"]["decoder"]["inputs"]


def test_genai_config_omits_the_ring_without_local_layers(monkeypatch, tmp_path):
    # sliding_window in the model config, but no layer the builder can point the ring at.
    config = _write_genai_config(monkeypatch, tmp_path, window_size=128, has_local_layers=False)

    assert "sliding_window" not in config["model"]["decoder"]
    assert "chunk_size" not in config["search"]
    assert "block_table_windowed" not in config["model"]["decoder"]["inputs"]
    assert config["model"]["decoder"]["inputs"]["block_table"] == "block_table"

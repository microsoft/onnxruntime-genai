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


def _new_model():
    model = Model.__new__(Model)
    model.context_length_attrs = {
        "state_window": 0,
        "state_window_dims": [],
        "window_kv_cache_slack": 0,
    }
    return model


# ===========================================================================
# make_context_length_init: when the ring applies
# ===========================================================================


def _window_model(extra_options, window_size):
    model = _new_model()
    model.ep = "cuda"
    model.extra_options = extra_options
    model.make_context_length_init(SimpleNamespace(sliding_window=window_size))
    model.num_layers = 2
    model.layer_types = ["sliding_attention", "full_attention"]
    return model


def test_ring_needs_paged_attention():
    # Without PagedAttention the sliding-window layers use the GQA windowed cache instead.
    assert _window_model({}, 128).has_windowed_paged_layers() is False


def test_ring_is_on_for_paged_sliding_window_models():
    assert _window_model({"use_paged_attention": True}, 128).has_windowed_paged_layers() is True


@pytest.mark.parametrize("window_size", [-1, 0, None])
def test_ring_needs_a_window(window_size):
    # A model with no sliding-window layers has nothing to shorten.
    assert _window_model({"use_paged_attention": True}, window_size).has_windowed_paged_layers() is False


def test_ring_honours_the_windowed_kv_cache_opt_out():
    # windowed_kv_cache=false builds a full-length-KV baseline, paged or not.
    extra_options = {"use_paged_attention": True, "windowed_kv_cache": False}
    assert _window_model(extra_options, 128).has_windowed_paged_layers() is False


# ===========================================================================
# is_windowed_paged_layer: which layers read the ring
# ===========================================================================


def _make_layer_model(use_ring, local_layers=(0, 2), num_layers=4):
    model = _new_model()
    model.context_length_attrs["window_kv_cache"] = use_ring
    model.use_paged_attention = True
    model.window_size = 128
    model.num_layers = num_layers
    model.layer_types = [
        "sliding_attention" if local_layers and layer_id in local_layers else "full_attention"
        for layer_id in range(num_layers)
    ]
    return model


def test_only_sliding_window_layers_use_the_ring():
    model = _make_layer_model(use_ring=True)

    assert [model.is_windowed_paged_layer(i) for i in range(4)] == [True, False, True, False]


def test_no_layer_uses_the_ring_when_it_is_off():
    model = _make_layer_model(use_ring=False)

    assert [model.is_windowed_paged_layer(i) for i in range(4)] == [False] * 4


def test_model_without_alternating_attention_uses_no_ring():
    # Models with a uniform attention pattern have no sliding_attention layers.
    model = _make_layer_model(use_ring=True, local_layers=None)

    assert model.is_windowed_paged_layer(0) is False


# ===========================================================================
# has_windowed_paged_layers: whether the ring is emitted at all
# ===========================================================================


def _make_layer_count_model(use_ring, local_layers=(0, 2), num_layers=4):
    return _make_layer_model(use_ring, local_layers, num_layers)


def test_ring_is_emitted_when_some_layer_is_local():
    assert _make_layer_count_model(use_ring=True).has_windowed_paged_layers() is True


def test_ring_is_not_emitted_without_local_layers():
    # A paged model whose config carries sliding_window but has no local layers does not emit a ring.
    assert _make_layer_count_model(use_ring=True, local_layers=None).has_windowed_paged_layers() is False


def test_ring_is_not_emitted_when_it_is_off():
    assert _make_layer_count_model(use_ring=False).has_windowed_paged_layers() is False


def test_ring_is_not_emitted_when_every_layer_is_local():
    assert (
        _make_layer_count_model(use_ring=True, local_layers=(0, 1), num_layers=2).has_windowed_paged_layers() is False
    )


# ===========================================================================
# make_key_value_cache_shape: the num_blocks_windowed dim
# ===========================================================================


_PAGED_CACHE_SHAPE = ["num_blocks", 256, 2, 16]


def _make_shape_model(use_ring, local_layers=(0,)):
    model = _make_layer_model(use_ring, local_layers)
    model.ep = "cuda"
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


def test_all_local_layers_keep_the_shared_block_count():
    model = _make_shape_model(use_ring=True, local_layers=(0, 1, 2, 3))

    assert model.make_key_value_cache_shape(0, list(_PAGED_CACHE_SHAPE)) == _PAGED_CACHE_SHAPE


# ===========================================================================
# make_inputs_init: the block_table_windowed graph input
# ===========================================================================


def _make_inputs_model(use_paged_attention, use_ring):
    model = _new_model()
    model.extra_options = {}
    model.use_paged_attention = use_paged_attention
    model.context_length_attrs["window_kv_cache"] = use_ring
    model.window_size = 128
    model.num_layers = 2
    model.layer_types = ["sliding_attention", "full_attention"]
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
        "past.conv": {},
        "past.recurrent": {},
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


def test_webgpu_paged_model_keeps_attention_metadata():
    model = _make_inputs_model(use_paged_attention=True, use_ring=False)
    model.ep = "webgpu"

    model.make_inputs_init()

    assert "attention_metadata" in model.input_names


def test_non_paged_model_drops_every_paged_input():
    model = _make_inputs_model(use_paged_attention=False, use_ring=False)

    model.make_inputs_init()

    for name in ["block_table", "block_table_windowed", "cumulative_sequence_lengths", "attention_metadata"]:
        assert name not in model.input_names


# ===========================================================================
# make_inputs_and_outputs: no dangling graph input when no layer is local
# ===========================================================================


def _make_graph_model(use_ring, local_layers=(0,), num_layers=2):
    model = _make_inputs_model(use_paged_attention=True, use_ring=use_ring)
    model.num_layers = num_layers
    model.layer_types = [
        "sliding_attention" if local_layers and layer_id in local_layers else "full_attention"
        for layer_id in range(num_layers)
    ]
    model.make_inputs_init()
    model.model = SimpleNamespace(graph=SimpleNamespace(inputs=[], outputs=[]))
    model.make_value = lambda name, dtype=None, shape=None: name
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
    model = _make_graph_model(use_ring=True, local_layers=None)

    model.make_inputs_and_outputs()

    assert "block_table_windowed" not in model.model.graph.inputs
    assert "block_table" in model.model.graph.inputs


def test_graph_drops_the_windowed_block_table_when_every_layer_is_local():
    model = _make_graph_model(use_ring=True, local_layers=(0, 1), num_layers=2)

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


def test_all_local_layers_read_the_growing_block_table():
    model = _make_attention_model(use_ring=True, local_layers=(0, 1, 2, 3))

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
    monkeypatch,
    out_dir,
    window_size,
    extra_options=None,
    num_layers=4,
    use_ring=True,
    has_local_layers=True,
    all_local_layers=False,
):
    hf_config = SimpleNamespace(eos_token_id=[2])
    monkeypatch.setattr(base_module, "GenerationConfig", _NoGenerationConfig)

    model = _new_model()
    model.hf_token = None
    model.hf_remote = False
    model.ep = "cuda"
    model.ep_attrs = {"cuda": {}}
    model.extra_options = dict(extra_options or {})
    model.attention_attrs = {"paged_block_size": 256}
    model.use_paged_attention = True
    model.context_length_attrs["window_kv_cache"] = use_ring
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
    model.context_length_attrs["window_kv_cache_slack"] = 0
    model.layer_types = [
        "sliding_attention" if has_local_layers and (all_local_layers or layer_id % 2 == 0) else "full_attention"
        for layer_id in range(num_layers)
    ]
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

    model.make_genai_config(hf_config, {}, str(out_dir))
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


def test_genai_config_omits_the_ring_when_every_layer_is_local(monkeypatch, tmp_path):
    config = _write_genai_config(monkeypatch, tmp_path, window_size=128, all_local_layers=True)

    assert "sliding_window" not in config["model"]["decoder"]
    assert "chunk_size" not in config["search"]
    assert "block_table_windowed" not in config["model"]["decoder"]["inputs"]
    assert config["model"]["decoder"]["inputs"]["block_table"] == "block_table"

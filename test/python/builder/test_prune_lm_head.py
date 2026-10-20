# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Tests for `prune_lm_head=true` on standard (non-paged) models.

A pruned LM head projects one hidden state per row. In a padded batch that state must be each row's
last real token, not the last column. The graph takes it at the last 1 in `attention_mask`.
"""

from __future__ import annotations

import types

import numpy as np
import onnxruntime as ort
import pytest
from _builder_test_utils import load_builder_module, ort_runs_causal_conv_with_state

base_module = load_builder_module("base")
ir = base_module.ir
torch = base_module.torch
Model = base_module.Model
# The real transformers, or the stub that `load_builder_module` puts in its place when it is missing.
transformers = base_module.transformers

HIDDEN_SIZE = 4
PAD_TOKEN_ID = 0
HEAD_SIZE = 16


def _make_lm_head_graph(tmp_path, io_dtype, with_mask):
    """Build only the pruned LM head. The MatMul is an Identity, so the logits are the picked hidden states."""
    model = Model.__new__(Model)
    model.use_paged_attention = False
    model.prune_lm_head = True
    model.io_dtype = io_dtype
    model.hidden_size = HIDDEN_SIZE
    model.vocab_size = HIDDEN_SIZE
    model.input_names = {"attention_mask": "attention_mask"} if with_mask else {}
    model.output_types = {"logits": io_dtype}
    model.output_shapes = {"logits": ["batch_size", "sequence_length", model.vocab_size]}
    model.layernorm_attrs = {"output_0": "hidden_states"}
    model.lm_head_attrs = {"scale": 1, "mask": None, "softcap": 0.0}
    model.values = {}
    model.node_names = set()
    graph = ir.Graph(inputs=(), outputs=(), nodes=(), opset_imports={"": 21}, name="pruned_lm_head_test")
    model.model = ir.Model(graph, ir_version=10)
    graph.inputs.append(model.make_value("hidden_states", io_dtype, ["batch_size", "sequence_length", HIDDEN_SIZE]))
    if with_mask:
        graph.inputs.append(
            model.make_value("attention_mask", ir.DataType.INT64, ["batch_size", "total_sequence_length"])
        )

    def make_matmul(_lm_head, name, root_input, **_kwargs):
        model.make_node("Identity", inputs=[root_input], outputs=["logits"], name=name)
        model.make_value("logits", io_dtype, ["batch_size", 1, model.vocab_size])
        return name

    model.make_matmul = make_matmul
    model.make_lm_head(types.SimpleNamespace(bias=None))
    graph.outputs.append(model.make_value("logits"))

    path = tmp_path / "pruned_lm_head.onnx"
    ir.save(model.model, path)
    return model, ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize(
    "mask, sequence_length, expected_columns",
    [
        # First prefill of a right-padded batch (genai pads on the right): each row's last 1.
        ([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], 5, [2, 4, 0]),
        # Left padding: every row ends at the last column.
        ([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], 5, [4, 4]),
        # Zeros inside a row: the Qwen-VL mask is 0 at every pad-id token, also inside a prompt.
        ([[1, 1, 0, 1, 1]], 5, [4]),
        ([[1, 0, 1, 0, 0], [1, 1, 0, 1, 1]], 5, [2, 4]),
        # Decode: one new token per row, so the only column is picked even though the mask is longer.
        ([[1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1]], 1, [0, 0]),
        # Batch-1 continuation (chunked prefill or append after generate): the last new token.
        ([[1, 1, 1, 1, 1, 1, 1]], 3, [2]),
        # Continuation over a padded past: every row adds the same 3 tokens, so both pick the last one.
        ([[1, 1, 1, 0, 0, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1]], 3, [2, 2]),
        # Static mask (graph capture): zeros fill the buffer to max_length, after the last real token.
        ([[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]], 4, [1, 3]),
        # A row with no real token reads column 0 instead of going out of range.
        ([[0, 0, 0], [1, 1, 1]], 3, [0, 2]),
    ],
)
@pytest.mark.parametrize("io_dtype, np_dtype", [(ir.DataType.FLOAT, np.float32), (ir.DataType.FLOAT16, np.float16)])
def test_pruned_lm_head_picks_each_rows_last_real_token(
    tmp_path, io_dtype, np_dtype, mask, sequence_length, expected_columns
):
    model, session = _make_lm_head_graph(tmp_path, io_dtype, with_mask=True)
    mask = np.asarray(mask, dtype=np.int64)
    batch_size = mask.shape[0]
    hidden_states = np.arange(batch_size * sequence_length * HIDDEN_SIZE, dtype=np_dtype)
    hidden_states = hidden_states.reshape(batch_size, sequence_length, HIDDEN_SIZE)

    (logits,) = session.run(None, {"hidden_states": hidden_states, "attention_mask": mask})

    expected = hidden_states[np.arange(batch_size), expected_columns][:, np.newaxis, :]
    np.testing.assert_array_equal(logits, expected)
    assert model.output_shapes["logits"] == ["batch_size", 1, HIDDEN_SIZE]
    assert session.get_outputs()[0].shape == ["batch_size", 1, HIDDEN_SIZE]


def test_pruned_lm_head_uses_shapes_only_as_expand_shapes(tmp_path):
    # Shape writes on the CPU. If its output fed device math, CUDA and WebGPU would add a copy node,
    # and a copy node blocks graph capture.
    model, _ = _make_lm_head_graph(tmp_path, ir.DataType.FLOAT, with_mask=True)
    shape_nodes = [node for node in model.model.graph if node.op_type == "Shape"]
    # One for the mask length T and one for the sequence length S.
    assert len(shape_nodes) == 2
    for shape_node in shape_nodes:
        consumers = list(shape_node.outputs[0].uses())
        assert [(node.op_type, index) for node, index in consumers] == [("Expand", 1)]


def test_pruned_lm_head_without_attention_mask_keeps_the_last_column(tmp_path):
    _, session = _make_lm_head_graph(tmp_path, ir.DataType.FLOAT, with_mask=False)
    hidden_states = np.arange(2 * 3 * HIDDEN_SIZE, dtype=np.float32).reshape(2, 3, HIDDEN_SIZE)

    (logits,) = session.run(None, {"hidden_states": hidden_states})

    np.testing.assert_array_equal(logits, hidden_states[:, -1:, :])


def _tiny_lfm2():
    config = transformers.Lfm2Config(
        architectures=["Lfm2ForCausalLM"],
        vocab_size=128,
        hidden_size=4 * HEAD_SIZE,
        # LFM2 rounds the FFN width up to a multiple of 256, so it does not collide with vocab_size.
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=PAD_TOKEN_ID,
    )
    return transformers.Lfm2ForCausalLM(config), load_builder_module("lfm2").LFM2Model


def _tiny_llama():
    config = transformers.LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=128,
        hidden_size=4 * HEAD_SIZE,
        # Not 128: the builder takes the first Linear with vocab_size outputs as the LM head, so an MLP of
        # that width would become the head (see test_the_last_layer_reaches_the_logits).
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=PAD_TOKEN_ID,
    )
    return transformers.LlamaForCausalLM(config), load_builder_module("llama").LlamaModel


TINY_MODELS = {"lfm2": (_tiny_lfm2, "Lfm2ForCausalLM"), "llama": (_tiny_llama, "LlamaForCausalLM")}
# Llama runs on every onnxruntime. The LFM2 conv layers need com.microsoft:CausalConvWithState.
TINY_MODEL_PARAMS = [
    pytest.param(
        "lfm2",
        marks=pytest.mark.skipif(
            not ort_runs_causal_conv_with_state(),
            reason="the installed onnxruntime cannot run com.microsoft:CausalConvWithState",
        ),
    ),
    "llama",
]


def _build_session(tmp_path, hf_model, builder_class, extra_options):
    out_dir = tmp_path / ("pruned" if extra_options else "full")
    out_dir.mkdir()
    model = builder_class(
        hf_model.config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path / "cache"), extra_options
    )
    model.load_weights = lambda input_path: hf_model
    model.make_model("")
    model.save_model(str(out_dir))
    return ort.InferenceSession(str(out_dir / model.filename), providers=["CPUExecutionProvider"])


def _empty_state(session, batch_size):
    state = {}
    for node_arg in session.get_inputs():
        if node_arg.name.startswith("past_key_values."):
            # The last dim is symbolic, so take the head size from the config.
            state[node_arg.name] = np.zeros((batch_size, node_arg.shape[1], 0, HEAD_SIZE), dtype=np.float32)
        elif node_arg.name.startswith("past."):
            state[node_arg.name] = np.zeros((batch_size, *node_arg.shape[1:]), dtype=np.float32)
    return state


def _run(session, input_ids, attention_mask, state):
    """Run one step and return the logits and the state to feed the next step."""
    feeds = {"input_ids": input_ids, "attention_mask": attention_mask, **state}
    output_names = [node_arg.name for node_arg in session.get_outputs()]
    outputs = dict(zip(output_names, session.run(None, feeds), strict=True))
    next_state = {}
    for name in state:
        # past_key_values.1.key -> present.1.key and past.0.conv -> present.0.conv
        next_state[name] = outputs["present." + name.split(".", 1)[1]]
    return outputs["logits"], next_state


@pytest.fixture(scope="module", params=TINY_MODEL_PARAMS)
def model_sessions(request, tmp_path_factory):
    make_model, class_name = TINY_MODELS[request.param]
    if not hasattr(transformers, class_name):
        # The transformers stub has no model classes.
        pytest.skip(f"transformers does not provide {class_name}")

    # Seed the weights without changing the global RNG for later tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model, builder_class = make_model()
    hf_model.eval()
    tmp_path = tmp_path_factory.mktemp(f"{request.param}_prune")
    full = _build_session(tmp_path, hf_model, builder_class, {})
    pruned = _build_session(tmp_path, hf_model, builder_class, {"prune_lm_head": True})
    return full, pruned


def test_the_last_layer_reaches_the_logits(model_sessions):
    # Guards the tiny configs: if an MLP had vocab_size outputs, the builder would take it as the LM head,
    # the later layers would not reach the logits, and the tests below would check much less.
    full, _ = model_sessions
    rng = np.random.default_rng(2)
    _, state = _run(full, rng.integers(1, 128, size=(1, 4)), np.ones((1, 4), dtype=np.int64), _empty_state(full, 1))
    # past_key_values.3.key -> 3
    last_layer_state = max(state, key=lambda name: int(name.split(".")[1]))
    changed_state = {**state, last_layer_state: state[last_layer_state] + 1}

    next_ids = rng.integers(1, 128, size=(1, 1))
    logits, _ = _run(full, next_ids, np.ones((1, 5), dtype=np.int64), state)
    changed_logits, _ = _run(full, next_ids, np.ones((1, 5), dtype=np.int64), changed_state)
    assert np.abs(changed_logits - logits).max() > 1e-3


# The row choice is exact (the head-only tests above use assert_array_equal). Here a MatMul over one row
# is compared with a MatMul over S rows, and the CPU kernels may round the last bits differently.
ATOL = 1e-5


def test_pruned_model_matches_full_logits_on_a_right_padded_batch(model_sessions):
    full, pruned = model_sessions
    assert [node_arg.shape for node_arg in pruned.get_outputs() if node_arg.name == "logits"] == [
        ["batch_size", 1, 128]
    ]

    # Three prompts of unequal length, padded on the right like og.Tokenizer.encode_batch.
    lengths = [5, 8, 3]
    rng = np.random.default_rng(0)
    input_ids = np.full((3, 8), PAD_TOKEN_ID, dtype=np.int64)
    attention_mask = np.zeros((3, 8), dtype=np.int64)
    for row, length in enumerate(lengths):
        input_ids[row, :length] = rng.integers(1, 128, size=length)
        attention_mask[row, :length] = 1

    full_logits, full_state = _run(full, input_ids, attention_mask, _empty_state(full, 3))
    pruned_logits, pruned_state = _run(pruned, input_ids, attention_mask, _empty_state(pruned, 3))

    assert pruned_logits.shape == (3, 1, 128)
    for row, length in enumerate(lengths):
        np.testing.assert_allclose(pruned_logits[row, 0], full_logits[row, length - 1], rtol=0, atol=ATOL)

    # Decode: one token per row. The mask keeps the pad zeros and adds a one.
    next_ids = rng.integers(1, 128, size=(3, 1))
    attention_mask = np.concatenate([attention_mask, np.ones((3, 1), dtype=np.int64)], axis=1)
    full_logits, _ = _run(full, next_ids, attention_mask, full_state)
    pruned_logits, _ = _run(pruned, next_ids, attention_mask, pruned_state)

    np.testing.assert_allclose(pruned_logits[:, 0], full_logits[:, 0], rtol=0, atol=ATOL)


def test_pruned_model_matches_full_logits_on_a_batch_1_continuation(model_sessions):
    full, pruned = model_sessions
    rng = np.random.default_rng(1)
    prompt = rng.integers(1, 128, size=(1, 4))
    chunk = rng.integers(1, 128, size=(1, 3))

    _, full_state = _run(full, prompt, np.ones((1, 4), dtype=np.int64), _empty_state(full, 1))
    _, pruned_state = _run(pruned, prompt, np.ones((1, 4), dtype=np.int64), _empty_state(pruned, 1))

    # Three new tokens after four past ones: the mask covers all seven.
    full_logits, _ = _run(full, chunk, np.ones((1, 7), dtype=np.int64), full_state)
    pruned_logits, _ = _run(pruned, chunk, np.ones((1, 7), dtype=np.int64), pruned_state)

    np.testing.assert_allclose(pruned_logits[:, 0], full_logits[:, -1], rtol=0, atol=ATOL)

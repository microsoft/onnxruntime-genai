# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the dense LFM2 model builder.

LFM2 configs call their RMSNorm epsilon `norm_eps` and have no `rms_norm_eps`, so the base builder's
epsilon falls back to 1e-6 and `LFM2Model` must set the checkpoint's value. Standalone Q/K norm nodes take
the layernorm epsilon; a GroupQueryAttention node that fuses the Q/K norms (CUDA and WebGPU) carries its
own `qk_norm_epsilon` attribute.

In a padded batch, each row must keep the conv state of its own last real tokens, not of the pads. In a paged
export, each packed sequence must keep its own conv state.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from _builder_test_utils import check_node_shapes, load_builder_module, ort_runs_causal_conv_with_state

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
# Extractor runs a part of a graph.
onnx_utils = pytest.importorskip("onnx.utils")
ort = pytest.importorskip("onnxruntime")
if not hasattr(ort, "InferenceSession"):
    # _builder_test_utils puts an empty onnxruntime module in place when the package is missing.
    pytest.skip("onnxruntime is not installed", allow_module_level=True)

lfm2_module = load_builder_module("lfm2")
ir = lfm2_module.ir
LFM2Model = lfm2_module.LFM2Model
LFM2MoEModel = lfm2_module.LFM2MoEModel

transformers = pytest.importorskip("transformers")
# `_builder_test_utils` substitutes a stub for transformers when it is missing, which has none of the
# LFM2 classes these tests build a model from.
pytestmark = pytest.mark.skipif(
    not hasattr(transformers, "Lfm2ForCausalLM"), reason="transformers does not provide the LFM2 model classes"
)


# Prompts use token ids below 256. The vocabulary is larger and differs from every hidden width: the builder
# takes the first Linear with vocab_size outputs as the LM head, and LFM2 rounds its FFN width up to 256.
VOCAB_SIZE = 320


def _tiny_lfm2_config():
    return transformers.Lfm2Config(
        architectures=["Lfm2ForCausalLM"],
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        # Neither LFM2's usual 1e-5 nor the builder's 1e-6 fallback, so no default can match it by chance.
        norm_eps=1e-3,
    )


def _qk_norm_epsilons(graph):
    """The epsilon of every Q/K norm in the graph, keyed by the name of the node that applies it."""
    epsilons = {}
    for node in graph:
        if node.op_type == "GroupQueryAttention" and "qk_norm_epsilon" in node.attributes:
            epsilons[node.name] = node.attributes["qk_norm_epsilon"].value
        elif node.op_type == "SimplifiedLayerNormalization" and re.search("/attn/[qk]_norm/", node.name):
            epsilons[node.name] = node.attributes["epsilon"].value
    return epsilons


@pytest.mark.parametrize(
    "ep, io_dtype, norm_nodes",
    [
        # CPU keeps the Q/K norms as separate nodes, which take the layernorm epsilon.
        ("cpu", ir.DataType.FLOAT, ["q_norm/SimplifiedLayerNormalization", "k_norm/SimplifiedLayerNormalization"]),
        # CUDA and WebGPU fuse both into GroupQueryAttention, which has an epsilon attribute of its own.
        ("cuda", ir.DataType.FLOAT16, ["GroupQueryAttention"]),
        ("webgpu", ir.DataType.FLOAT, ["GroupQueryAttention"]),
    ],
)
def test_lfm2_qk_norms_use_the_checkpoint_norm_eps(ep, io_dtype, norm_nodes):
    config = _tiny_lfm2_config()
    hf_model = transformers.Lfm2ForCausalLM(config).eval()

    model = LFM2Model(config, io_dtype, io_dtype, ep, None, {})
    model.load_weights = lambda input_path: hf_model
    model.make_model("")

    attention_layers = [i for i, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"]
    expected = {f"/model/layers.{i}/attn/{node}": config.norm_eps for i in attention_layers for node in norm_nodes}
    assert _qk_norm_epsilons(model.model.graph) == pytest.approx(expected)


def _tiny_lfm2_moe_config():
    return transformers.Lfm2MoeConfig(
        architectures=["Lfm2MoeForCausalLM"],
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_dense_layers=1,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["conv", "full_attention", "conv", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        norm_eps=1e-3,
    )


# LFM2-MoE shares the conv layers, so its padded batches take the same path.
TINY_MODELS = {
    "lfm2": (_tiny_lfm2_config, "Lfm2ForCausalLM", LFM2Model),
    "lfm2_moe": (_tiny_lfm2_moe_config, "Lfm2MoeForCausalLM", LFM2MoEModel),
}

# The tests below run the export in onnxruntime, which needs com.microsoft:CausalConvWithState.
requires_causal_conv_with_state = pytest.mark.skipif(
    not ort_runs_causal_conv_with_state(),
    reason="the installed onnxruntime cannot run com.microsoft:CausalConvWithState",
)


def _build_tiny_lfm2(extra_options=None, ep="cpu", io_dtype=ir.DataType.FLOAT, kind="lfm2"):
    make_config, class_name, builder_class = TINY_MODELS[kind]
    if not hasattr(transformers, class_name):
        pytest.skip(f"transformers does not provide {class_name}")
    config = make_config()
    # Larger weights than the default 0.02, so a wrong conv state moves the logits by about 1 to 4, not 1e-5.
    config.initializer_range = 0.2
    # Seed the weights without changing the global RNG for later tests.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model = getattr(transformers, class_name)(config).eval()

    model = builder_class(config, io_dtype, io_dtype, ep, None, extra_options or {})
    model.load_weights = lambda input_path: hf_model
    model.make_model("")
    return model, hf_model


def _conv_layers(model):
    return [layer_id for layer_id, layer_type in enumerate(model.layer_types) if layer_type == "conv"]


class Lfm2Runner:
    """Runs a tiny LFM2 export one call at a time and feeds each present state back as the next past state.

    Each run also returns the state CausalConvWithState wrote itself (`op.N.conv`), which the export
    otherwise leaves unused.
    """

    def __init__(self, tmp_path, extra_options=None, kind="lfm2"):
        model, hf_model = _build_tiny_lfm2(extra_options, kind=kind)
        graph = model.model.graph
        self.conv_layers = _conv_layers(model)
        self.op_state_names = {}
        for layer_id in self.conv_layers:
            op_state = graph.node(f"/model/layers.{layer_id}/conv/CausalConvWithState").outputs[1]
            if not op_state.is_graph_output():
                graph.outputs.append(op_state)
            self.op_state_names[op_state.name] = f"op.{layer_id}.conv"
        ir.save(model.model, tmp_path / "model.onnx")
        self.session = ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"])
        # Without the embedding layer the graph takes inputs_embeds, so the runner looks the tokens up itself.
        self.embed_tokens = hf_model.get_input_embeddings().weight.detach().numpy() if model.exclude_embeds else None
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.past_shapes = {
            "past_key_values": (model.num_kv_heads, 0, model.head_size),
            "past": (model.hidden_size, model.conv_L_cache - 1),
        }

    def empty_past(self, batch_size):
        past = {}
        for graph_input in self.session.get_inputs():
            prefix = graph_input.name.split(".")[0]
            if prefix in self.past_shapes:
                past[graph_input.name] = np.zeros((batch_size, *self.past_shapes[prefix]), np.float32)
        return past

    def run(self, input_ids, attention_mask, past):
        """Return the logits, the present state to feed back, and the op's own conv states."""
        input_ids = np.asarray(input_ids, np.int64)
        feeds = {"attention_mask": np.asarray(attention_mask, np.int64)}
        if self.embed_tokens is None:
            feeds["input_ids"] = input_ids
        else:
            feeds["inputs_embeds"] = self.embed_tokens[input_ids]
        outputs = dict(zip(self.output_names, self.session.run(None, {**feeds, **past}), strict=True))
        present = {name: outputs["present." + name.split(".", 1)[1]] for name in past}
        op_states = {short: outputs[name] for name, short in self.op_state_names.items()}
        return outputs["logits"], present, op_states


def _generate_alone(runner, chunks, decode_tokens):
    """Logits of the last prompt token and of each decode step, for one row run on its own.

    The prompt can come in several chunks, like a batch-1 continuation.
    """
    past = runner.empty_past(1)
    length = 0
    for chunk in chunks:
        length += len(chunk)
        logits, past, _ = runner.run([chunk], np.ones((1, length)), past)
    steps = [logits[0, -1]]
    for token in decode_tokens:
        length += 1
        logits, past, _ = runner.run([[token]], np.ones((1, length)), past)
        steps.append(logits[0, -1])
    return steps


def _generate_padded_batch(runner, prompts, decode_tokens, pad_token_id=0):
    """The same logits for each row of one right-padded batch. The mask grows the way genai grows it."""
    batch_size = len(prompts)
    padded_length = max(len(prompt) for prompt in prompts)
    input_ids = np.full((batch_size, padded_length), pad_token_id)
    attention_mask = np.zeros((batch_size, padded_length))
    for row, prompt in enumerate(prompts):
        input_ids[row, : len(prompt)] = prompt
        attention_mask[row, : len(prompt)] = 1

    logits, past, _ = runner.run(input_ids, attention_mask, runner.empty_past(batch_size))
    steps = [[logits[row, len(prompt) - 1]] for row, prompt in enumerate(prompts)]
    for step in range(len(decode_tokens[0])):
        # Each new token adds a 1 to the mask. The pad columns stay 0.
        attention_mask = np.concatenate([attention_mask, np.ones((batch_size, 1))], axis=1)
        logits, past, _ = runner.run([[tokens[step]] for tokens in decode_tokens], attention_mask, past)
        for row in range(batch_size):
            steps[row].append(logits[row, -1])
    return steps


@requires_causal_conv_with_state
# exclude_embeds is how the LFM2-VL and LFM2-Audio decoders are exported.
@pytest.mark.parametrize("kind, exclude_embeds", [("lfm2", False), ("lfm2", True), ("lfm2_moe", False)])
def test_lfm2_right_padded_batch_matches_each_row_run_alone(tmp_path, kind, exclude_embeds):
    runner = Lfm2Runner(tmp_path, {"exclude_embeds": exclude_embeds}, kind=kind)

    rng = np.random.default_rng(0)
    # Every row but the longest gets pads. The 1-token row is shorter than the conv state.
    prompts = [rng.integers(1, 256, size=length).tolist() for length in (5, 9, 1, 7)]
    decode_tokens = [rng.integers(1, 256, size=3).tolist() for _ in prompts]

    batch_steps = _generate_padded_batch(runner, prompts, decode_tokens)
    for row, prompt in enumerate(prompts):
        alone_steps = _generate_alone(runner, [prompt], decode_tokens[row])
        for step, (batch_logits, alone_logits) in enumerate(zip(batch_steps[row], alone_steps, strict=True)):
            np.testing.assert_allclose(batch_logits, alone_logits, rtol=0, atol=1e-5, err_msg=f"row {row}, step {step}")


@requires_causal_conv_with_state
def test_lfm2_prompt_in_chunks_matches_one_prefill(tmp_path):
    runner = Lfm2Runner(tmp_path)

    rng = np.random.default_rng(1)
    prompt = rng.integers(1, 256, size=8).tolist()
    decode_tokens = rng.integers(1, 256, size=3).tolist()

    one_prefill = _generate_alone(runner, [prompt], decode_tokens)
    # The 1-token chunk is shorter than the conv state, so its state also keeps a column of the past.
    in_chunks = _generate_alone(runner, [prompt[:5], prompt[5:6], prompt[6:]], decode_tokens)
    for step, (chunked_logits, one_prefill_logits) in enumerate(zip(in_chunks, one_prefill, strict=True)):
        np.testing.assert_allclose(chunked_logits, one_prefill_logits, rtol=0, atol=1e-5, err_msg=f"step {step}")


@requires_causal_conv_with_state
def test_lfm2_batch_1_keeps_the_op_state(tmp_path):
    # Batch 1 has no pads, so the gathered state must be the op's own state, bit for bit, in prefill,
    # in a continuation and in decode.
    runner = Lfm2Runner(tmp_path)
    rng = np.random.default_rng(2)
    past = runner.empty_past(1)
    length = 0
    for chunk_length in (6, 3, 1, 1):
        length += chunk_length
        _, past, op_states = runner.run([rng.integers(1, 256, size=chunk_length)], np.ones((1, length)), past)
        for name, op_state in op_states.items():
            np.testing.assert_array_equal(past["past" + name.removeprefix("op")], op_state, err_msg=name)


@requires_causal_conv_with_state
def test_lfm2_every_layer_reaches_the_logits(tmp_path):
    # Guards the tiny config: if vocab_size matched an FFN width, the builder would take that layer's FFN
    # as the LM head, and the later layers would not reach the logits.
    runner = Lfm2Runner(tmp_path)
    rng = np.random.default_rng(3)
    _, past, _ = runner.run([rng.integers(1, 256, size=4)], np.ones((1, 4)), runner.empty_past(1))
    next_token = [rng.integers(1, 256, size=1)]
    logits, _, _ = runner.run(next_token, np.ones((1, 5)), past)
    for name in past:
        changed_logits, _, _ = runner.run(next_token, np.ones((1, 5)), {**past, name: past[name] + 1})
        assert np.abs(changed_logits - logits).max() > 1e-3, name


def test_lfm2_conv_state_of_each_row(tmp_path):
    # Runs only the nodes that write present.0.conv, from the conv input and the past state. A row of pads
    # cannot run through the whole model: GroupQueryAttention rejects a row with no real token.
    model, _ = _build_tiny_lfm2()
    ir.save(model.model, tmp_path / "model.onnx")
    (shape_node,) = [node for node in model.model.graph if node.name == "/model/last_real_token/seq_len/Shape"]
    hidden_states = shape_node.inputs[0].name
    # S comes from layer 0's [B, S, H] input. A Shape of the [B, C, S] conv input stops WebGPU graph capture.
    assert hidden_states == "/model/layers.0/operator_layernorm/output_0"
    conv_input = "/model/layers.0/conv/Mul_1/output_0"
    subgraph = onnx_utils.Extractor(onnx.load(tmp_path / "model.onnx")).extract_model(
        ["attention_mask", hidden_states, conv_input, "past.0.conv"], ["present.0.conv"]
    )
    session = ort.InferenceSession(subgraph.SerializeToString(), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(4)
    channels, state_size, sequence_length = model.hidden_size, model.conv_L_cache - 1, 4
    past = rng.standard_normal((4, channels, state_size)).astype(np.float32)
    inputs = rng.standard_normal((4, channels, sequence_length)).astype(np.float32)
    attention_mask = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], np.int64)
    feeds = {
        "attention_mask": attention_mask,
        hidden_states: np.zeros((4, sequence_length, channels), np.float32),
        conv_input: inputs,
        "past.0.conv": past,
    }
    (present,) = session.run(None, feeds)

    columns = np.concatenate([past, inputs], axis=2)
    # A row with no real token keeps its past state.
    np.testing.assert_array_equal(present[0], past[0])
    # Each other row keeps the K - 1 columns that end at its last real token, past columns included.
    for row, real_tokens in [(1, 1), (2, 3), (3, 4)]:
        np.testing.assert_array_equal(present[row], columns[row, :, real_tokens : real_tokens + state_size])


@requires_causal_conv_with_state
def test_lfm2_left_padded_batch_keeps_each_rows_last_columns(tmp_path):
    # With left padding every row ends at the last column, so each row keeps the op's own state, as before
    # this change. Only the conv state is checked: the first real tokens of a left-padded row still see pad
    # columns in their conv window, and attention expects right padding, so the logits are not compared.
    runner = Lfm2Runner(tmp_path)
    rng = np.random.default_rng(5)
    lengths = (5, 2, 3)
    prompts = [rng.integers(1, 256, size=length) for length in lengths]
    input_ids = np.zeros((len(prompts), max(lengths)), np.int64)
    attention_mask = np.zeros(input_ids.shape)
    for row, prompt in enumerate(prompts):
        input_ids[row, -len(prompt) :] = prompt
        attention_mask[row, -len(prompt) :] = 1

    _, present, op_states = runner.run(input_ids, attention_mask, runner.empty_past(len(prompts)))

    for name, op_state in op_states.items():
        np.testing.assert_array_equal(present["past" + name.removeprefix("op")], op_state, err_msg=name)
    # The first conv layer comes before any attention, and each row has at least K - 1 real tokens, so its
    # state also equals the state of that row run alone.
    first_conv = f"past.{runner.conv_layers[0]}.conv"
    for row, prompt in enumerate(prompts):
        _, alone, _ = runner.run([prompt], np.ones((1, len(prompt))), runner.empty_past(1))
        np.testing.assert_allclose(present[first_conv][row], alone[first_conv][0], rtol=0, atol=1e-6)


def _conv_state_producers(graph):
    """The op type of the node that writes each present conv state."""
    return {
        output.name: output.producer().op_type
        for output in graph.outputs
        if re.fullmatch(r"present\.\d+\.conv", output.name)
    }


@pytest.mark.parametrize("kind", sorted(TINY_MODELS))
def test_lfm2_takes_the_conv_state_at_each_rows_last_real_token(kind):
    model, _ = _build_tiny_lfm2({"prune_lm_head": True}, kind=kind)

    graph = model.model.graph
    assert _conv_state_producers(graph) == {"present.0.conv": "GatherElements", "present.2.conv": "GatherElements"}
    # The conv layers and the pruned LM head share one search for the last real token.
    shared_nodes = [node for node in graph if node.name.startswith("/model/last_real_token/")]
    assert [node.name for node in shared_nodes if node.op_type in ("CumSum", "Min")] == [
        "/model/last_real_token/positions/CumSum",
        "/model/last_real_token/Min",
    ]
    assert "/model/last_real_token/Min/output_0" in [value.name for value in graph.node("/lm_head/prune/Sub").inputs]
    # T and S reach the math only as Expand shape inputs, which ORT reads on the CPU. A Shape value fed to a
    # device op (such as Clip's max) would add a copy node, and CUDA and WebGPU graph capture would fail.
    index_nodes = shared_nodes + [node for node in graph if node.name.startswith("/model/conv_state/")]
    assert not [node.name for node in index_nodes if node.op_type == "Clip"]
    shape_nodes = [node for node in index_nodes if node.op_type == "Shape"]
    assert len(shape_nodes) == 2
    for shape_node in shape_nodes:
        assert [(user.op_type, index) for user, index in shape_node.outputs[0].uses()] == [("Expand", 1)]


def test_lfm2_last_real_token_with_a_static_mask(tmp_path):
    # CUDA graph capture runs with a [batch_size, max_length] mask that is zero after the real tokens.
    # GroupQueryAttention on the CPU rejects that mask, so run only the nodes that find the last real token.
    model, _ = _build_tiny_lfm2()
    ir.save(model.model, tmp_path / "model.onnx")
    graph = model.model.graph
    (shape_node,) = [node for node in graph if node.name == "/model/last_real_token/seq_len/Shape"]
    hidden_states = shape_node.inputs[0].name
    positions = "/model/last_real_token/Min/output_0"
    # Extractor skips the ONNX checker, which does not know the com.microsoft ops in the full graph.
    subgraph = onnx_utils.Extractor(onnx.load(tmp_path / "model.onnx")).extract_model(
        ["attention_mask", hidden_states], [positions]
    )
    session = ort.InferenceSession(subgraph.SerializeToString(), providers=["CPUExecutionProvider"])

    def position(attention_mask, sequence_length):
        attention_mask = np.asarray(attention_mask, np.int64)
        feeds = {
            "attention_mask": attention_mask,
            hidden_states: np.zeros((len(attention_mask), sequence_length, model.hidden_size), np.float32),
        }
        return session.run(None, feeds)[0][:, 0].tolist()

    def static_mask(row_ones, max_length):
        return [[1] * ones + [0] * (max_length - ones) for ones in row_ones]

    # First prefill of a right-padded batch: each row's real length. A row of pads only gives 0.
    assert position(static_mask([5, 9, 1, 7, 0], max_length=16), sequence_length=9) == [5, 9, 1, 7, 0]
    # Decode: the mask also counts past tokens, so every row takes its one new token.
    assert position(static_mask([6, 10, 2, 8], max_length=16), sequence_length=1) == [1, 1, 1, 1]
    # Batch-1 continuation of 4 tokens after 8 past tokens.
    assert position(static_mask([12], max_length=16), sequence_length=4) == [4]
    # Left padding, and a zero inside a row (the Qwen-VL mask), end at the last column.
    assert position([[0, 0, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]], sequence_length=4) == [4, 4, 2]


@pytest.mark.parametrize("kind", sorted(TINY_MODELS))
def test_lfm2_paged_conv_layers_keep_each_sequence_apart(kind):
    # CausalConvWithState would run the packed tokens of all sequences as one sequence.
    model, _ = _build_tiny_lfm2({"use_paged_attention": True}, "cuda", ir.DataType.FLOAT16, kind=kind)

    graph = model.model.graph
    assert _conv_state_producers(graph) == {
        "present.0.conv": "VarlenCausalConvWithState",
        "present.2.conv": "VarlenCausalConvWithState",
    }
    for layer_id in _conv_layers(model):
        conv = graph.node(f"/model/layers.{layer_id}/conv/VarlenCausalConvWithState")
        assert [value.name if value is not None else "" for value in conv.inputs] == [
            f"/model/layers.{layer_id}/conv/Mul_1/output_0",
            f"model.layers.{layer_id}.conv.conv.weight",
            "cumulative_sequence_lengths",
            "",
            f"past.{layer_id}.conv",
        ]
        assert conv.attributes["activation"].value == "none"
    assert model.make_decoder_state_groups({}, {}) == [
        {"kind": "paged_kv", "layer_ids": [1, 3]},
        {"kind": "fixed_conv", "layer_ids": [0, 2]},
    ]
    # A rank-3 Transpose of the packed rank-2 tokens fails to load in onnxruntime.
    assert not [node.name for node in graph if "/conv/" in node.name and node.op_type == "Transpose"]
    check_node_shapes(ir.serde.serialize_model(model.model))


@pytest.mark.parametrize("kind", sorted(TINY_MODELS))
def test_lfm2_paged_conv_layers_need_cuda(kind):
    with pytest.raises(NotImplementedError, match="CUDA only"):
        _build_tiny_lfm2({"use_paged_attention": True}, "webgpu", ir.DataType.FLOAT16, kind=kind)

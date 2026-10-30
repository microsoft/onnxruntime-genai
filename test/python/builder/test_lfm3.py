# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM3 model builder.

LFM3 adds to LFM2 sliding-window attention layers, RoPE per attention layer type (a zero-frequency tail
on the full-attention layers), a sigmoid output gate per head and sandwich norms. A tiny export must
match the PyTorch model through a prefill longer than the window and the decode steps after it; that
test needs transformers' LFM3 classes and an onnxruntime that runs CausalConvWithState. The other tests
never run the graph, so they need neither: they build from a stand-in config, and the graph tests from a
stand-in module tree with the module names of the checkpoints' remote code.
"""

from __future__ import annotations

import copy
import importlib.util
import re
import types

import numpy as np
import pytest
from _builder_test_utils import BUILDERS_DIR, load_builder_module, ort_runs_causal_conv_with_state

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
if not hasattr(ort, "InferenceSession"):
    # _builder_test_utils puts an empty onnxruntime module in place when the package is missing.
    pytest.skip("onnxruntime is not installed", allow_module_level=True)

lfm2_module = load_builder_module("lfm2")
ir = lfm2_module.ir
LFM3Model = lfm2_module.LFM3Model

transformers = pytest.importorskip("transformers")

VOCAB_SIZE = 320
SLIDING_WINDOW = 4
LAYER_TYPES = ["conv", "full_attention", "conv", "sliding_attention"]
ROPE_PARAMETERS = {
    "full_attention": {"rope_type": "default", "rope_theta": 1e6, "partial_rotary_factor": 0.5},
    "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
}


def _build_tiny_lfm3(tmp_path, full_attention_rope_type):
    config = transformers.Lfm3Config(
        architectures=["Lfm3ForCausalLM"],
        vocab_size=VOCAB_SIZE,
        # The defaults index LFM3's 128k vocabulary.
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        layer_types=LAYER_TYPES,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=SLIDING_WINDOW,
        rope_parameters={
            **ROPE_PARAMETERS,
            # transformers' own "default" ignores partial_rotary_factor and rotates the whole head.
            "full_attention": {**ROPE_PARAMETERS["full_attention"], "rope_type": "proportional"},
        },
        norm_eps=1e-3,
        # Larger weights than the default 0.02, so a missing gate or norm moves the logits visibly.
        initializer_range=0.2,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model = transformers.Lfm3ForCausalLM(config).eval()

    # Only the export reads the spelling under test.
    builder_config = copy.deepcopy(config)
    builder_config.rope_parameters["full_attention"]["rope_type"] = full_attention_rope_type
    model = LFM3Model(builder_config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})
    model.load_weights = lambda input_path: hf_model
    model.make_model("")
    ir.save(model.model, tmp_path / "model.onnx")
    return ort.InferenceSession(str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]), model, hf_model


@pytest.mark.skipif(
    not hasattr(transformers, "Lfm3ForCausalLM"), reason="transformers does not provide the LFM3 model classes"
)
@pytest.mark.skipif(
    not ort_runs_causal_conv_with_state(),
    reason="the installed onnxruntime cannot run com.microsoft:CausalConvWithState",
)
# The Hub checkpoints spell the zero-tail RoPE of the full-attention layers "default" with a
# partial_rotary_factor, transformers spells it "proportional"; the export must read both the same way.
@pytest.mark.parametrize("full_attention_rope_type", ["default", "proportional"])
def test_lfm3_matches_pytorch_past_the_sliding_window(tmp_path, full_attention_rope_type):
    session, model, hf_model = _build_tiny_lfm3(tmp_path, full_attention_rope_type)
    past = {}
    for graph_input in session.get_inputs():
        if graph_input.name.startswith("past_key_values"):
            past[graph_input.name] = np.zeros((1, model.num_kv_heads, 0, model.head_size), np.float32)
        elif graph_input.name.endswith(".conv"):
            past[graph_input.name] = np.zeros((1, model.hidden_size, model.conv_L_cache - 1), np.float32)
    output_names = [output.name for output in session.get_outputs()]

    rng = np.random.default_rng(0)
    # Three times the window, so the sliding layer drops keys the full layer still attends to.
    tokens = rng.integers(1, 256, size=3 * SLIDING_WINDOW + 4).tolist()
    prompt, decode_tokens = tokens[:-4], tokens[-4:]
    with torch.no_grad():
        expected = hf_model(torch.tensor([tokens])).logits[0].numpy()

    chunks = [prompt] + [[token] for token in decode_tokens]
    length = 0
    for chunk in chunks:
        length += len(chunk)
        feeds = {"input_ids": np.array([chunk], np.int64), "attention_mask": np.ones((1, length), np.int64)}
        outputs = dict(zip(output_names, session.run(None, {**feeds, **past}), strict=True))
        past = {name: outputs["present." + name.split(".", 1)[1]] for name in past}
        np.testing.assert_allclose(
            outputs["logits"][0], expected[length - len(chunk) : length], rtol=0, atol=1e-4, err_msg=f"length {length}"
        )


def _stand_in_config(architecture="Lfm3ForCausalLM", layer_types=LAYER_TYPES, rope_parameters=ROPE_PARAMETERS):
    """The attributes of an LFM3 config.json that the builder reads, at a tiny size."""
    return types.SimpleNamespace(
        architectures=[architecture],
        _name_or_path="LiquidAI/LFM3-1.7B",
        vocab_size=VOCAB_SIZE,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=len(layer_types),
        layer_types=list(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=SLIDING_WINDOW,
        rope_parameters=rope_parameters,
        norm_eps=1e-3,
        conv_L_cache=3,
        tie_word_embeddings=False,
    )


class Lfm3DecoderLayer(torch.nn.Module):
    """A layer with the module names of the checkpoints' remote code. The builder finds layers by this
    class name's suffix and reads their weights by attribute name, without running them."""

    def __init__(self, config, layer_type, use_sandwich_norm):
        super().__init__()
        hidden, intermediate = config.hidden_size, config.intermediate_size
        heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        if layer_type == "conv":
            self.conv = torch.nn.Module()
            self.conv.in_proj = torch.nn.Linear(hidden, 3 * hidden, bias=False)
            self.conv.conv = torch.nn.Conv1d(hidden, hidden, config.conv_L_cache, groups=hidden, bias=False)
            self.conv.out_proj = torch.nn.Linear(hidden, hidden, bias=False)
        else:
            self.self_attn = torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Linear(hidden, heads * head_dim, bias=False)
            self.self_attn.k_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
            self.self_attn.v_proj = torch.nn.Linear(hidden, kv_heads * head_dim, bias=False)
            self.self_attn.o_proj = torch.nn.Linear(heads * head_dim, hidden, bias=False)
            self.self_attn.gate_proj = torch.nn.Linear(hidden, heads, bias=False)
            self.self_attn.q_norm = torch.nn.RMSNorm(head_dim)
            self.self_attn.k_norm = torch.nn.RMSNorm(head_dim)
        self.feed_forward = torch.nn.Module()
        self.feed_forward.w1 = torch.nn.Linear(hidden, intermediate, bias=False)
        self.feed_forward.w3 = torch.nn.Linear(hidden, intermediate, bias=False)
        self.feed_forward.w2 = torch.nn.Linear(intermediate, hidden, bias=False)
        self.operator_norm = torch.nn.RMSNorm(hidden)
        self.ffn_norm = torch.nn.RMSNorm(hidden)
        if use_sandwich_norm:
            self.post_operator_norm = torch.nn.RMSNorm(hidden)
            self.post_ffn_norm = torch.nn.RMSNorm(hidden)


def _stand_in_lfm3(config, use_sandwich_norm):
    hf_model = torch.nn.Module()
    hf_model.model = torch.nn.Module()
    hf_model.model.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    hf_model.model.layers = torch.nn.ModuleList(
        Lfm3DecoderLayer(config, layer_type, use_sandwich_norm) for layer_type in config.layer_types
    )
    hf_model.model.embedding_norm = torch.nn.RMSNorm(config.hidden_size)
    hf_model.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    return hf_model


@pytest.mark.parametrize("use_sandwich_norm", [True, False])
@pytest.mark.parametrize("ep,io_dtype", [("cpu", ir.DataType.FLOAT), ("cuda", ir.DataType.FLOAT16)])
def test_lfm3_graph_wiring(ep, io_dtype, use_sandwich_norm):
    config = _stand_in_config()
    model = LFM3Model(config, io_dtype, io_dtype, ep, None, {})
    model.load_weights = lambda input_path: _stand_in_lfm3(config, use_sandwich_norm)
    model.make_model("")
    graph = model.model.graph
    nodes = {node.name: node for node in graph}

    for layer_id, layer_type in enumerate(LAYER_TYPES):
        if layer_type == "conv":
            continue
        prefix = f"/model/layers.{layer_id}/attn"
        (attention,) = [
            node
            for name, node in nodes.items()
            if name.startswith(f"{prefix}/") and node.op_type == "GroupQueryAttention"
        ]
        # The window applies to the sliding layers only, and each layer rotates with its own type's RoPE caches.
        window = SLIDING_WINDOW if layer_type == "sliding_attention" else -1
        assert attention.attributes["local_window_size"].value == window
        attention_inputs = {value.name for value in attention.inputs if value is not None}
        assert {f"cos_cache_{layer_type}", f"sin_cache_{layer_type}"} <= attention_inputs
        assert not {f"cos_cache_{other}" for other in ROPE_PARAMETERS if other != layer_type} & attention_inputs

        # o_proj reads attention * sigmoid(gate_proj(x)): MatMul -> Sigmoid -> (Reshape) -> Mul -> (Reshape) -> o_proj,
        # with gate_proj reading the same normed input as the Q/K/V projections.
        gate = nodes[f"{prefix}/gate/Mul"]
        sigmoid = gate.inputs[1].producer().inputs[0].producer()
        assert sigmoid.op_type == "Sigmoid"
        assert sigmoid.inputs[0].producer() is nodes[f"{prefix}/gate/MatMul"]
        assert gate.inputs[0].producer().inputs[0].producer() is attention
        assert nodes[f"{prefix}/gate/MatMul"].inputs[0].name == f"/model/layers.{layer_id}/operator_layernorm/output_0"
        assert nodes[f"{prefix}/o_proj/MatMul"].inputs[0].producer().inputs[0].producer() is gate

    assert {name for name in graph.initializers if name.startswith(("cos_cache", "sin_cache"))} == {
        f"{cache}_{layer_type}" for cache in ("cos_cache", "sin_cache") for layer_type in ROPE_PARAMETERS
    }
    post_norms = [name for name in nodes if re.search(r"/post_(operator|ffn)_layernorm/", name)]
    assert len(post_norms) == (2 * len(LAYER_TYPES) if use_sandwich_norm else 0)
    # A sandwich norm reads its block's output and is what joins the residual stream at the next norm.
    for layer_id, layer_type in enumerate(LAYER_TYPES if use_sandwich_norm else []):
        layer = f"/model/layers.{layer_id}"
        block = f"{layer}/conv/out_proj" if layer_type == "conv" else f"{layer}/attn/o_proj"
        next_norm = "final_norm" if layer_id == len(LAYER_TYPES) - 1 else "operator"
        assert nodes[f"{layer}/post_operator_layernorm/LayerNorm"].inputs[0].name == f"{block}/MatMul/output_0"
        assert (
            nodes[f"{layer}/ffn_layernorm/SkipLayerNorm"].inputs[1].name == f"{layer}/post_operator_layernorm/output_0"
        )
        assert (
            nodes[f"/model/layers.{layer_id + 1}/{next_norm}_layernorm/SkipLayerNorm"].inputs[1].name
            == f"{layer}/post_ffn_layernorm/output_0"
        )


@pytest.mark.parametrize("exclude_embeds", [False, True])
def test_lfm3_builds_the_decoder_of_a_multimodal_checkpoint(exclude_embeds):
    # LFM3-VL, LFM3-Audio and LFM3-Omni keep the decoder, and its final norm, under model.language_model.
    config = _stand_in_config("Lfm3VlForConditionalGeneration")
    text_model = _stand_in_lfm3(config, use_sandwich_norm=True)
    hf_model = torch.nn.Module()
    hf_model.model = torch.nn.Module()
    hf_model.model.language_model = text_model.model
    hf_model.lm_head = text_model.lm_head
    model = LFM3Model(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {"exclude_embeds": exclude_embeds})
    model.load_weights = lambda input_path: hf_model
    model.make_model("")

    graph = model.model.graph
    assert {"input_ids", "inputs_embeds"} & {value.name for value in graph.inputs} == {
        "inputs_embeds" if exclude_embeds else "input_ids"
    }
    nodes = {node.name for node in graph}
    assert f"/model/layers.{len(LAYER_TYPES) - 1}/post_ffn_layernorm/LayerNorm" in nodes
    assert f"/model/layers.{len(LAYER_TYPES)}/final_norm_layernorm/SkipLayerNorm" in nodes


@pytest.mark.parametrize(
    "rope_parameters",
    [
        ROPE_PARAMETERS,
        # transformers 5.15.0 adds flat rope_theta and rope_type entries next to the per-type ones.
        {**ROPE_PARAMETERS, "rope_theta": 1e6, "rope_type": "default"},
        # An entry without rope_type is the default RoPE.
        {
            layer_type: {k: v for k, v in params.items() if k != "rope_type"}
            for layer_type, params in ROPE_PARAMETERS.items()
        },
    ],
)
def test_lfm3_reads_rope_parameters_per_layer_type(rope_parameters):
    model = LFM3Model(
        _stand_in_config(rope_parameters=rope_parameters), ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {}
    )

    assert model.rope_cache_names == {
        layer_type: (f"cos_cache_{layer_type}", f"sin_cache_{layer_type}") for layer_type in ROPE_PARAMETERS
    }


def test_lfm3_builds_rope_caches_only_for_its_attention_layer_types():
    config = _stand_in_config(layer_types=["conv", "full_attention", "conv", "full_attention"])
    model = LFM3Model(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})

    assert model.rope_cache_names == {"full_attention": ("cos_cache_full_attention", "sin_cache_full_attention")}


@pytest.mark.parametrize("rope_type", ["default", "proportional"])
def test_lfm3_full_attention_rope_has_a_zero_frequency_tail(rope_type):
    full_attention = {**ROPE_PARAMETERS["full_attention"], "rope_type": rope_type}
    config = _stand_in_config(rope_parameters={**ROPE_PARAMETERS, "full_attention": full_attention})
    model = LFM3Model(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})

    cos, sin = model.make_lfm3_rope_caches("full_attention")

    # Both spellings, as transformers' proportional RoPE: partial_rotary_factor 0.5 of head_dim 16 rotates the
    # first 4 of 8 frequencies, spaced over the full head_dim, and leaves the other 4 at zero.
    inv_freq = torch.cat((1.0 / 1e6 ** (torch.arange(0, 8, 2).float() / 16), torch.zeros(4)))
    angles = torch.outer(torch.arange(model.context_length).float(), inv_freq)
    torch.testing.assert_close(cos, angles.cos())
    torch.testing.assert_close(sin, angles.sin())


@pytest.mark.parametrize(
    "config_kwargs,error,message",
    [
        (
            {"rope_parameters": {"rope_theta": 1e6, "rope_type": "default"}},
            NotImplementedError,
            "per attention layer type",
        ),
        (
            {
                "rope_parameters": {
                    **ROPE_PARAMETERS,
                    "sliding_attention": {"rope_type": "yarn", "rope_theta": 1e4, "factor": 4.0},
                }
            },
            NotImplementedError,
            "sliding_attention RoPE",
        ),
        # transformers 5.15.0 loads the checkpoints' conv layers as linear_attention.
        (
            {"layer_types": ["linear_attention", "full_attention", "conv", "sliding_attention"]},
            ValueError,
            "linear_attention",
        ),
    ],
)
def test_lfm3_refuses_configs_it_cannot_export(config_kwargs, error, message):
    with pytest.raises(error, match=message):
        LFM3Model(_stand_in_config(**config_kwargs), ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})


@pytest.mark.parametrize(
    "ep,io_dtype,error",
    [
        ("cpu", ir.DataType.FLOAT, None),
        ("cuda", ir.DataType.FLOAT16, None),
        ("webgpu", ir.DataType.FLOAT16, None),
        ("cuda", ir.DataType.FLOAT, "selects Attention"),
        ("dml", ir.DataType.FLOAT, "selects Attention"),
        ("cpu", ir.DataType.FLOAT16, "selects MultiHeadAttention"),
        # DML's GroupQueryAttention ignores local_window_size.
        ("dml", ir.DataType.FLOAT16, "DML"),
    ],
)
def test_lfm3_needs_an_attention_op_with_a_sliding_window(ep, io_dtype, error):
    def build():
        return LFM3Model(_stand_in_config(), io_dtype, io_dtype, ep, None, {})

    if error is None:
        assert build().attention_attrs["op_type"] == "GroupQueryAttention"
    else:
        with pytest.raises(NotImplementedError, match=error):
            build()


def test_lfm3_refuses_paged_attention_with_conv_layers():
    # Paged inputs pack the tokens of every sequence into one axis, which the batched conv op would mix.
    with pytest.raises(NotImplementedError, match="conv layers"):
        LFM3Model(
            _stand_in_config(), ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None, {"use_paged_attention": True}
        )


def _load_builder_entrypoint_module():
    spec = importlib.util.spec_from_file_location("models_builder_entrypoint", BUILDERS_DIR.parent / "builder.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "architecture,loader,text_type,pipeline_type",
    [
        # builder.py leaves the architecture as the model type of a text checkpoint, which the genai config
        # shortens to "lfm3".
        ("Lfm3ForCausalLM", "AutoModelForCausalLM", "Lfm3ForCausalLM", "Lfm3ForCausalLM"),
        ("Lfm3VlForConditionalGeneration", "AutoModelForImageTextToText", "lfm3", "lfm3_vl"),
        ("Lfm3AudioForConditionalGeneration", "AutoModelForMultimodalLM", "lfm3", "lfm3_audio"),
        ("Lfm3OmniForConditionalGeneration", "AutoModelForMultimodalLM", "lfm3", "lfm3_omni"),
    ],
)
def test_lfm3_architectures_pick_the_loader_and_the_model_type(
    monkeypatch, tmp_path, architecture, loader, text_type, pipeline_type
):
    builder_module = _load_builder_entrypoint_module()
    captured = {}

    class RecordingLFM3Model(builder_module.LFM3Model):
        def make_genai_config(self, *args):
            self.resolve_transformers_class = lambda class_name: class_name
            captured[self.exclude_embeds] = (self.resolve_model_loader(), self.model_type)

        def save_processing(self, *args):
            pass

    monkeypatch.setattr(builder_module, "LFM3Model", RecordingLFM3Model)
    config = _stand_in_config(architecture)
    for exclude_embeds in (False, True):
        builder_module.create_model(
            config._name_or_path,
            str(tmp_path / "input"),
            str(tmp_path / "output"),
            "fp32",
            "cpu",
            str(tmp_path / "cache"),
            config_only=True,
            exclude_embeds=exclude_embeds,
            hf_details={"extra_kwargs": {}, "hf_name": config._name_or_path, "hf_config": config},
        )

    assert captured == {False: (loader, text_type), True: (loader, pipeline_type)}


@pytest.mark.parametrize(
    "switch_token_ids,existing,expected",
    [
        # LFM3-Audio / LFM3-Omni: <|audio_start|>, <|mixed_start|>, <|text_end|>
        ([125012, 125015, 125014], [124900, 125014], [124900, 125014, 125012, 125015]),
        ([125012, 125015, 125014], 124900, [124900, 125012, 125015, 125014]),
        # LFM3-1.7B / LFM3-VL
        ([], [124900], [124900]),
    ],
)
def test_lfm3_stops_where_the_model_starts_speaking(switch_token_ids, existing, expected):
    # After <|audio_start|> or <|mixed_start|> LFM3-Audio and LFM3-Omni answer in speech, which this
    # runtime cannot generate for LFM3, so generation has to end there.
    builder = LFM3Model.__new__(LFM3Model)
    builder.layer_types = ["conv", "sliding_attention"]
    builder.conv_L_cache = 3
    builder.modality_switch_token_ids = switch_token_ids
    genai_config = {"model": {"decoder": {}, "eos_token_id": existing}}

    builder.update_genai_config(genai_config)

    assert genai_config["model"]["eos_token_id"] == expected
    assert genai_config["model"]["decoder"]["layer_types"] == ["conv", "sliding_attention"]

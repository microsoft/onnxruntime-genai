# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Unit tests for the LFM3 model builder.

LFM3 adds to LFM2 sliding-window attention layers, RoPE per attention layer type (a zero-frequency tail
on the full-attention layers), a sigmoid output gate per head and sandwich norms. A tiny export must
match the PyTorch model through a prefill longer than the window and the decode steps after it.
"""

from __future__ import annotations

import numpy as np
import pytest
from _builder_test_utils import load_builder_module, ort_runs_causal_conv_with_state

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


def _build_tiny_lfm3(tmp_path):
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
        layer_types=["conv", "full_attention", "conv", "sliding_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=SLIDING_WINDOW,
        rope_parameters={
            "full_attention": {"rope_type": "proportional", "rope_theta": 1e6, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
        },
        norm_eps=1e-3,
        # Larger weights than the default 0.02, so a missing gate or norm moves the logits visibly.
        initializer_range=0.2,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        hf_model = transformers.Lfm3ForCausalLM(config).eval()

    model = LFM3Model(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})
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
def test_lfm3_matches_pytorch_past_the_sliding_window(tmp_path):
    session, model, hf_model = _build_tiny_lfm3(tmp_path)
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

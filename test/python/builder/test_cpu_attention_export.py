# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
import transformers
from _builder_test_utils import load_builder_module

LlamaModel = load_builder_module("llama").LlamaModel
SmolLM3Model = load_builder_module("smollm").SmolLM3Model


@pytest.mark.parametrize("head_size", [4, 8, 12, 24])
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_cpu_llama_prefill_and_decode_match_pytorch(tmp_path, head_size, kv_heads):
    config = transformers.LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=101,
        hidden_size=4 * head_size,
        intermediate_size=8 * head_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        head_dim=head_size,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    reference = transformers.LlamaForCausalLM(config).eval()
    model = LlamaModel(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})
    model.load_weights = lambda _: reference
    model.make_model("")
    path = tmp_path / "model.onnx"
    ir.save(model.model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = [output.name for output in session.get_outputs()]
    past = {
        item.name: np.zeros((1, kv_heads, 0, head_size), dtype=np.float32)
        for item in session.get_inputs()
        if item.name.startswith("past_key_values")
    }
    tokens = [3, 5, 7, 9]
    with torch.no_grad():
        expected = reference(torch.tensor([tokens])).logits[0].numpy()
    length = 0
    for chunk in [tokens[:3], tokens[3:]]:
        positions = np.arange(length, length + len(chunk), dtype=np.int64)[None, :]
        length += len(chunk)
        feeds = {"input_ids": np.array([chunk], dtype=np.int64), "attention_mask": np.ones((1, length), np.int64)}
        if any(item.name == "position_ids" for item in session.get_inputs()):
            feeds["position_ids"] = positions
        result = dict(zip(outputs, session.run(None, {**feeds, **past}), strict=True))
        past = {name: result["present." + name.split(".", 1)[1]] for name in past}
        np.testing.assert_allclose(result["logits"][0], expected[length - len(chunk) : length], rtol=0, atol=1e-4)


@pytest.mark.parametrize("head_size", [16, 24])
def test_cpu_smollm3_layer_rope_selection_matches_pytorch(tmp_path, head_size):
    config = transformers.SmolLM3Config(
        architectures=["SmolLM3ForCausalLM"],
        vocab_size=101,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=4 * head_size,
        intermediate_size=8 * head_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_size,
        max_position_embeddings=32,
        no_rope_layers=[0, 1],
        layer_types=["full_attention", "full_attention"],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    reference = transformers.SmolLM3ForCausalLM(config).eval()
    model = SmolLM3Model(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", None, {})
    model.load_weights = lambda _: reference
    model.make_model("")
    path = tmp_path / "model.onnx"
    ir.save(model.model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    tokens = [3, 5, 7]
    feeds = {"input_ids": np.array([tokens], np.int64), "attention_mask": np.ones((1, len(tokens)), np.int64)}
    for item in session.get_inputs():
        if item.name.startswith("past_key_values"):
            feeds[item.name] = np.zeros((1, 2, 0, head_size), dtype=np.float32)
        elif item.name == "position_ids":
            feeds[item.name] = np.arange(len(tokens), dtype=np.int64)[None, :]
    with torch.no_grad():
        expected = reference(torch.tensor([tokens])).logits.numpy()
    actual = session.run(["logits"], feeds)[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

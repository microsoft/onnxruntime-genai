# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CPU-executed equivalence checks, not substitutes for required GPU execution."""

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import test_multimodal_turns as turns
from create.create_multimodal_gqa_model import _media_models
from create.create_multimodal_gqa_model import create_model as create_gqa_model
from create.create_multimodal_turn_test_model import (
    IMAGE_TOKEN_ID,
    QWEN_FAMILIES,
    VOCAB_SIZE,
    create_model,
    make_decoder_model,
    make_embedding_model,
)
from onnx import TensorProto as T
from onnx.reference import ReferenceEvaluator

FAMILIES = ("phi3v", "mistral3", *QWEN_FAMILIES, "gemma3", "llama")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("chunks", [(192,), (1, 7, 8, 16, 31, 64, 65)], ids=["prefill", "dynamic-cache"])
def test_webgpu_decoder_matches_integer_fixture_and_independent_oracle(family, chunks):
    count = sum(chunks)
    ids = np.arange(count, dtype=np.int32)[None, :] % 16 + 2
    # Include negative and fractional history to detect floor-vs-truncation mistakes.
    embeds = np.random.default_rng(42).integers(-2048, 2048, (1, count, 1)).astype(np.float32) / 4
    position_ids = np.arange(count, dtype=np.int64)[None, :]
    if family in QWEN_FAMILIES:
        position_ids = np.stack([position_ids, position_ids // 2, position_ids // 3])
        positions = (position_ids * np.array([1, 2, 4])[:, None, None]).sum(axis=0)
    else:
        positions = 7 * position_ids
    if family == "llama":
        embeds = (2 * ids[..., None] + 3).astype(np.float32)
    positions = positions[..., None].astype(np.float32)
    keys = embeds + positions
    values = 2 * embeds + 3 * positions + 1
    history = np.cumsum(keys, axis=1) + np.cumsum(values, axis=1)
    target = history.astype(np.int32) % 16 + 2
    expected_logits = (-((np.arange(VOCAB_SIZE) - target) ** 2) + history / 1024).astype(np.float32)
    expected_keys, expected_values = keys[:, None, :, :], values[:, None, :, :]
    models = [make_decoder_model(family, device=device) for device in ("cpu", "webgpu")]
    runners = [
        ReferenceEvaluator(model)
        if reference
        else ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        for model in models
        for reference in (True, False)
    ]
    for runner in runners:
        past_key = past_value = np.empty((1, 1, 0, 1), dtype=np.float32)
        start = 0
        for width in chunks:
            end = start + width
            feeds = {
                "input_ids" if family == "llama" else "inputs_embeds": (
                    ids[:, start:end] if family == "llama" else embeds[:, start:end]
                ),
                "position_ids": position_ids[..., start:end],
                "attention_mask": np.ones((1, end), dtype=np.int64),
                "past_key_values.0.key": past_key,
                "past_key_values.0.value": past_value,
            }
            logits, past_key, past_value = runner.run(None, feeds)
            np.testing.assert_array_equal(logits, expected_logits[:, start:end])
            np.testing.assert_array_equal(past_key, expected_keys[:, :, :end])
            np.testing.assert_array_equal(past_value, expected_values[:, :, :end])
            start = end


@pytest.mark.parametrize("family", FAMILIES)
def test_webgpu_fixture_preserves_media_and_dynamic_cache_contracts(tmp_path, family):
    directories = {device: create_model(tmp_path / device, family, device=device) for device in ("cpu", "webgpu")}
    graphs = {device: onnx.load(path / "decoder.onnx") for device, path in directories.items()}
    for field in ("input", "output"):
        assert list(getattr(graphs["cpu"].graph, field)) == list(getattr(graphs["webgpu"].graph, field))
    portable = graphs["webgpu"]
    assert all(node.op_type != "Mod" for node in portable.graph.node)
    assert sum(node.op_type == "Floor" for node in portable.graph.node) == 1
    assert sum(node.op_type == "CumSum" for node in portable.graph.node) == 2
    assert sum(node.op_type == "Concat" for node in portable.graph.node) == 2
    assert next(x for x in portable.graph.input if x.name == "position_ids").type.tensor_type.elem_type == T.INT64
    # No large lookup table, fixed-size cache, or conversation-output constants.
    assert len(portable.SerializeToString()) < 8192
    if family != "llama":
        assert (directories["cpu"] / "vision.onnx").read_bytes() == (directories["webgpu"] / "vision.onnx").read_bytes()
        embeddings = {
            device: onnx.shape_inference.infer_shapes(onnx.load(path / "embedding.onnx"))
            for device, path in directories.items()
        }
        for field in ("input", "output"):
            assert list(getattr(embeddings["cpu"].graph, field)) == list(getattr(embeddings["webgpu"].graph, field))
        for device, tensor_type in (("cpu", T.INT32), ("webgpu", T.FLOAT)):
            mask = next(x for x in embeddings[device].graph.value_info if x.name == "image_mask")
            assert mask.type.tensor_type.elem_type == tensor_type


@pytest.mark.parametrize("family", FAMILIES[:-1])
@pytest.mark.parametrize("variant", ["arithmetic", "gqa-fp32", "gqa-fp16"])
@pytest.mark.parametrize("images", ["empty", "mixed", "all"])
def test_webgpu_embedding_counts_match_integer_fixture_and_oracle(family, variant, images):
    dtype = np.float16 if variant == "gqa-fp16" else np.float32
    models = [
        make_embedding_model(family, device=device)
        if variant == "arithmetic"
        else _media_models(family, dtype, device=device)[1]
        for device in ("cpu", "webgpu")
    ]
    width = 1 if variant == "arithmetic" else 64
    image_mask = np.zeros((1, 256), dtype=bool)
    if images == "all":
        image_mask[:] = True
    elif images == "mixed":
        image_mask[:, 1::3] = True
    ids = np.where(image_mask, -1 if family == "phi3v" else IMAGE_TOKEN_ID, 0).astype(np.int32)
    features = (np.arange(int(image_mask.sum()) * width).reshape(-1, width) / 8).astype(dtype)
    expected = np.full((1, 256, width), 3 if variant == "arithmetic" else 0.03, dtype=dtype)
    expected[image_mask] = features
    for model in models:
        for runner in (
            ReferenceEvaluator(model),
            ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]),
        ):
            actual = runner.run(None, {"input_ids": ids, "image_features": features})[0]
            np.testing.assert_array_equal(actual, expected)


def test_turn_factory_selects_webgpu_graph_without_cpu_fallback(tmp_path, monkeypatch):
    captured = {}

    class Model:
        device_type = "WebGPU"

        def __init__(self, path):
            captured["config"] = json.loads((tmp_path / "phi3v" / "genai_config.json").read_text(encoding="utf-8"))
            captured["decoder"] = onnx.load(tmp_path / "phi3v" / "decoder.onnx")

    monkeypatch.setattr(turns.og, "Model", Model)
    make = turns.model_factory.__wrapped__(tmp_path, "webgpu")
    make("phi3v", asynchronous=True)
    assert all(node.op_type != "Mod" for node in captured["decoder"].graph.node)
    for role in ("vision", "embedding", "decoder"):
        config = captured["config"]["model"][role]
        assert config["session_options"] == {
            "provider_options": [
                {
                    "webgpu": {
                        "device_filtering_options": {"hardware_device_type": "gpu"},
                        "enableInt64": "1",
                    }
                }
            ],
            "session.disable_cpu_ep_fallback": "1",
        }
        assert config["run_options"] == {"disable_synchronize_execution_providers": "1"}


@pytest.mark.parametrize("dtype", ["fp32", "fp16"])
@pytest.mark.parametrize(
    "shared,capture", [(False, False), (True, False), (True, True)], ids=["dynamic", "shared", "captured"]
)
def test_webgpu_gqa_fixture_enables_position_casts_without_media_fallback(tmp_path, dtype, shared, capture):
    directory = create_gqa_model(tmp_path, device="webgpu", dtype=dtype, shared=shared, capture=capture)
    config = json.loads((directory / "genai_config.json").read_text(encoding="utf-8"))
    assert config["search"]["past_present_share_buffer"] == shared
    for role in ("vision", "embedding", "decoder"):
        session = config["model"][role]["session_options"]
        assert session["provider_options"] == [
            {
                "webgpu": {
                    "device_filtering_options": {"hardware_device_type": "gpu"},
                    "enableInt64": "1",
                    "enableGraphCapture": "1" if capture and role == "decoder" else "0",
                }
            }
        ]
        assert session["session.disable_cpu_ep_fallback"] == ("0" if role == "decoder" else "1")
    decoder = onnx.load(directory / "decoder.onnx")
    assert any(node.op_type == "GroupQueryAttention" for node in decoder.graph.node)
    assert all(node.op_type != "Concat" for node in decoder.graph.node)

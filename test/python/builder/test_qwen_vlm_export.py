# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
from types import SimpleNamespace

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from models.builders.expansions import trt_rtx_qwen35_vlm_export as vlm_export
from models.builders.expansions.trt_rtx_qwen35_vlm_export import Qwen35VLMModel


def _session(builder):
    proto = ir.to_proto(builder.model)
    onnx.checker.check_model(proto)
    assert not {node.op_type for node in proto.graph.node} & {"Loop", "NonZero", "ScatterND"}
    return ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("ids", [[[1, 2, 3]], [[1, 31, 31, 2]], [[31, 1, 2], [3, 31, 31]]])
def test_embedding_matches_placeholder_replacement(tmp_path, monkeypatch, ids):
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: pytest.fail("PyTorch ONNX export was called"))
    config = SimpleNamespace(image_token_id=31, text_config=SimpleNamespace(hidden_size=8))
    weight = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    builder = Qwen35VLMModel(
        config, {"embed_tokens.weight": weight}, ir.DataType.FLOAT, "embedding.onnx", str(tmp_path)
    )
    builder.make_embedding()
    ids = np.asarray(ids, dtype=np.int64)
    mask = ids == config.image_token_id
    features = np.arange(mask.sum() * 8, dtype=np.float32).reshape(-1, 8) + 1000
    expected = weight.numpy()[ids].copy()
    expected[mask] = features
    actual = _session(builder).run(None, {"input_ids": ids, "image_features": features})[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("grid", [[[1, 4, 6]], [[1, 4, 4], [1, 2, 6]], [[2, 4, 4]]])
def test_vision_matches_hugging_face_with_dynamic_grids(tmp_path, monkeypatch, grid):
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: pytest.fail("PyTorch ONNX export was called"))
    torch.manual_seed(42)
    config = Qwen3_5VisionConfig(
        depth=2,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=8,
        num_position_embeddings=16,
    )
    reference = Qwen3_5VisionModel(config).eval()
    reference.config._attn_implementation = "eager"
    builder = Qwen35VLMModel(
        SimpleNamespace(vision_config=config), reference.state_dict(), ir.DataType.FLOAT, "vision.onnx", str(tmp_path)
    )
    builder.make_vision()
    grid = torch.tensor(grid, dtype=torch.int64)
    pixels = torch.randn(int(grid.prod(-1).sum()), 3 * config.temporal_patch_size * config.patch_size**2)
    with torch.no_grad():
        expected = reference(pixels, grid).pooler_output.numpy()
    actual = _session(builder).run(None, {"pixel_values": pixels.numpy(), "image_grid_thw": grid.numpy()})[0]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)


def test_export_keeps_processor_within_trt_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: pytest.fail("PyTorch ONNX export was called"))
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=8,
        num_position_embeddings=16,
    )
    config = SimpleNamespace(
        image_token_id=31,
        vision_start_token_id=30,
        text_config=SimpleNamespace(hidden_size=8),
        vision_config=vision_config,
    )
    state = {f"visual.{name}": tensor for name, tensor in Qwen3_5VisionModel(vision_config).state_dict().items()}
    state["embed_tokens.weight"] = torch.zeros(32, 8)
    monkeypatch.setattr(vlm_export, "load_qwen35_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(vlm_export, "_load_qwen35_aux_state", lambda *args: state)
    (tmp_path / "genai_config.json").write_text(json.dumps({"model": {}}))
    vlm_export.export_qwen35_vlm_components("unused", str(tmp_path), None, False, "trt-rtx", ir.DataType.FLOAT)
    processor = json.loads((tmp_path / "processor_config.json").read_text())
    resize = next(
        t["operation"]["attrs"] for t in processor["processor"]["transforms"] if t["operation"]["type"] == "Resize"
    )
    model = json.loads((tmp_path / "genai_config.json").read_text())
    options = model["model"]["vision"]["session_options"]["provider_options"][0]["NvTensorRtRtx"]
    max_patches = int(options["nv_profile_max_shapes"].split(":")[1].split("x")[0])
    assert resize["max_pixels"] == max_patches * vision_config.patch_size**2
    assert model["search"]["past_present_share_buffer"]
    assert options["enable_cuda_graph"] == "0"
    for filename in ("embedding.onnx", "vision.onnx"):
        onnx.checker.check_model(str(tmp_path / filename))

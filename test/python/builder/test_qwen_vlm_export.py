# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from models.builders import qwen_vlm
from models.builders.qwen import Qwen35Model, Qwen35MoEModel
from models.builders.qwen_vlm import Qwen35EmbeddingModel, Qwen35VisionModel


def _config(vision_config=None, moe=False):
    config_class = Qwen3_5MoeConfig if moe else Qwen3_5Config
    return config_class(
        architectures=["Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"],
        image_token_id=31,
        vision_start_token_id=30,
        text_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": 32,
            "max_position_embeddings": 128,
            "layer_types": ["full_attention"],
            "mtp_num_hidden_layers": 0,
        },
        vision_config=(
            vision_config
            or Qwen3_5VisionConfig(
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
        ).to_dict(),
    )


def _session(builder):
    proto = ir.to_proto(builder.model)
    onnx.checker.check_model(proto, full_check=True)
    assert not {node.op_type for node in proto.graph.node} & {"Loop", "NonZero", "ScatterND"}
    return ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("ids", [[[1, 2, 3]], [[1, 31, 31, 2]], [[31, 1, 2], [3, 31, 31]]])
def test_embedding_matches_placeholder_replacement(tmp_path, monkeypatch, ids):
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: pytest.fail("PyTorch ONNX export was called"))
    config = _config()
    weight = torch.arange(32 * 8, dtype=torch.float32).reshape(32, 8)
    builder = Qwen35EmbeddingModel(config, ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path), {})
    monkeypatch.setattr(builder, "load_weights", lambda _: {"weight": weight})
    builder.make_model("")
    ids = np.asarray(ids, dtype=np.int64)
    mask = ids == config.image_token_id
    features = np.arange(mask.sum() * 8, dtype=np.float32).reshape(-1, 8) + 1000
    expected = weight.numpy()[ids].copy()
    expected[mask] = features
    actual = _session(builder).run(None, {"input_ids": ids, "image_features": features})[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("ep", ["cpu", "cuda", "trt-rtx"])
@pytest.mark.parametrize("moe", [False, True])
def test_vision_dimensions_are_set_before_attention_initialization(tmp_path, monkeypatch, ep, moe):
    config = _config(moe=moe)
    config.text_config.hidden_size = 24
    config.text_config.num_attention_heads = 3
    config.text_config.head_dim = 8
    original_config = config.to_dict()
    dimensions = []
    initialize_attention = qwen_vlm.Model.make_attention_init

    def record_attention_dimensions(self, config):
        dimensions.append((self.num_attn_heads, self.head_size))
        return initialize_attention(self, config)

    monkeypatch.setattr(qwen_vlm.Model, "make_attention_init", record_attention_dimensions)
    builder = Qwen35VisionModel(config, ir.DataType.FLOAT, ir.DataType.FLOAT, ep, str(tmp_path), {})

    expected = (config.vision_config.num_heads, config.vision_config.hidden_size // config.vision_config.num_heads)
    assert dimensions == [expected]
    assert (builder.num_attn_heads, builder.head_size) == expected
    assert builder.attention_attrs["scale"] == pytest.approx(expected[1] ** -0.5)
    assert config.to_dict() == original_config


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
    builder = Qwen35VisionModel(_config(config), ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path), {})
    monkeypatch.setattr(builder, "load_weights", lambda _: reference.state_dict())
    builder.make_model("")
    grid = torch.tensor(grid, dtype=torch.int64)
    pixels = torch.randn(int(grid.prod(-1).sum()), 3 * config.temporal_patch_size * config.patch_size**2)
    with torch.no_grad():
        expected = reference(pixels, grid).pooler_output.numpy()
    actual = _session(builder).run(None, {"pixel_values": pixels.numpy(), "image_grid_thw": grid.numpy()})[0]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("ep", ["cuda", "trt-rtx"])
@pytest.mark.parametrize("moe", [False, True])
def test_parent_exports_all_components_and_provider_options(tmp_path, monkeypatch, ep, moe):
    monkeypatch.setattr(torch.onnx, "export", lambda *args, **kwargs: pytest.fail("PyTorch ONNX export was called"))
    config = _config(moe=moe)
    vision_config = config.vision_config
    parent_class = Qwen35MoEModel if moe else Qwen35Model
    parent = parent_class(
        config,
        ir.DataType.FLOAT,
        ir.DataType.INT4,
        ep,
        str(tmp_path),
        {
            "exclude_embeds": True,
            "enable_cuda_graph": True,
        },
    )
    monkeypatch.setattr(parent.embedding, "load_weights", lambda _: {"weight": torch.zeros(32, 8)})
    monkeypatch.setattr(parent.vision, "load_weights", lambda _: Qwen3_5VisionModel(vision_config).state_dict())
    decoder_calls = []
    monkeypatch.setattr(parent.decoder, "make_model", lambda _: decoder_calls.append("build"))
    monkeypatch.setattr(parent.decoder, "save_model", lambda _: decoder_calls.append("save"))
    monkeypatch.setattr(
        parent.decoder,
        "make_genai_config",
        lambda *_: (tmp_path / "genai_config.json").write_text(
            json.dumps({"model": {"decoder": {"filename": "text.onnx"}}, "search": {"past_present_share_buffer": True}})
        ),
    )
    monkeypatch.setattr(parent.decoder, "save_processing", lambda *_: decoder_calls.append("processing"))
    parent.make_model("")
    parent.save_model(str(tmp_path))
    parent.make_genai_config(config, {}, str(tmp_path))
    parent.save_processing("unused", {}, str(tmp_path))
    assert decoder_calls == ["build", "save", "processing"]
    processor = json.loads((tmp_path / "processor_config.json").read_text())
    resize = next(
        t["operation"]["attrs"] for t in processor["processor"]["transforms"] if t["operation"]["type"] == "Resize"
    )
    model = json.loads((tmp_path / "genai_config.json").read_text())
    assert model["search"]["past_present_share_buffer"]
    assert model["model"]["type"] == config.model_type
    assert parent.decoder.onnx_dtype == ir.DataType.INT4
    assert parent.embedding.onnx_dtype == parent.vision.onnx_dtype == ir.DataType.FLOAT
    if ep == "trt-rtx":
        options = model["model"]["vision"]["session_options"]["provider_options"][0]["NvTensorRtRtx"]
        max_patches = int(options["nv_profile_max_shapes"].split(":")[1].split("x")[0])
        assert resize["max_pixels"] == max_patches * vision_config.patch_size**2
        assert options["enable_cuda_graph"] == "0"
        assert parent.decoder.ep_attrs[ep]["enable_cuda_graph"] == "1"
    else:
        assert model["model"]["vision"]["session_options"]["provider_options"] == [{"cuda": {"enable_cuda_graph": "0"}}]
        assert parent.decoder.ep_attrs[ep]["enable_cuda_graph"] == "1"
        assert resize["max_pixels"] == 16777216
    for filename in ("embedding.onnx", "vision.onnx"):
        onnx.checker.check_model(str(tmp_path / filename))


@pytest.mark.parametrize("moe", [False, True])
def test_decoder_only_export_does_not_build_auxiliary_graphs(tmp_path, moe):
    parent_class = Qwen35MoEModel if moe else Qwen35Model
    parent = parent_class(_config(moe=moe), ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path), {})
    assert not parent.exclude_embeds
    assert parent.embedding is None
    assert parent.vision is None


@pytest.mark.parametrize("ep", ["cuda", "trt-rtx"])
@pytest.mark.parametrize("exclude_embeds,use_qdq", [(True, True), (True, False), (False, True)])
def test_qdq_decoder_does_not_quantize_rotary_gathers(tmp_path, ep, exclude_embeds, use_qdq):
    parent = Qwen35Model(
        _config(),
        ir.DataType.FLOAT16,
        ir.DataType.INT4,
        ep,
        str(tmp_path),
        {
            "exclude_embeds": exclude_embeds,
            "shared_embeddings": True,
            "use_qdq": use_qdq,
            "op_types_to_quantize": ("MatMul", "Gather"),
        },
    )
    expected = ("MatMul",) if exclude_embeds and use_qdq else ("MatMul", "Gather")
    assert parent.decoder.quant_attrs["op_types_to_quantize"] == expected


@pytest.mark.parametrize("value", [0, [0], [1, 2]])
def test_standard_constants_preserve_tensor_shape(tmp_path, value):
    builder = Qwen35EmbeddingModel(_config(), ir.DataType.FLOAT, ir.DataType.FLOAT, "cpu", str(tmp_path), {})
    name = f"/model/constants/INT64/{value}"
    builder.make_constant(name)
    assert list(builder.values[name].shape) == list(np.asarray(value).shape)


def test_component_loader_uses_official_config_and_only_requested_weights(tmp_path):
    from safetensors.torch import save_file
    from transformers import AutoConfig

    _config().save_pretrained(tmp_path)
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.ones(32, 8),
            "model.language_model.layers.0.weight": torch.zeros(8, 8),
            "model.visual.pos_embed.weight": torch.zeros(16, 16),
        },
        str(tmp_path / "model.safetensors"),
    )
    assert isinstance(AutoConfig.from_pretrained(tmp_path), Qwen3_5Config)
    weights = qwen_vlm.load_qwen35_component_weights(str(tmp_path), "model.visual.", None, False)
    assert list(weights) == ["pos_embed.weight"]

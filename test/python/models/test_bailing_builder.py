# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Small, offline regressions for Bailing checkpoint loading and graph export."""

import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from safetensors.torch import save_file
from transformers import PretrainedConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

sys.path.insert(0, str(Path(__file__).parents[3] / "src" / "python" / "py" / "models"))

import builder
from builders import bailing
from builders.bailing import BailingMoeV3Model, _BailingWeights
from builders.bailing_vl import BailingVLModel, BailingVLTextModel, BailingVLVisualModel
from builders.qwen3_8.vision import save_image_processor
from loaders.base import TensorModule
from loaders.vision import load_bailing_visual


def _config(**overrides):
    config = PretrainedConfig(
        architectures=["BailingMoeV3ForCausalLM"],
        _name_or_path="test/bailing",
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        head_dim=64,
        rope_theta=10000.0,
    )
    config.update(overrides)
    return config


def _model(
    tmp_path,
    *,
    dtype=ir.DataType.FLOAT,
    precision=None,
    ep="cpu",
    quant_method=None,
    model_class=BailingMoeV3Model,
    config_overrides=None,
    **options,
):
    config = _config(**(config_overrides or {}))
    if quant_method:
        config.quantization_config = {"quant_method": quant_method}
    return model_class(config, dtype, precision or dtype, ep, str(tmp_path / "cache"), options)


@pytest.mark.parametrize("text_only", [False, True])
def test_ling_composite_normalizes_decoder_without_losing_native_fp8_metadata(tmp_path, text_only):
    text = _config(q_lora_rank=None)
    config = PretrainedConfig(
        _name_or_path="test/ling-vl",
        text_config=text.to_dict(),
        vision_config={"out_hidden_size": 4096},
        tie_word_embeddings=False,
        mrope_section=[8, 12, 12],
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["gate"],
        },
    )
    model = BailingVLModel(
        config,
        ir.DataType.BFLOAT16,
        ir.DataType.BFLOAT16,
        "cuda",
        str(tmp_path),
        {"use_paged_attention": True, "exclude_embeds": not text_only, "text_only": text_only},
    )
    decoder = model.decoder
    assert decoder.config.quantization_config == config.quantization_config
    assert decoder.config.quantization_config is not config.quantization_config
    assert decoder.config.name_or_path == "test/ling-vl"
    assert decoder.q_lora_rank is None
    assert not decoder.config.tie_word_embeddings
    assert decoder.rope_attrs["interleaved"] == 1
    assert decoder.rope_attrs["op_type"] == ("RotaryEmbedding" if text_only else "MRotaryEmbedding")
    assert decoder.model_type == ("bailing_hybrid" if text_only else "bailing_moe_v3_vl")
    assert "mrope_section" not in config.text_config
    assert model.config.vision_config.out_hidden_size == 4096


@pytest.mark.parametrize(
    "options",
    [
        {"use_paged_attention": False},
        {"exclude_embeds": False},
        {"text_only": True},
        {"dflash2_path": "drafter"},
        {"filename": "vision.onnx"},
    ],
)
def test_ling_composite_rejects_unsupported_component_contracts(tmp_path, options):
    config = PretrainedConfig(text_config=_config(), vision_config=PretrainedConfig(), mrope_section=[8, 12, 12])
    with pytest.raises(ValueError, match="Ling"):
        BailingVLModel(
            config,
            ir.DataType.BFLOAT16,
            ir.DataType.BFLOAT16,
            "cuda",
            str(tmp_path),
            {"use_paged_attention": True, "exclude_embeds": True, **options},
        )


@pytest.mark.parametrize("text_only", [False, True])
def test_ling_raw_configuration_preserves_source_activation_default(tmp_path, text_only):
    text = _config().to_dict()
    del text["hidden_act"]
    config = {
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "model_type": "bailing_moe_v3_vl",
        "text_config": text,
        "vision_config": {"out_hidden_size": 4096},
        "mrope_section": [8, 12, 12],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = BailingVLModel(
        BailingVLModel.load_config(str(tmp_path)),
        ir.DataType.BFLOAT16,
        ir.DataType.BFLOAT16,
        "cuda",
        str(tmp_path),
        {"use_paged_attention": True, "exclude_embeds": not text_only, "text_only": text_only},
    )
    assert model.decoder.config.hidden_act == "silu"


def test_ling_raw_configuration_and_image_processor_assets(tmp_path):
    config = {
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "model_type": "bailing_moe_v3_vl",
        "text_config": {"hidden_size": 2560},
        "vision_config": {"out_hidden_size": 4096},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = BailingVLModel.load_config(str(tmp_path))
    assert loaded.text_config == config["text_config"]
    assert loaded.name_or_path == str(tmp_path)
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "min_pixels": 4096,
                "max_pixels": 4194304,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
            }
        )
    )
    save_image_processor(str(tmp_path), {}, str(tmp_path), None)
    processor = json.loads((tmp_path / "processor_config.json").read_text())["processor"]
    operations = {transform["operation"]["type"]: transform["operation"] for transform in processor["transforms"]}
    assert operations["Resize"]["attrs"]["min_pixels"] == 4096
    assert operations["Resize"]["attrs"]["max_pixels"] == 4194304
    assert operations["PatchImage"]["attrs"] == {"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2}


@pytest.mark.parametrize("text_only", [False, True])
def test_ling_composite_config_preserves_component_and_position_contracts(tmp_path, text_only):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 2, "pad_token_id": 0}))
    config = PretrainedConfig(
        _name_or_path=str(tmp_path),
        text_config=_config(bos_token_id=1, eos_token_id=2, pad_token_id=0),
        vision_config={"patch_size": 16, "spatial_merge_size": 2, "out_hidden_size": 4096},
        mrope_section=[8, 12, 12],
        image_token_id=29,
        vision_start_token_id=30,
    )
    model = BailingVLModel(
        config,
        ir.DataType.BFLOAT16,
        ir.DataType.BFLOAT16,
        "cuda",
        str(tmp_path),
        {"use_paged_attention": True, "exclude_embeds": not text_only, "text_only": text_only},
    )
    model.decoder.make_inputs_and_outputs()
    model.make_genai_config(config, {}, str(tmp_path))
    saved = json.loads((tmp_path / "genai_config.json").read_text())["model"]
    assert saved["type"] == ("bailing_hybrid" if text_only else "bailing_moe_v3_vl")
    assert saved["decoder"]["session_options"]["provider_options"][0]["cuda"]["use_tf32"] == "0"
    with pytest.raises(ValueError, match="FP32 routing"):
        builder.apply_runtime_config(
            {"model": saved},
            {"model": {"decoder": {"session_options": {"provider_options": [{"cuda": {"use_tf32": "1"}}]}}}},
        )
    assert model.decoder.input_shapes["position_ids"] == (["num_tokens"] if text_only else [3, "num_tokens"])
    if text_only:
        assert "vision" not in saved
        assert "embedding" not in saved
    else:
        assert saved["image_token_id"] == 29
        assert saved["vision"]["patch_size"] == 16
        assert saved["vision"]["outputs"] == {"image_features": "image_features"}
        assert saved["embedding"]["inputs"] == {"input_ids": "input_ids"}
        assert saved["embedding"]["outputs"] == {"inputs_embeds": "inputs_embeds"}
        assert saved["decoder"]["inputs"]["inputs_embeds"] == "inputs_embeds"
        with pytest.raises(RuntimeError, match="make_model"):
            model.save_model(str(tmp_path))


def _checkpoint(tmp_path, out_features=16, in_features=128):
    scales = torch.arange(
        1, ((out_features + 127) // 128) * ((in_features + 127) // 128) + 1, dtype=torch.float32
    ).reshape((out_features + 127) // 128, (in_features + 127) // 128)
    save_file(
        {
            "model.word_embeddings.weight": torch.ones(32, 128, dtype=torch.bfloat16),
            "model.norm.weight": torch.ones(128, dtype=torch.bfloat16),
            "projection.weight": torch.ones(out_features, in_features).to(torch.float8_e4m3fn),
            "projection.weight_scale_inv": scales,
        },
        tmp_path / "model.safetensors",
    )
    return scales


def test_untied_bailing_checkpoint_requires_its_language_model_head(tmp_path):
    _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="lm_head.weight"):
        _BailingWeights(str(tmp_path), _config(tie_word_embeddings=False), False, ir.DataType.FLOAT)


@pytest.fixture
def bailing_vision_checkpoint(tmp_path):
    config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=24,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
        out_hidden_size=64,
        deepstack_visual_indexes=[],
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        visual = Qwen3VLVisionModel(config).eval()
        visual.merger = torch.nn.Sequential(
            OrderedDict(
                norm=torch.nn.LayerNorm(16, eps=1e-6),
                group=torch.nn.Unflatten(0, (-1, 4)),
                flatten=torch.nn.Flatten(1),
            )
        )
        projector = torch.nn.Sequential(torch.nn.Linear(64, 12), torch.nn.GELU(), torch.nn.Linear(12, 12)).eval()
    tensors = {f"model.visual.{name}": tensor.contiguous() for name, tensor in visual.state_dict().items()}
    tensors.update({f"linear_proj.{name}": tensor.contiguous() for name, tensor in projector.state_dict().items()})
    save_file(tensors, tmp_path / "model.safetensors")
    return config, visual, projector


@pytest.mark.parametrize("grid", [(1, 2, 2), (1, 4, 6)])
def test_ling_vision_checkpoint_and_projector_execute_after_reload(tmp_path, bailing_vision_checkpoint, grid):
    config, reference, projector = bailing_vision_checkpoint
    visual = load_bailing_visual(config, 12, str(tmp_path))
    model = BailingVLVisualModel(config, visual, 12, ir.DataType.FLOAT)
    model.save_model(str(tmp_path))
    patches = torch.linspace(-1, 1, int(np.prod(grid)) * 24).reshape(-1, 24)
    image_grid = torch.tensor([grid], dtype=torch.int64)
    with torch.no_grad():
        expected = projector(reference(patches, grid_thw=image_grid).pooler_output).numpy()
    session = ort.InferenceSession(str(tmp_path / "vision.onnx"), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"pixel_values": patches.numpy(), "image_grid_thw": image_grid.numpy()})[0]

    assert config.out_hidden_size == 64
    assert actual.shape == (int(np.prod(grid)) // 4, 12)
    assert (tmp_path / "vision.onnx.data").exists()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("grid", [(1, 2, 2), (1, 4, 6)])
def test_ling_bf16_vision_preserves_boundaries_with_fp16_arithmetic(tmp_path, bailing_vision_checkpoint, grid):
    config, reference, projector = bailing_vision_checkpoint
    visual = load_bailing_visual(config, 12, str(tmp_path))
    model = BailingVLVisualModel(config, visual, 12, ir.DataType.BFLOAT16)
    model.save_model(str(tmp_path))
    patches = torch.linspace(-1, 1, int(np.prod(grid)) * 24).reshape(-1, 24).to(torch.bfloat16)
    image_grid = torch.tensor([grid], dtype=torch.int64)
    with torch.no_grad():
        expected = projector(reference(patches.float(), grid_thw=image_grid).pooler_output).numpy()
    session = ort.InferenceSession(str(tmp_path / "vision.onnx"), providers=["CPUExecutionProvider"])
    output = torch.empty((int(np.prod(grid)) // 4, 12), dtype=torch.bfloat16)
    binding = session.io_binding()
    binding.bind_input("pixel_values", "cpu", 0, onnx.TensorProto.BFLOAT16, patches.shape, patches.data_ptr())
    binding.bind_cpu_input("image_grid_thw", image_grid.numpy())
    binding.bind_output("image_features", "cpu", 0, onnx.TensorProto.BFLOAT16, output.shape, output.data_ptr())
    session.run_with_iobinding(binding)

    assert session.get_inputs()[0].type == session.get_outputs()[0].type == "tensor(bfloat16)"
    assert model.graph.initializers["visual.patch_embed.proj.weight"].dtype == ir.DataType.FLOAT16
    np.testing.assert_allclose(output.float().numpy(), expected, atol=0.002, rtol=0.02)


@pytest.mark.parametrize("key", ["linear_proj.0.weight", "model.visual.merger.norm.bias"])
def test_ling_vision_rejects_missing_component_weights(tmp_path, bailing_vision_checkpoint, key):
    from safetensors.torch import load_file  # noqa: PLC0415

    config, _, _ = bailing_vision_checkpoint
    tensors = load_file(tmp_path / "model.safetensors")
    del tensors[key]
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match=key):
        load_bailing_visual(config, 12, str(tmp_path))


@pytest.mark.parametrize("q_lora_rank", [None, 8])
def test_mla_query_projection_executes_direct_and_lora_paths(tmp_path, q_lora_rank):
    model = _model(tmp_path, config_overrides={"q_lora_rank": q_lora_rank})
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(384, q_lora_rank or 128, generator=generator) * 0.1
    hidden = torch.randn(1, 3, 128, generator=generator)
    model.graph.inputs.append(model.make_value("hidden", ir.DataType.FLOAT, [1, 3, 128]))
    if q_lora_rank is None:
        attention = SimpleNamespace(q_proj=TensorModule(weight=weight))
        expected = hidden @ weight.T
    else:
        low_rank = torch.randn(q_lora_rank, 128, generator=generator) * 0.1
        norm = torch.randn(q_lora_rank, generator=generator)
        attention = SimpleNamespace(
            q_a_proj=TensorModule(weight=low_rank),
            q_a_layernorm=TensorModule(weight=norm),
            q_b_proj=TensorModule(weight=weight),
        )
        expected = torch.nn.functional.rms_norm(hidden @ low_rank.T, [q_lora_rank], norm, eps=1e-6) @ weight.T
    name = model.make_mla_query(3, attention, "hidden")
    model.graph.outputs.append(model.values[f"{name}/output_0"])
    session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"hidden": hidden.numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("multimodal", [False, True])
def test_direct_query_loader_preserves_ling_pair_order_only_for_multimodal(tmp_path, multimodal):
    query = torch.arange(384, dtype=torch.float32).reshape(-1, 1).expand(-1, 128).contiguous()
    kv = torch.arange(80, dtype=torch.float32).reshape(-1, 1).expand(-1, 128).contiguous()
    save_file(
        {
            "model.word_embeddings.weight": torch.ones(32, 128),
            "model.norm.weight": torch.ones(128),
            "model.layers.3.attention.q_proj.weight": query,
            "model.layers.3.attention.kv_a_proj_with_mqa.weight": kv,
        },
        tmp_path / "model.safetensors",
    )
    config = PretrainedConfig(
        num_hidden_layers=4,
        num_attention_heads=2,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        mrope_section=[8, 12, 12] if multimodal else None,
    )
    weights = _BailingWeights(str(tmp_path), config, False, ir.DataType.FLOAT)
    try:
        attention = weights.build_layer(3, True, False).self_attn
        expected_query = query.clone()
        expected_kv = kv.clone()
        if not multimodal:
            for start in (128, 320):
                expected_query[start : start + 64] = torch.cat(
                    [query[start : start + 64 : 2], query[start + 1 : start + 64 : 2]]
                )
            expected_kv[16:] = torch.cat([kv[16::2], kv[17::2]])
        torch.testing.assert_close(attention.q_proj.weight, expected_query, rtol=0, atol=0)
        torch.testing.assert_close(attention.kv_a_proj_with_mqa.weight, expected_kv, rtol=0, atol=0)
    finally:
        weights.close()


@pytest.mark.parametrize("text_only,text_coordinates", [(False, False), (False, True), (True, True)])
def test_ling_packed_rotary_matches_adjacent_pairs_and_reuses_current_rows(tmp_path, text_only, text_coordinates):
    model = _model(
        tmp_path,
        model_class=BailingVLTextModel,
        use_paged_attention=True,
        dtype=ir.DataType.FLOAT16,
        ep="cuda",
        text_only=text_only,
        config_overrides={"q_lora_rank": None, "mrope_section": [8, 12, 12]},
    )
    position_shape = ["num_tokens"] if text_only else [3, "num_tokens"]
    model.graph.inputs.append(model.make_value("position_ids", ir.DataType.INT64, position_shape))
    positions = np.array([[1, 3, 5]] * 3 if text_coordinates else [[0, 3, 5], [2, 4, 6], [7, 8, 9]], dtype=np.int64)
    feeds = {"position_ids": positions[0] if text_only else positions}
    expected = []
    for label, heads in (("query", 2), ("key", 1)):
        shape = [3, heads, 64]
        data = np.sin(np.arange(np.prod(shape), dtype=np.float32) * 0.03).reshape(shape).astype(np.float16)
        feeds[label] = data.reshape(3, -1)
        model.graph.inputs.append(model.make_value(label, ir.DataType.FLOAT16, ["num_tokens", heads * 64]))
        output = model.make_mla_rope(f"/{label}/rotary", label, heads, ["num_tokens"])
        model.graph.outputs.append(model.values[output])
        owners = np.array([1, 2] * 12 + [0] * 8)
        frequencies = 1 / 10000 ** (np.arange(32, dtype=np.float32) * 2 / 64)
        angles = positions[owners, :].T * frequencies
        cos, sin = np.cos(angles).astype(np.float16)[:, None], np.sin(angles).astype(np.float16)[:, None]
        rotated = np.empty_like(data)
        rotated[..., 0::2] = data[..., 0::2] * cos - data[..., 1::2] * sin
        rotated[..., 1::2] = data[..., 1::2] * cos + data[..., 0::2] * sin
        expected.append(rotated.reshape(3, -1))
    session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
    actual = session.run(None, feeds)
    for output, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(output, reference, rtol=2e-3, atol=1e-3)
    cache_gathers = [
        node for node in model.graph if node.name.startswith("/model/rotary/packed/cos_") and node.op_type == "Gather"
    ]
    assert len(cache_gathers) == (0 if text_only else 3)


@pytest.mark.parametrize("name", ["expert_swiglu_limit_list", "share_expert_swiglu_limit_list"])
@pytest.mark.parametrize("limits", [[4], [0, 0, -4, 0], [0, 0, float("nan"), 0]])
def test_invalid_expert_limits_rejected(tmp_path, name, limits):
    with pytest.raises(ValueError, match=name):
        _model(tmp_path, config_overrides={name: limits})


def test_shared_expert_post_silu_limit_does_not_leak_between_layers(tmp_path):
    model = _model(tmp_path, config_overrides={"share_expert_swiglu_limit_list": [0, 0, 4, 0]})
    model.intermediate_size = 2
    gate = torch.zeros(2, 128)
    gate[:, 0] = torch.tensor([10, -10])
    up = torch.zeros(2, 128)
    up[:, 0] = torch.tensor([2, -10])
    down = torch.zeros(128, 2)
    down[:2, :2] = torch.eye(2)
    mlp = SimpleNamespace(
        gate_proj=TensorModule(weight=gate), up_proj=TensorModule(weight=up), down_proj=TensorModule(weight=down)
    )
    hidden = torch.zeros(1, 3, 128)
    hidden[0, :, 0] = torch.tensor([1, 0.4, -1])
    model.graph.inputs.append(model.make_value("hidden", ir.DataType.FLOAT, list(hidden.shape)))
    for layer_id in (2, 3):
        model.make_mlp_proj(layer_id, mlp, "hidden")
        model.graph.outputs.append(model.values[model.mlp_attrs["output_0"]])
    session = ort.InferenceSession(ir.to_proto(model.model).SerializeToString(), providers=["CPUExecutionProvider"])
    clipped, unbounded = session.run(None, {"hidden": hidden.numpy()})
    expected_gate = torch.nn.functional.silu(hidden @ gate.T)
    expected_up = hidden @ up.T
    expected_clipped = (expected_gate.clamp(max=4) * expected_up.clamp(-4, 4)) @ down.T
    expected_unbounded = (expected_gate * expected_up) @ down.T
    np.testing.assert_allclose(clipped, expected_clipped.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(unbounded, expected_unbounded.numpy(), rtol=1e-6, atol=1e-6)
    assert clipped[0, 0, 0] == 8


@pytest.mark.parametrize(
    ("precision", "quant_method"),
    [(ir.DataType.BFLOAT16, None), (ir.DataType.BFLOAT16, "fp8"), (ir.DataType.INT4, None), (ir.DataType.INT8, None)],
)
@pytest.mark.parametrize("routed_limits", [None, [0, 0, 4, 0]])
def test_routed_expert_clipping_is_per_call_and_preserves_unbounded_layers(
    tmp_path, precision, quant_method, routed_limits
):
    model = _model(
        tmp_path,
        dtype=ir.DataType.BFLOAT16,
        precision=precision,
        ep="cuda",
        quant_method=quant_method,
        config_overrides={
            "expert_swiglu_limit_list": routed_limits,
            "share_expert_swiglu_limit_list": [0, 0, 0, 7],
        },
    )
    model.make_mlp_proj = Mock()
    model.mlp_attrs["output_0"] = "shared"
    original_attrs = model.moe_attrs.copy()
    for layer_id in (2, 3):
        model.make_moe_subgraph(
            layer_id, SimpleNamespace(shared_experts=None), "hidden", router_probs="router", output_scale="scale"
        )
        name = f"/model/layers.{layer_id}/moe/{model.moe_attrs['op_type']}/output_0"
        node = model.values[name].producer()
        if routed_limits is not None and routed_limits[layer_id] > 0:
            assert node.attributes["swiglu_limit"].as_float() == routed_limits[layer_id]
            assert node.attributes["swiglu_limit_after_activation"].as_int() == 1
        else:
            assert "swiglu_limit" not in node.attributes
            assert "swiglu_limit_after_activation" not in node.attributes
    assert model.moe_attrs == original_attrs


@pytest.mark.parametrize("trust", [False, True])
def test_config_failure_preserves_remote_code_policy(tmp_path, monkeypatch, trust):
    load_config = Mock(side_effect=ValueError("custom configuration requires remote code"))
    tokenizer = Mock()
    monkeypatch.setattr(builder.AutoConfig, "from_pretrained", load_config)
    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(builder.LFM2AudioModel, "load_config", Mock(return_value=None))
    monkeypatch.setattr(builder.BailingVLModel, "load_config", Mock(return_value=None))
    options = {"hf_remote": trust}

    with pytest.raises(ValueError, match="requires remote code"):
        builder.get_hf_details("test/bailing", "", str(tmp_path), options)

    load_config.assert_called_once_with("test/bailing", token=True, trust_remote_code=trust, cache_dir=str(tmp_path))
    tokenizer.assert_not_called()
    assert options["hf_remote"] is trust


def test_explicit_remote_code_opt_in_reaches_tokenizer(tmp_path, monkeypatch):
    config = PretrainedConfig(architectures=["BailingMoeV3ForCausalLM"])
    monkeypatch.setattr(builder.AutoConfig, "from_pretrained", Mock(return_value=config))
    tokenizer = Mock(return_value=Mock(bos_token_id=1, eos_token_id=2, pad_token_id=0))
    tokenizer.return_value.get_vocab.return_value = {}
    monkeypatch.setattr(builder.AutoTokenizer, "from_pretrained", tokenizer)

    builder.get_hf_details("test/bailing", "", str(tmp_path), {"hf_remote": True})

    assert tokenizer.call_args.kwargs["trust_remote_code"] is True


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
def test_cpu_graph_declares_float_recurrent_state(tmp_path, dtype):
    model = _model(tmp_path, dtype=dtype)
    model.make_inputs_and_outputs()

    states = [value for value in [*model.model.graph.inputs, *model.model.graph.outputs] if ".recurrent" in value.name]
    assert len(states) == 6
    assert all(value.dtype == ir.DataType.FLOAT for value in states)
    assert all(list(value.shape)[1:] == [2, 64, 64] for value in states)
    conv_states = [value for value in model.model.graph.inputs if ".conv" in value.name]
    assert all(list(value.shape)[1:] == [384, 3] for value in conv_states)


@pytest.mark.parametrize("ep", ["cuda", "webgpu", "cpu"])
@pytest.mark.parametrize("options", [{"state_update_capacity": 1}, {"state_window": 2}])
def test_compact_capture_rejected_for_per_key_decay(tmp_path, ep, options):
    with pytest.raises(ValueError, match="capture and replay do not support per-key decay"):
        _model(tmp_path, dtype=ir.DataType.FLOAT16, ep=ep, use_paged_attention=ep != "cpu", **options)


def test_streaming_loader_rejects_adapters_before_loading(tmp_path, monkeypatch):
    model = _model(tmp_path, adapter_path="test/adapter")
    download = Mock()
    monkeypatch.setattr(bailing, "snapshot_download", download)

    with pytest.raises(ValueError, match="does not support adapter_path"):
        model.load_weights("")

    download.assert_not_called()


@pytest.mark.parametrize("remote", [False, True])
def test_streaming_loader_resolves_checkpoint_source(tmp_path, monkeypatch, remote):
    _checkpoint(tmp_path)
    model = _model(tmp_path, hf_token=False, base_revision="test-revision")
    download = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(bailing, "snapshot_download", download)

    weights = model.load_weights("" if remote else str(tmp_path))
    try:
        assert weights.input_path == str(tmp_path)
        assert weights.embedding.weight.shape == (32, 128)
    finally:
        weights.close()
    if remote:
        download.assert_called_once_with(
            repo_id="test/bailing",
            cache_dir=model.cache_dir,
            token=False,
            revision="test-revision",
            allow_patterns=["*.safetensors", "*.safetensors.index.json"],
        )
    else:
        download.assert_not_called()


@pytest.mark.parametrize("quant_method", ["fp8", "compressed-tensors"])
def test_dequantized_source_is_quantized_to_int4_qdq(tmp_path, quant_method):
    _checkpoint(tmp_path)
    model = _model(tmp_path, precision=ir.DataType.INT4, quant_method=quant_method, use_qdq=True, block_size=32)
    model.model.graph.inputs.append(model.make_value("input", ir.DataType.FLOAT, [1, 1, 128]))
    weights = model.load_weights(str(tmp_path))
    try:
        projection = weights.make_linear("projection")
        assert projection.weight.dtype == torch.bfloat16
        model.make_matmul(projection, "/projection/MatMul", "input")
    finally:
        weights.close()
    model.model.graph.outputs.append(model.values["/projection/MatMul/output_0"])
    out_dir = tmp_path / "export"
    out_dir.mkdir()

    model.save_model(str(out_dir))

    exported = onnx.load(out_dir / "model.onnx")
    assert any(node.op_type == "DequantizeLinear" for node in exported.graph.node)
    assert any(
        tensor.data_type in {onnx.TensorProto.INT4, onnx.TensorProto.UINT4} for tensor in exported.graph.initializer
    )


@pytest.mark.parametrize("layout", ["scalar", "channel", "row", "column"])
def test_fp8_per_channel_layouts_keep_one_input_block(tmp_path, layout):
    model = _model(tmp_path)
    weight = torch.ones(16, 4096).to(torch.float8_e4m3fn)
    scales = torch.arange(1, 17, dtype=torch.float32)
    layouts = {"scalar": torch.tensor(2.0), "channel": scales, "row": scales[None, :], "column": scales[:, None]}

    model.make_matmul_block_quantized_fp8_weight("/projection", "input", weight, layouts[layout])

    node = model.values["/projection/output_0"].producer()
    assert node.attributes["block_size"].as_int() == 4096
    expected = np.full((16, 1), 2.0) if layout == "scalar" else scales.numpy()[:, None]
    np.testing.assert_array_equal(node.inputs[2].const_value.numpy(), expected)


def test_block_fp8_metadata_and_partial_blocks_survive_concatenation(tmp_path):
    scales = _checkpoint(tmp_path, out_features=130, in_features=257)
    model = _model(tmp_path, quant_method="fp8")
    weights = model.load_weights(str(tmp_path))
    try:
        projection = weights.make_linear("projection")
        fused = _BailingWeights.concat_linears([projection, projection])
        model.make_matmul(fused, "/projection", "input")
    finally:
        weights.close()

    node = model.values["/projection/output_0"].producer()
    assert node.attributes["block_size"].as_int() == 128
    normalized = scales.repeat_interleave(128, dim=0)[:130].numpy()
    np.testing.assert_array_equal(node.inputs[2].const_value.numpy(), np.concatenate([normalized, normalized]))


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
@pytest.mark.parametrize("precision", [ir.DataType.INT4, ir.DataType.INT8, ir.DataType.BFLOAT16])
def test_group_router_keeps_projection_and_selection_in_fp32(tmp_path, dtype, precision):
    model = _model(tmp_path, dtype=dtype, precision=precision)
    model.moe_attrs["num_experts"] = 512
    model.model.graph.inputs.append(model.make_value("hidden", dtype, model.make_hidden_state_shape()))
    gate = TensorModule(weight=torch.linspace(-1, 1, 512 * 128).reshape(512, 128).to(torch.bfloat16))
    moe = SimpleNamespace(gate=gate, expert_bias=torch.zeros(512))

    scores, _ = model.make_moe_router(2, moe, "hidden")

    projection = model.values["/model/layers.2/moe/router/MatMul/output_0"].producer()
    router = model.values["/model/layers.2/moe/router/MoEGroupRouter/output_0"].producer()
    assert projection.inputs[0].dtype == ir.DataType.FLOAT
    assert projection.inputs[1].dtype == ir.DataType.FLOAT
    assert projection.outputs[0].dtype == ir.DataType.FLOAT
    assert router.inputs[0] is projection.outputs[0]
    assert router.outputs[0].dtype == ir.DataType.FLOAT
    assert projection.name in model.quant_attrs["nodes_to_exclude"]
    assert model.values[scores].dtype == dtype
    if dtype != ir.DataType.FLOAT:
        assert model.values[scores].producer().inputs[0] is router.outputs[0]


@pytest.mark.parametrize("activation", [None, "silu", "sigmoid"])
def test_trt_rms_norm_expansion_matches_gate_math(tmp_path, activation):
    model = _model(tmp_path, ep="trt-rtx")
    shape = [1, 2, 128]
    for name in ("input", "gate"):
        model.model.graph.inputs.append(model.make_value(name, ir.DataType.FLOAT, shape))
    scale = torch.linspace(0.5, 1.5, 64)
    model.make_initializer(scale, "scale")
    model.make_gated_rms_norm("/norm", "input", "scale", "gate", shape, activation=activation)
    model.model.graph.outputs.append(model.values["/norm/output_0"])
    model.model.graph.sort()
    inputs = np.linspace(-2, 2, 256, dtype=np.float32).reshape(shape)
    gate = np.linspace(-3, 3, 256, dtype=np.float32).reshape(shape)

    actual = ReferenceEvaluator(ir.serde.serialize_model(model.model)).run(None, {"input": inputs, "gate": gate})[0]

    grouped = inputs.reshape(1, 2, 2, 64)
    normalized = (grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-5) * scale.numpy()).reshape(shape)
    expected = normalized / (1 + np.exp(-gate))
    if activation != "sigmoid":
        expected *= gate
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

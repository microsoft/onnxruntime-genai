import json
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch
from quantization import QuantConfig
from safetensors.torch import save_file
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from models.builders.base import Model
from models.builders.qwen3_8 import Qwen4ExpModel, Qwen4ExpMTPTextModel
from models.loaders.quant_model import QuantModel
from models.loaders.qwen import Qwen4ExpMTPModel
from models.loaders.qwen3_8 import Qwen38ModeloptModel


@pytest.fixture
def checkpoint(tmp_path):
    config = {
        "model_type": "qwen4_exp",
        "vision_config": {
            "hidden_size": 16,
            "intermediate_size": 16,
            "out_hidden_size": 16,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "depth": 1,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
        },
        "text_config": {
            "num_hidden_layers": 2,
            "num_experts": 2,
            "ple_layer_ids": [2],
            "eos_token_id": 0,
            "split_ngram_parts": 2,
            "mtp_num_hidden_layers": 1,
        },
    }
    tensors = {}

    def dense(prefix, rows=16, columns=16):
        tensors[f"{prefix}.weight"] = torch.ones((rows, columns), dtype=torch.bfloat16)

    def norm(prefix):
        tensors[f"{prefix}.weight"] = torch.zeros(16, dtype=torch.bfloat16)

    def hyper(prefix):
        norm(f"{prefix}.hc_norm")
        for name in ("input_mix_weight_down", "input_mix_weight_up", "block_inject_weight"):
            dense(f"{prefix}.{name}")

    def layer(prefix, mtp=False):
        if prefix.endswith(".0") and not mtp:
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                dense(f"{prefix}.linear_attn.{name}")
            norm(f"{prefix}.linear_attn.norm")
            tensors[f"{prefix}.linear_attn.conv1d.weight"] = torch.ones((16, 1, 4))
            tensors[f"{prefix}.linear_attn.A_log"] = torch.zeros(16)
            tensors[f"{prefix}.linear_attn.dt_bias"] = torch.zeros(16)
        else:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                dense(f"{prefix}.self_attn.{name}")
            norm(f"{prefix}.self_attn.q_norm")
            norm(f"{prefix}.self_attn.k_norm")
            dense(f"{prefix}.self_attn.indexer.index_qk_proj")
            norm(f"{prefix}.self_attn.indexer.q_layernorm")
            norm(f"{prefix}.self_attn.indexer.k_layernorm")
        hyper(f"{prefix}.attn_hyper_connection")
        hyper(f"{prefix}.mlp_hyper_connection")
        dense(f"{prefix}.mlp.gate", 2)
        dense(f"{prefix}.mlp.shared_expert_gate", 1)
        for name in ("gate_proj", "up_proj", "down_proj"):
            dense(f"{prefix}.mlp.shared_expert.{name}")
        for expert_id in range(2):
            for name in ("gate_proj", "up_proj", "down_proj"):
                base = f"{prefix}.mlp.experts.{expert_id}.{name}"
                if mtp:
                    value = 1 + expert_id * 3 + ("gate_proj", "up_proj", "down_proj").index(name)
                    tensors[f"{base}.weight"] = torch.full((16, 16), value, dtype=torch.float8_e4m3fn)
                    tensors[f"{base}.weight_scale_inv"] = torch.full((1, 1), value + 1.0, dtype=torch.bfloat16)
                else:
                    tensors[f"{base}.weight"] = torch.full((16, 8), 0x21, dtype=torch.uint8)
                    tensors[f"{base}.weight_scale"] = torch.ones((16, 1), dtype=torch.float8_e4m3fn)
                    tensors[f"{base}.weight_scale_2"] = torch.tensor(0.5)

    dense("model.language_model.embed_tokens", 8)
    dense("lm_head", 8)
    hyper("model.language_model.hyper_connection_mixer")
    layer("model.language_model.layers.0")
    layer("model.language_model.layers.1")
    prefix = "model.language_model.layers.1.ple"
    for name in ("key_proj", "value_proj"):
        dense(f"{prefix}.{name}")
    for name in ("norm_key", "norm_query", "norm_conv"):
        norm(f"{prefix}.{name}")
    tensors[f"{prefix}.conv1d.weight"] = torch.ones((16, 1, 4))
    embedding = f"{prefix}.ple_embedding"
    for name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
        tensors[f"{embedding}.{name}"] = torch.arange(3, dtype=torch.int64)
    for shard_id in (1, 0):
        tensors[f"{embedding}.ngram_embedding.shard_{shard_id}.weight"] = torch.full(
            (4, 8), shard_id + 1.0, dtype=torch.float8_e4m3fn
        )
    tensors[f"{embedding}.ngram_embedding.weight_scale"] = torch.tensor([0.25], dtype=torch.bfloat16)
    layer("mtp.layers.0", mtp=True)
    hyper("mtp.hyper_connection_mixer")
    for name in ("fc_embedding", "fc_hidden"):
        dense(f"mtp.{name}")
    norm("mtp.pre_fc_norm_embedding")
    norm("mtp.pre_fc_norm_hidden")
    visual = Qwen4ExpVisionModel(Qwen4ExpVisionConfig(**config["vision_config"])).to(torch.bfloat16)
    tensors.update({f"model.visual.{name}": value.contiguous() for name, value in visual.state_dict().items()})
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(tensors, tmp_path / "model.safetensors")
    return tmp_path


def load(checkpoint, num_layers=2):
    return QuantModel.from_pretrained(
        "modelopt",
        input_path=str(checkpoint),
        quant_attrs={"config": {}},
        q_size=16,
        kv_size=16,
        intermediate_size=16,
        num_layers=num_layers,
    )


def test_nvfp4_dispatch_and_qwen38_surface(checkpoint):
    model = load(checkpoint)
    assert isinstance(model, Qwen38ModeloptModel)
    assert model.modules() == [model.embedding, *model.layers, model.lm_head]
    assert model.layers[0].mlp.experts.quant_type == "nvfp4"
    assert model.layers[0].mlp.experts.weights_prepacked == 1
    assert model.layers[0].mlp.experts.gate_up_qweight.shape == (2, 16, 16)
    assert model.layers[0].mlp.experts.gate_up_global_scales.tolist() == [0.5, 0.5]
    assert model.layers[0].attn_hyper_connection.input_mix_weight_down.out_features == 16
    assert model.layers[1].self_attn.indexer.index_qk_proj.weight.dtype == torch.bfloat16
    assert model.model.language_model.layers is model.layers
    assert model.handles == {}


@pytest.mark.parametrize("weights_prepacked", [None, -1, 0, 1])
def test_nvfp4_experts_keep_row_major_checkpoint_bytes(weights_prepacked):
    loader = object.__new__(Qwen38ModeloptModel)
    experts = []
    for expert_id in range(2):
        projections = {}
        for projection_id, name in enumerate(("gate_proj", "up_proj", "down_proj")):
            projections[name] = SimpleNamespace(
                weight=(torch.arange(128).reshape(16, 8) + expert_id * 53 + projection_id * 37).to(torch.uint8),
                weight_scale=torch.arange(1, 17).reshape(16, 1).to(torch.float8_e4m3fn),
                weight_scale_2=torch.tensor(0.5 + expert_id),
            )
        experts.append(SimpleNamespace(**projections))

    prepared = loader.prepare_qmoe_experts(experts)
    assert prepared.weights_prepacked == 1
    assert prepared.gate_up_qweight.dtype == prepared.down_qweight.dtype == torch.uint8
    for expert_id, expert in enumerate(experts):
        fused_weight = prepared.gate_up_qweight[expert_id].reshape(32, 8)
        assert torch.equal(fused_weight[0::2], expert.gate_proj.weight)
        assert torch.equal(fused_weight[1::2], expert.up_proj.weight)
        assert torch.equal(prepared.down_qweight[expert_id].reshape(16, 8), expert.down_proj.weight)
        fused_scales = prepared.gate_up_scales[expert_id].reshape(32, 1)
        assert torch.equal(fused_scales[0::2], expert.gate_proj.weight_scale.view(torch.uint8))
        assert torch.equal(fused_scales[1::2], expert.up_proj.weight_scale.view(torch.uint8))
        assert torch.equal(prepared.down_scales[expert_id], expert.down_proj.weight_scale.view(torch.uint8))
    assert prepared.gate_up_global_scales.tolist() == prepared.down_global_scales.tolist() == [0.5, 1.5]

    builder = object.__new__(Model)
    builder.ep = "cuda"
    builder.io_dtype = ir.DataType.FLOAT16
    builder.moe_attrs = {
        "op_type": "QMoE", "quant_type": "nvfp4", "num_experts": 2,
        "weights_prepacked": 1,
        "expert_weight_bits": 4, "top_k": 1, "normalize_routing_weights": True,
        "activation_type": "swiglu", "activation_alpha": 1.702, "activation_beta": 1.0,
        "swiglu_fusion": 1, "swiglu_limit": 7.0, "use_sparse_mixer": False,
    }
    initializers = {}
    builder.make_initializer = lambda tensor, name, **kwargs: initializers.setdefault(name, tensor)
    prepared.weights_prepacked = weights_prepacked
    builder.make_moe_expert_initializers(0, prepared)
    names = builder.make_moe_expert_names(0)
    assert torch.equal(initializers[names["gate_up_weight"]], prepared.gate_up_qweight)
    assert torch.equal(initializers[names["down_weight"]], prepared.down_qweight)
    recorded = {}
    builder.make_node = lambda op_type, **kwargs: recorded.update(op_type=op_type, **kwargs)
    builder.make_value = lambda *args, **kwargs: None
    builder.make_hidden_state_shape = lambda: ["batch", "sequence", 16]
    builder.make_moe_op(
        "/model/layers.0/moe/QMoE", root_input="hidden", router_probs="router",
        weight1=names["gate_up_weight"], scales1=names["gate_up_scales"],
        weight2=names["down_weight"], scales2=names["down_scales"],
    )
    if weights_prepacked in (None, -1):
        assert "weights_prepacked" not in recorded
    else:
        assert recorded["weights_prepacked"] == weights_prepacked
    assert "nvfp4_weight_layout" not in recorded
    assert recorded["quant_type"] == "nvfp4" and recorded["block_size"] == 16


def test_ple_shards_preserve_fp8_scale_and_order(checkpoint):
    model = load(checkpoint)
    embedding = model.layers[1].ple.ple_embedding
    assert embedding.eos_token_id == 0
    assert embedding.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert embedding.ngram_embedding.weight.shape == (8, 8)
    assert embedding.ngram_embedding.weight.float()[:, 0].tolist() == [1.0] * 4 + [2.0] * 4
    assert embedding.ngram_embedding.weight_scale.item() == 0.25


def test_mtp_block_fp8_is_preserved_without_transformers(checkpoint):
    model = load(checkpoint, num_layers=1)
    mtp = Qwen4ExpMTPModel.from_modelopt(model, None, False)
    assert mtp.embedding is model.embedding
    assert mtp.lm_head is model.lm_head
    experts = mtp.layers[0].mlp.experts
    assert experts.quant_type == "fp8_block"
    for expert_id in range(2):
        for name in ("gate_proj", "up_proj", "down_proj"):
            projection = getattr(getattr(experts, str(expert_id)), name)
            value = 1 + expert_id * 3 + ("gate_proj", "up_proj", "down_proj").index(name)
            assert projection.weight.dtype == torch.float8_e4m3fn
            assert torch.equal(
                projection.weight.view(torch.uint8),
                torch.full((16, 16), value).to(torch.float8_e4m3fn).view(torch.uint8),
            )
            assert projection.weight_scale_inv.dtype == torch.bfloat16
            assert projection.weight_scale_inv.item() == value + 1.0
    assert mtp.layers[0].self_attn.indexer.index_qk_proj.weight.shape == (16, 16)
    assert model.handles == {}


def test_visual_blocks_are_ordered_and_loaded_lazily(checkpoint):
    model = load(checkpoint)
    assert not hasattr(model.model, "visual")
    visual = model.load_visual()
    assert visual.blocks[0].attn.qkv.weight.shape == (48, 16)
    assert visual.num_grid_per_side == 4
    assert visual.rotary_pos_emb.inv_freq.device.type == "cpu"
    assert visual.blocks[0].norm1.eps == 1e-6
    assert model.handles == {}


def test_native_mtp_expert_initializers_preserve_checkpoint_bytes(checkpoint):
    mtp = load(checkpoint).load_mtp()
    model = object.__new__(Qwen4ExpMTPTextModel)
    model.ep = "cuda"
    model.io_dtype = ir.DataType.FLOAT16
    model.moe_attrs = {"num_experts": 2}
    model.moe_intermediate_size = model.hidden_size = 16
    captured = {}
    model.make_initializer = lambda tensor, name, **kwargs: captured.setdefault(name, tensor)
    model.make_moe_preprocessing(0, mtp.layers[0].mlp, "hidden")
    names = model.make_moe_expert_names(0)
    assert model.moe_attrs["quant_type"] == "fp8"
    assert model.moe_attrs["activation_type"] == "silu"
    assert model.moe_attrs["swiglu_fusion"] == 0
    assert captured[names["gate_up_weight"]].dtype == torch.float8_e4m3fn
    assert captured[names["gate_up_scales"]].shape == (2, 1, 1)
    assert captured[names["gate_up_scales"]].dtype == torch.bfloat16
    expert = getattr(mtp.layers[0].mlp.experts, "0")
    assert torch.equal(
        captured[names["gate_up_weight"]][0, :16].view(torch.uint8), expert.gate_proj.weight.view(torch.uint8)
    )
    assert torch.equal(captured[names["down_weight"]][0].view(torch.uint8), expert.down_proj.weight.view(torch.uint8))
    up_weight, up_scales = model.moe_attrs["up_projection_names"]
    assert torch.equal(captured[up_weight][0].view(torch.uint8), expert.up_proj.weight.view(torch.uint8))
    assert torch.equal(captured[up_scales][0], expert.up_proj.weight_scale_inv)
    assert names["gate_up_bias"] == names["down_bias"] == ""
    model.moe_attrs.update(
        top_k=1,
        normalize_routing_weights=True,
        use_sparse_mixer=False,
        activation_alpha=1.0,
        activation_beta=0.0,
        swiglu_limit=None,
    )
    recorded = {}
    model.make_node = lambda op_type, **kwargs: recorded.update(op_type=op_type, **kwargs)
    model.make_value = lambda *args, **kwargs: None
    model.make_hidden_state_shape = lambda: ["batch", "sequence", 16]
    model.make_moe_op(
        "/model/layers.0/moe/QMoE",
        root_input="hidden",
        router_probs="router",
        weight1=names["gate_up_weight"],
        scales1=names["gate_up_scales"],
        weight2=names["down_weight"],
        scales2=names["down_scales"],
    )
    assert recorded["quant_type"] == "fp8" and recorded["block_size"] == 128
    assert recorded["inputs"][8:10] == [up_weight, up_scales]
    assert recorded["inputs"][4] == recorded["inputs"][7] == ""


def test_incomplete_ple_shards_are_rejected(checkpoint):
    model = load(checkpoint, num_layers=1)
    model.text_config["split_ngram_parts"] = 3
    try:
        with pytest.raises(ValueError, match="Incomplete.*PLE"):
            model.make_ple("model.language_model.layers.1.ple")
    finally:
        model.close()


@pytest.mark.parametrize("explicit", [False, True])
def test_nvfp4_mtp_unquantized_module_config_and_explicit_override(monkeypatch, explicit):
    config = QuantConfig.from_dict({"io_dtype": "fp16", "moe": {"type": "nvfp4"}})
    wrapper = object.__new__(Qwen4ExpModel)
    wrapper.decoder = SimpleNamespace(quant_type="modelopt", quant_config=config)
    wrapper.mtp_attrs = {}
    options = {"_quant_config": config}
    if explicit:
        options["mtp_quant_config"] = QuantConfig.from_dict({"io_dtype": "bf16", "moe": {"type": "int8"}})
    captured = {}

    def mtp_builder(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        captured.update(io_dtype=io_dtype, options=extra_options)
        return SimpleNamespace()

    monkeypatch.setitem(Qwen4ExpModel.make_mtp_model.__globals__, "Qwen4ExpMTPTextModel", mtp_builder)
    wrapper.make_mtp_model(SimpleNamespace(), ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", None, options)
    assert captured["options"]["_quant_config"].moe.type == ("int8" if explicit else "none")
    assert captured["io_dtype"] == (ir.DataType.BFLOAT16 if explicit else ir.DataType.FLOAT16)
    assert config.moe.type == "nvfp4"


def test_fp8_blocks_validate_partial_blocks_without_conversion():
    projection = SimpleNamespace(
        weight=torch.ones((129, 129), dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    original_weight = projection.weight
    original_scale = projection.weight_scale_inv
    Qwen38ModeloptModel.validate_fp8_blocks(projection)
    assert projection.weight is original_weight
    assert projection.weight_scale_inv is original_scale


def test_mtp_rmsnorm_uses_onnx_domain():
    model = object.__new__(Qwen4ExpMTPTextModel)
    model.io_dtype = ir.DataType.FLOAT16
    model.hidden_size = 16
    model.use_paged_attention = False
    model.layernorm_attrs = {"add_offset": 1, "epsilon": 1e-6}
    recorded = []
    model.make_initializer = lambda *args, **kwargs: None
    model.make_value = lambda *args, **kwargs: None
    model.make_node = lambda *args, **kwargs: recorded.append(kwargs)
    model.make_offset_rmsnorm("/model/mtp/norm", "hidden", torch.zeros(16))
    assert recorded[0].get("domain", "") == ""


@pytest.mark.parametrize("scale", [None, torch.ones(1), torch.zeros((2, 2)), torch.full((2, 2), float("nan"))])
def test_fp8_blocks_reject_invalid_scales(scale):
    projection = SimpleNamespace(weight=torch.ones((129, 129), dtype=torch.float8_e4m3fn), weight_scale_inv=scale)
    with pytest.raises(ValueError, match="scale"):
        Qwen38ModeloptModel.validate_fp8_blocks(projection)

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""End-to-end export of a tiny Qwen-3.8 Flash Next (``Qwen4Exp``) checkpoint.

A randomly initialized, very small ``Qwen4Exp`` model is built with the reference
``transformers`` implementation and pushed through the real builder, so the whole path is
exercised: hyper-connections, PLE, GatedDeltaNet linear attention, QSA attention, packed MoE
experts, the final mixer, and the three-file multimodal export.

Skipped when the installed ``transformers`` does not ship ``qwen4_exp``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
pytest.importorskip("transformers.models.qwen4_exp")

import onnx_ir as ir  # noqa: E402

from transformers.models.qwen4_exp.configuration_qwen4_exp import (  # noqa: E402
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
    Qwen4ExpVisionConfig,
)
from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
)

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))

sys.modules.setdefault("models", types.ModuleType("models"))
_builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
_builders_package.__path__ = [str(BUILDERS_DIR)]


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


base_module = _load_builder_module("base")
_load_builder_module("quant_config")
qwen_module = _load_builder_module("qwen")
qwen4exp = _load_builder_module("qwen4exp")

NUM_LAYERS = 4
HIDDEN = 64
HC_COUNT = 2
PLE_LAYER_IDS = [1, 2]  # one-based; PLE is only allowed on linear-attention layers
LINEAR_LAYERS = [0, 1, 2]
ATTENTION_LAYERS = [3]


def _text_config():
    return Qwen4ExpTextConfig(
        vocab_size=512,
        hidden_size=HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        # The published checkpoints call the attention layers "qwen_sparse_attention".
        layer_types=["linear_attention"] * 3 + ["qwen_sparse_attention"],
        hc_count=HC_COUNT,
        hc_lowrank=16,
        ple_layer_ids=list(PLE_LAYER_IDS),
        ple_embed_dim=32,
        ple_conv_kernel_size=2,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=1000,
        make_ngram_vocab_size_divisible_by=8,
        seed=1234,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=64,
        indexer_compress_ratio=4,
        eos_token_id=2,
        pad_token_id=0,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "mrope_section": [2, 1, 1]},
    )


def _multimodal_config():
    config = Qwen4ExpConfig(
        text_config=_text_config(),
        vision_config=Qwen4ExpVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            depth=2,
            patch_size=4,
            temporal_patch_size=2,
            in_channels=3,
            spatial_merge_size=2,
            out_hidden_size=HIDDEN,
            num_position_embeddings=16,
        ),
    )
    config.architectures = ["Qwen4ExpForConditionalGeneration"]
    return config


def _stub_checkpoint_io(monkeypatch, hf_model):
    """Feed the builder an in-memory checkpoint and a local HF config."""
    monkeypatch.setattr(base_module.Model, "load_weights", lambda self, path: hf_model)
    monkeypatch.setattr(
        qwen4exp.Qwen4ExpTextModel,
        "load_weights",
        lambda self, path: (setattr(self, "_hf_model", hf_model) or hf_model),
    )

    hf_config = SimpleNamespace(
        eos_token_id=2, bos_token_id=1, pad_token_id=0, save_pretrained=lambda *a, **k: None
    )
    auto_config = SimpleNamespace(from_pretrained=lambda *a, **k: hf_config)

    class _NoGenerationConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise FileNotFoundError("no generation_config.json")

    for module in (base_module, qwen_module):
        monkeypatch.setattr(module, "AutoConfig", auto_config, raising=False)
        monkeypatch.setattr(module, "GenerationConfig", _NoGenerationConfig, raising=False)


def _op_counts(path):
    model = onnx.load(str(path), load_external_data=False)
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return model, counts


@pytest.fixture(scope="module")
def _multimodal_export_root(tmp_path_factory):
    """Export the tiny multimodal model once (non-paged) and reuse it across assertions."""
    monkeypatch = pytest.MonkeyPatch()
    try:
        torch.manual_seed(0)
        config = _multimodal_config()
        hf_model = Qwen4ExpForConditionalGeneration(config).eval()
        _stub_checkpoint_io(monkeypatch, hf_model)

        out_dir = tmp_path_factory.mktemp("qwen4exp_mm")
        cache_dir = tmp_path_factory.mktemp("qwen4exp_mm_cache")
        builder = qwen4exp.Qwen4ExpModel(
            config, ir.DataType.FLOAT16, ir.DataType.FLOAT16, "cuda", str(cache_dir), {}
        )
        builder.make_model("in-memory")
        builder.save_model(str(out_dir))
        builder.make_genai_config("in-memory", {}, str(out_dir))
        return out_dir
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="module")
def _paged_export_root(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    try:
        torch.manual_seed(0)
        config = _multimodal_config()
        hf_model = Qwen4ExpForConditionalGeneration(config).eval()
        _stub_checkpoint_io(monkeypatch, hf_model)

        out_dir = tmp_path_factory.mktemp("qwen4exp_paged")
        cache_dir = tmp_path_factory.mktemp("qwen4exp_paged_cache")
        builder = qwen4exp.Qwen4ExpModel(
            config,
            ir.DataType.FLOAT16,
            ir.DataType.FLOAT16,
            "cuda",
            str(cache_dir),
            {"use_paged_attention": "true"},
        )
        builder.make_model("in-memory")
        builder.save_model(str(out_dir))
        builder.make_genai_config("in-memory", {}, str(out_dir))
        return out_dir
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="module")
def _text_only_export_root(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    try:
        torch.manual_seed(0)
        config = _text_config()
        config.architectures = ["Qwen4ExpForCausalLM"]
        hf_model = Qwen4ExpForCausalLM(config).eval()
        _stub_checkpoint_io(monkeypatch, hf_model)

        out_dir = tmp_path_factory.mktemp("qwen4exp_text")
        cache_dir = tmp_path_factory.mktemp("qwen4exp_text_cache")
        builder = qwen4exp.Qwen4ExpTextModel(
            config,
            ir.DataType.FLOAT16,
            ir.DataType.FLOAT16,
            "cuda",
            str(cache_dir),
            {"exclude_embeds": False},
        )
        builder.make_model("in-memory")
        builder.save_model(str(out_dir))
        builder.make_genai_config("in-memory", {}, str(out_dir))
        return out_dir
    finally:
        monkeypatch.undo()


#####################################################################################
# Multimodal export
#####################################################################################


def test_multimodal_export_writes_three_graphs(_multimodal_export_root):
    produced = {path.name for path in _multimodal_export_root.iterdir()}
    assert {"text.onnx", "embedding.onnx", "vision.onnx"} <= produced
    for name in ("text.onnx", "embedding.onnx", "vision.onnx"):
        assert (_multimodal_export_root / name).stat().st_size > 0


def test_text_graph_uses_the_new_contrib_ops_once_per_owning_layer(_multimodal_export_root):
    _, counts = _op_counts(_multimodal_export_root / "text.onnx")

    assert counts["NGramHashMapping"] == len(PLE_LAYER_IDS)
    assert counts["EngramGate"] == len(PLE_LAYER_IDS)
    assert counts["ShortConvWithState"] == len(PLE_LAYER_IDS)
    assert counts["QwenSparseAttention"] == len(ATTENTION_LAYERS)
    assert "GroupQueryAttention" not in counts, "QSA layers must not fall back to dense attention"
    assert "PagedAttention" not in counts


def test_text_graph_keeps_one_moe_block_per_layer(_multimodal_export_root):
    _, counts = _op_counts(_multimodal_export_root / "text.onnx")

    assert counts["MoE"] == NUM_LAYERS
    # Shared expert is folded into the routed output with the fused GatedAdd on CUDA.
    assert counts["GatedAdd"] == NUM_LAYERS


def test_text_graph_exposes_input_ids_for_the_ple_hash(_multimodal_export_root):
    model, _ = _op_counts(_multimodal_export_root / "text.onnx")
    input_names = [value.name for value in model.graph.input]

    # `inputs_embeds` is the decoder entry point, but PLE hashes the raw token ids, so both
    # must be graph inputs in the multimodal build.
    assert "inputs_embeds" in input_names
    assert "input_ids" in input_names


def test_text_graph_declares_every_per_layer_state(_multimodal_export_root):
    model, _ = _op_counts(_multimodal_export_root / "text.onnx")
    input_names = {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}

    for layer_id in ATTENTION_LAYERS:
        assert f"past_key_values.{layer_id}.key" in input_names
        assert f"past_key_values.{layer_id}.indexer_key" in input_names
        assert f"present.{layer_id}.indexer_key" in output_names
    for layer_id in LINEAR_LAYERS:
        assert f"past_key_values.{layer_id}.conv_state" in input_names
        assert f"past_key_values.{layer_id}.recurrent_state" in input_names
    for layer_id in [i - 1 for i in PLE_LAYER_IDS]:
        assert f"past_key_values.{layer_id}.ple_conv_state" in input_names
        assert f"past_key_values.{layer_id}.ple_tokens" in input_names
        assert f"present.{layer_id}.ple_conv_state" in output_names
        assert f"present.{layer_id}.ple_tokens" in output_names


def test_linear_layers_have_no_kv_or_indexer_state(_multimodal_export_root):
    model, _ = _op_counts(_multimodal_export_root / "text.onnx")
    input_names = {value.name for value in model.graph.input}

    for layer_id in LINEAR_LAYERS:
        assert f"past_key_values.{layer_id}.key" not in input_names
        assert f"past_key_values.{layer_id}.indexer_key" not in input_names


def test_ple_state_is_absent_on_non_ple_layers(_multimodal_export_root):
    model, _ = _op_counts(_multimodal_export_root / "text.onnx")
    input_names = {value.name for value in model.graph.input}

    ple_layers = {i - 1 for i in PLE_LAYER_IDS}
    for layer_id in set(range(NUM_LAYERS)) - ple_layers:
        assert f"past_key_values.{layer_id}.ple_conv_state" not in input_names
        assert f"past_key_values.{layer_id}.ple_tokens" not in input_names


def test_embedding_graph_scatters_image_features_into_token_embeddings(_multimodal_export_root):
    model, _ = _op_counts(_multimodal_export_root / "embedding.onnx")

    assert [value.name for value in model.graph.input] == ["input_ids", "image_features"]
    assert [value.name for value in model.graph.output] == ["inputs_embeds"]
    assert any(node.op_type == "ScatterND" for node in model.graph.node)


def test_vision_graph_exposes_the_documented_geometry_inputs(_multimodal_export_root):
    model, counts = _op_counts(_multimodal_export_root / "vision.onnx")

    assert [value.name for value in model.graph.input] == [
        "pixel_values",
        "pos_embed_indices",
        "pos_embed_weights",
        "vision_cos",
        "vision_sin",
        "vision_attention_bias",
    ]
    assert [value.name for value in model.graph.output] == ["image_features"]
    assert counts["MultiHeadAttention"] == 2, "one attention op per vision block"


def test_multimodal_genai_config_describes_all_three_graphs(_multimodal_export_root):
    config = json.loads((_multimodal_export_root / "genai_config.json").read_text())["model"]

    assert config["type"] == "qwen4exp"
    assert config["decoder"]["filename"] == "text.onnx"
    assert config["embedding"]["filename"] == "embedding.onnx"
    assert config["embedding"]["outputs"]["inputs_embeds"] == "inputs_embeds"
    assert config["vision"]["filename"] == "vision.onnx"
    assert config["vision"]["outputs"]["image_features"] == "image_features"
    assert config["vision"]["spatial_merge_size"] == 2
    assert "image_token_id" in config and "vision_start_token_id" in config


#####################################################################################
# Paged export
#####################################################################################


def test_paged_export_swaps_in_the_paged_sparse_attention_op(_paged_export_root):
    _, counts = _op_counts(_paged_export_root / "text.onnx")

    assert counts["SparsePagedAttention"] == len(ATTENTION_LAYERS)
    assert "QwenSparseAttention" not in counts


def test_paged_export_publishes_the_full_state_manifest(_paged_export_root):
    decoder = json.loads((_paged_export_root / "genai_config.json").read_text())["model"]["decoder"]

    assert [group["kind"] for group in decoder["state_groups"]] == [
        "paged_kv",
        "fixed_conv",
        "fixed_recurrent",
        "fixed_ple_conv",
        "fixed_ple_tokens",
        "paged_indexer_key",
    ]
    by_kind = {group["kind"]: group["layer_ids"] for group in decoder["state_groups"]}
    assert by_kind["paged_kv"] == ATTENTION_LAYERS
    assert by_kind["paged_indexer_key"] == ATTENTION_LAYERS
    assert by_kind["fixed_conv"] == LINEAR_LAYERS
    assert by_kind["fixed_ple_conv"] == [i - 1 for i in PLE_LAYER_IDS]


def test_paged_export_registers_the_new_state_name_templates(_paged_export_root):
    decoder = json.loads((_paged_export_root / "genai_config.json").read_text())["model"]["decoder"]

    assert decoder["inputs"]["past_ple_conv_names"] == "past_key_values.%d.ple_conv_state"
    assert decoder["inputs"]["past_indexer_key_names"] == "past_key_values.%d.indexer_key"
    assert decoder["outputs"]["present_ple_token_names"] == "present.%d.ple_tokens"


#####################################################################################
# Text-only export
#####################################################################################


def test_text_only_export_produces_a_single_decoder_graph(_text_only_export_root):
    produced = {path.name for path in _text_only_export_root.iterdir()}

    assert "model.onnx" in produced
    assert "embedding.onnx" not in produced
    assert "vision.onnx" not in produced


def test_text_only_export_embeds_tokens_inside_the_decoder(_text_only_export_root):
    model, counts = _op_counts(_text_only_export_root / "model.onnx")
    input_names = [value.name for value in model.graph.input]

    assert input_names[0] == "input_ids"
    assert "inputs_embeds" not in input_names
    # The PLE and QSA machinery is identical to the multimodal build.
    assert counts["NGramHashMapping"] == len(PLE_LAYER_IDS)
    assert counts["QwenSparseAttention"] == len(ATTENTION_LAYERS)


def test_text_only_genai_config_uses_the_text_model_type(_text_only_export_root):
    config = json.loads((_text_only_export_root / "genai_config.json").read_text())["model"]

    assert config["type"] == "qwen4exp_text"
    assert "vision" not in config
    assert "embedding" not in config

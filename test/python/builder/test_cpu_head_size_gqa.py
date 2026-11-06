# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tests for GitHub issue #2791:
CPU FP32 models with head_size % 16 != 0 build but fail at runtime.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import onnx
import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
MODELS_DIR = BUILDERS_DIR.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]


def _load_builder_module(module_name):
    qualified = f"models.builders.{module_name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


base_module = _load_builder_module("base")
Model = base_module.Model
ir = base_module.ir


class _FakeGQAModel:
    is_fused_rope_supported = Model.is_fused_rope_supported
    is_gqa_supported = Model.is_gqa_supported
    is_packed_matmul_supported = Model.is_packed_matmul_supported
    is_fused_qk_norm_gqa_supported = Model.is_fused_qk_norm_gqa_supported
    is_packed_attn_supported = Model.is_packed_attn_supported
    make_attention_init = Model.make_attention_init
    make_paged_attention_init = Model.make_paged_attention_init

    def __init__(self, ep="cpu", head_size=16, q_norm=False, k_norm=False):
        self.ep = ep
        self.io_dtype = ir.DataType.FLOAT
        self.head_size = head_size
        self.num_attn_heads = 4
        self.num_kv_heads = 2
        self.use_paged_attention = False
        self.matmul_attrs = {"use_lora": False}
        self.input_names = {"input_ids": "input_ids", "position_ids": "position_ids"}
        self.attention_attrs = {
            "op_type": "GroupQueryAttention",
            "rope": True,
            "use_rope_in_attn": True,
            "use_packed_matmul": False,
            "use_matmul_in_attn": False,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "scale": 0.125,
            "softcap": 0.0,
            "qk_norm_epsilon": 1e-6,
            "q_path": "",
            "k_path": "",
            "v_path": "",
        }
        self.rope_attrs = {"op_type": "RotaryEmbedding"}
        self.extra_options = {}


@pytest.mark.parametrize("head_size,expected", [(8, False), (16, True), (24, False), (32, True), (64, True)])
def test_is_fused_rope_supported_cpu(head_size, expected):
    model = _FakeGQAModel(ep="cpu", head_size=head_size)
    assert model.is_fused_rope_supported() is expected


@pytest.mark.parametrize("ep", ["cuda", "webgpu", "dml"])
@pytest.mark.parametrize("head_size", [8, 16, 24, 32])
def test_is_fused_rope_supported_other_providers(ep, head_size):
    model = _FakeGQAModel(ep=ep, head_size=head_size)
    assert model.is_fused_rope_supported() is (ep != "dml")


@pytest.mark.parametrize("head_size,expected_packed,expected_rope_in_attn", [
    (8, False, False),
    (16, True, True),
    (24, False, False),
    (32, True, True),
])
def test_cpu_gqa_packed_matmul_selection(head_size, expected_packed, expected_rope_in_attn):
    model = _FakeGQAModel(ep="cpu", head_size=head_size)
    model.make_attention_init(config=None)

    assert model.attention_attrs["use_rope_in_attn"] is expected_rope_in_attn
    assert model.attention_attrs["use_packed_matmul"] is expected_packed
    if expected_rope_in_attn:
        assert "position_ids" not in model.input_names
    else:
        assert "position_ids" in model.input_names


def test_cpu_gqa_packed_matmul_allowed_when_qk_norm_present():
    # When q_norm and k_norm are enabled, make_attention_input_proj splits the packed projection,
    # so packed matmul remains safe even when unfused RoPE is used.
    model = _FakeGQAModel(ep="cpu", head_size=8, q_norm=True, k_norm=True)
    model.make_attention_init(config=None)
    assert model.attention_attrs["use_rope_in_attn"] is False
    assert model.attention_attrs["use_packed_matmul"] is True


@pytest.mark.parametrize("head_size", [4, 6, 10, 12, 14])
def test_cpu_gqa_rejects_unsupported_head_size_at_build_time(head_size):
    model = _FakeGQAModel(ep="cpu", head_size=head_size)
    with pytest.raises(ValueError, match="CPU GroupQueryAttention requires head_size to be a multiple of 8"):
        model.make_attention_init(config=None)


def _save_local_llama_checkpoint(model_dir, head_dim):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab_size = 256
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=4 * head_dim,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
    )
    hf_model = LlamaForCausalLM(cfg).eval()
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(str(model_dir))

    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, **{f"w{i}": i for i in range(3, vocab_size)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<pad>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
    ).save_pretrained(str(model_dir))

    return hf_model


def _create_and_export_llama_model(head_dim, tmp_path):
    torch = pytest.importorskip("torch")
    import builder

    torch.manual_seed(42)
    model_dir = tmp_path / f"hf_model_{head_dim}"
    hf_model = _save_local_llama_checkpoint(model_dir, head_dim)

    output_dir = tmp_path / f"onnx_{head_dim}"
    cache_dir = tmp_path / f"cache_{head_dim}"

    extra_options = builder.parse_extra_options(
        "", str(model_dir), str(output_dir), "fp32", "cpu", str(cache_dir), None
    )
    builder.create_model(
        "", str(model_dir), str(output_dir), "fp32", "cpu", str(cache_dir), **extra_options
    )

    return hf_model, output_dir


@pytest.mark.parametrize(
    "head_dim,expected_do_rotary,has_standalone_rope",
    [
        (8, 0, True),
        (16, 1, False),
        (24, 0, True),
        (32, 1, False),
    ],
)
def test_cpu_fp32_end_to_end_and_numerical_parity(tmp_path, head_dim, expected_do_rotary, has_standalone_rope):
    torch = pytest.importorskip("torch")
    og = pytest.importorskip("onnxruntime_genai")

    hf_model, output_dir = _create_and_export_llama_model(head_dim, tmp_path)

    # 1. Inspect exported ONNX graph
    onnx_path = output_dir / "model.onnx"
    assert onnx_path.exists()
    model_proto = onnx.load(str(onnx_path), load_external_data=False)

    gqa_nodes = [node for node in model_proto.graph.node if node.op_type == "GroupQueryAttention"]
    assert len(gqa_nodes) == 1
    gqa = gqa_nodes[0]

    do_rotary_attrs = [attr.i for attr in gqa.attribute if attr.name == "do_rotary"]
    assert len(do_rotary_attrs) == 1
    assert do_rotary_attrs[0] == expected_do_rotary

    rope_nodes = [node for node in model_proto.graph.node if node.op_type == "RotaryEmbedding"]
    if has_standalone_rope:
        assert len(rope_nodes) == 2
        rope_outputs = {node.output[0] for node in rope_nodes}
        # Discrete Q and K rotary outputs must directly feed GQA query and key inputs
        assert gqa.input[0] in rope_outputs
        assert gqa.input[1] in rope_outputs
        # GQA value input must be discrete and non-empty
        assert gqa.input[2] != ""
        assert gqa.input[2] not in rope_outputs
        # Both rotary embedding operations must consume position_ids
        for r_node in rope_nodes:
            assert "position_ids" in r_node.input[1]
    else:
        assert len(rope_nodes) == 0

    # 2. Run inference on CPU with onnxruntime-genai (First token / prefill)
    config = og.Config(str(output_dir))
    config.clear_providers()
    genai_model = og.Model(config)
    params = og.GeneratorParams(genai_model)
    params.set_search_options(max_length=32)
    generator = og.Generator(genai_model, params)

    token_ids = list(range(3, 11))
    generator.append_tokens(token_ids)

    got_logits = np.array(generator.get_output("logits"))[0]
    with torch.no_grad():
        want_logits = hf_model(torch.tensor([token_ids])).logits[0].numpy()

    # Numerical agreement check: FP32 differences should be within 1e-4
    max_diff = np.abs(got_logits - want_logits).max()
    assert max_diff < 1e-4, f"Head dim {head_dim} logits diverged: max diff {max_diff}"

    # 3. Step 2 & 3: Decode continuation using KV-cache
    generator.generate_next_token()
    tok1 = int(generator.get_next_tokens()[0])
    with torch.no_grad():
        hf_tok1 = int(np.argmax(hf_model(torch.tensor([token_ids])).logits[0, -1].numpy()))
    assert tok1 == hf_tok1, f"Head dim {head_dim} first decode token mismatch: {tok1} vs {hf_tok1}"

    generator.generate_next_token()
    tok2 = int(generator.get_next_tokens()[0])
    with torch.no_grad():
        hf_tok2 = int(np.argmax(hf_model(torch.tensor([token_ids + [tok1]])).logits[0, -1].numpy()))
    assert tok2 == hf_tok2, f"Head dim {head_dim} second decode token mismatch: {tok2} vs {hf_tok2}"


def test_cpu_build_rejects_unsupported_head_size(tmp_path):
    import builder

    model_dir = tmp_path / "hf_model_12"
    _save_local_llama_checkpoint(model_dir, head_dim=12)

    output_dir = tmp_path / "onnx_12"
    cache_dir = tmp_path / "cache_12"

    extra_options = builder.parse_extra_options(
        "", str(model_dir), str(output_dir), "fp32", "cpu", str(cache_dir), None
    )
    with pytest.raises(ValueError, match="CPU GroupQueryAttention requires head_size to be a multiple of 8, got 12"):
        builder.create_model(
            "", str(model_dir), str(output_dir), "fp32", "cpu", str(cache_dir), **extra_options
        )

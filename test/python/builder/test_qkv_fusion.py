# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import onnx_ir as ir
import pytest
import torch

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parent))


def _load_builder_module(module_name):
    spec = importlib.util.spec_from_file_location(f"models.builders.{module_name}", BUILDERS_DIR / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"models.builders.{module_name}"] = module
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("models", types.ModuleType("models"))
builders_package = sys.modules.setdefault("models.builders", types.ModuleType("models.builders"))
builders_package.__path__ = [str(BUILDERS_DIR)]

base_module = _load_builder_module("base")
qwen_module = _load_builder_module("qwen")
Model = base_module.Model
Qwen35TextModel = qwen_module.Qwen35TextModel

HIDDEN, HEADS, KV_HEADS, HEAD_SIZE = 16, 2, 1, 4
Q_SIZE, KV_SIZE = HEADS * HEAD_SIZE, KV_HEADS * HEAD_SIZE


def _model(model_type, extra_options, use_paged_attention=True):
    model = model_type.__new__(model_type)
    model.ep = "cuda"
    model.io_dtype = ir.DataType.FLOAT16
    model.extra_options = dict(extra_options)
    model.use_paged_attention = use_paged_attention
    model.hidden_rows_dim = "num_tokens"
    model.num_attn_heads, model.num_kv_heads, model.head_size = HEADS, KV_HEADS, HEAD_SIZE
    model.original_context_length = 4096
    model.rope_attrs = {}
    model.matmul_attrs = {"use_lora": False}
    model.input_names = {}
    model.attention_attrs = {
        "q_norm": False,
        "k_norm": False,
        "use_packed_matmul": False,
        "use_matmul_in_attn": False,
        "use_rope_in_attn": False,
        "q_path": "",
        "k_path": "",
        "v_path": "",
    }
    return model


@pytest.mark.parametrize(("extra_options", "expected"), [({}, True), ({"disable_qkv_fusion": True}, False)])
def test_qwen35_paged_packs_qkv_unless_disabled(extra_options, expected):
    model = _model(Qwen35TextModel, extra_options)

    model.make_attention_init(config=None)

    assert model.attention_attrs["op_type"] == "PagedAttention"
    assert model.attention_attrs["use_packed_matmul"] is expected


@pytest.mark.parametrize(("extra_options", "expected"), [({}, True), ({"disable_qkv_fusion": True}, False)])
def test_qwen35_gqa_packs_qkv_unless_disabled(extra_options, expected):
    model = _model(Qwen35TextModel, extra_options, use_paged_attention=False)

    assert model.is_packed_matmul_supported() is expected


def test_other_paged_models_with_qk_norm_keep_separate_projections():
    model = _model(Model, {})
    model.attention_attrs["q_norm"] = model.attention_attrs["k_norm"] = True

    model.make_attention_init(config=None)

    assert model.attention_attrs["use_packed_matmul"] is False


@pytest.mark.parametrize(("marker", "value"), [("quant_type", "fp8"), ("exclude_from_quantization", True)])
def test_natively_quantized_or_excluded_projections_are_not_packed(marker, value):
    model = _model(Qwen35TextModel, {})
    model.make_attention_init(config=None)
    matmuls = []
    model.make_packed_matmul = lambda *args, **kwargs: pytest.fail("Q/K/V must not be packed")
    model.make_matmul = lambda proj, basename, root_input, **kwargs: matmuls.append(basename) or basename
    model.make_split = lambda *args, **kwargs: None
    model.make_reshape = lambda *args, **kwargs: None
    attention = SimpleNamespace(
        q_proj=torch.nn.Linear(HIDDEN, 2 * Q_SIZE, bias=False),
        k_proj=torch.nn.Linear(HIDDEN, KV_SIZE, bias=False),
        v_proj=torch.nn.Linear(HIDDEN, KV_SIZE, bias=False),
    )
    setattr(attention.k_proj, marker, value)

    model.make_attention_input_proj(3, attention, "root")

    assert [name.split("/")[-2] for name in matmuls] == ["q_proj", "k_proj", "v_proj"]


def test_qwen35_fused_qkv_splits_gated_q_before_the_q_gate_split():
    model = _model(Qwen35TextModel, {})
    model.make_attention_init(config=None)
    packed, splits, reshapes = [], [], []

    def make_packed_matmul(self, q, k, v, basename, root_input):
        packed.append(basename)
        return basename

    def make_matmul(self, *args, **kwargs):
        raise AssertionError("Q/K/V must not be emitted as separate MatMuls")

    model.make_packed_matmul = types.MethodType(make_packed_matmul, model)
    model.make_matmul = types.MethodType(make_matmul, model)
    model.make_split = lambda name, inputs, outputs, dtypes, shapes, axis=-1, **_: splits.append((name, inputs, shapes))
    model.make_reshape = lambda name, inputs, dtype, shape: reshapes.append((name, inputs, shape))

    attention = SimpleNamespace(
        q_proj=torch.nn.Linear(HIDDEN, 2 * Q_SIZE, bias=False),
        k_proj=torch.nn.Linear(HIDDEN, KV_SIZE, bias=False),
        v_proj=torch.nn.Linear(HIDDEN, KV_SIZE, bias=False),
    )

    model.make_attention_input_proj(3, attention, "root")

    assert packed == ["/model/layers.3/attn/qkv_proj/MatMul"]
    qkv_split, q_gate_split = splits
    assert qkv_split[0] == "/model/layers.3/attn/qkv_proj/Split"
    assert qkv_split[1] == [
        "/model/layers.3/attn/qkv_proj/MatMul/output_0",
        f"/model/constants/INT64/[{2 * Q_SIZE}, {KV_SIZE}, {KV_SIZE}]",
    ]
    assert qkv_split[2] == [["num_tokens", 2 * Q_SIZE], ["num_tokens", KV_SIZE], ["num_tokens", KV_SIZE]]
    assert reshapes[0][1][0] == "/model/layers.3/attn/qkv_proj/Split/output_0"
    assert q_gate_split[0] == "/model/layers.3/attn/q_gate/Split"
    assert model.attention_attrs["k_path"] == "/model/layers.3/attn/qkv_proj/Split/output_1"
    assert model.attention_attrs["v_path"] == "/model/layers.3/attn/qkv_proj/Split/output_2"
    assert model.attention_attrs["q_path"] == "/model/layers.3/attn/q_proj/Reshape/output_0"

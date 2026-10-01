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

from loaders.base import QuantizedTensorModule  # noqa: E402
from loaders.modelopt import ModeloptModel  # noqa: E402

HIDDEN, HEADS, KV_HEADS, HEAD_SIZE = 16, 2, 1, 4
Q_SIZE, KV_SIZE = HEADS * HEAD_SIZE, KV_HEADS * HEAD_SIZE


def _model(model_type, extra_options, use_paged_attention=True):
    model = model_type.__new__(model_type)
    model.ep = "cuda"
    model.io_dtype = ir.DataType.FLOAT16
    model.onnx_dtype = ir.DataType.FLOAT16
    model.extra_options = dict(extra_options)
    model.use_paged_attention = use_paged_attention
    model.hidden_rows_dim = "num_tokens"
    model.num_layers = 8
    model.num_attn_heads, model.num_kv_heads, model.head_size = HEADS, KV_HEADS, HEAD_SIZE
    model.original_context_length = 4096
    model.rope_attrs = {}
    model.matmul_attrs = {"use_lora": False}
    model.quant_attrs = {"nodes_to_exclude": [], "use_qdq": False}
    model.exact_quant_override_names = set()
    model.int4_customized_weight_config = {}
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


@pytest.mark.parametrize(
    ("extra_options", "expected"),
    [
        ({}, True),
        ({"fuse_qkv": True}, True),
        ({"fuse_qkv": False}, False),
        ({"disable_qkv_fusion": True}, False),
        ({"disable_qkv_fusion": False}, True),
        ({"disable_qkv_fusion": True, "fuse_qkv": True}, True),
    ],
)
def test_qwen35_paged_packs_qkv_unless_disabled(extra_options, expected):
    model = _model(Qwen35TextModel, extra_options)

    model.make_attention_init(config=None)

    assert model.attention_attrs["op_type"] == "PagedAttention"
    assert model.attention_attrs["use_packed_matmul"] is expected


@pytest.mark.parametrize(
    ("extra_options", "expected"),
    [
        ({}, True),
        ({"fuse_qkv": True}, True),
        ({"fuse_qkv": False}, False),
        ({"disable_qkv_fusion": True}, False),
        ({"disable_qkv_fusion": False}, True),
        ({"disable_qkv_fusion": True, "fuse_qkv": True}, True),
    ],
)
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


SEPARATE = ["q_proj", "k_proj", "v_proj"]
PACKED = ["qkv_proj"]


def _float_attention():
    return SimpleNamespace(
        q_proj=torch.nn.Linear(HIDDEN, 2 * Q_SIZE, bias=False),
        k_proj=torch.nn.Linear(HIDDEN, KV_SIZE, bias=False),
        v_proj=torch.nn.Linear(HIDDEN, KV_SIZE, bias=False),
    )


def _gptq_proj(out_features, g_idx=None, group_size=4):
    proj = QuantizedTensorModule()
    proj.qweight = torch.zeros((out_features, HIDDEN // group_size, group_size // 2), dtype=torch.uint8)
    proj.scales = torch.ones((out_features, HIDDEN // group_size))
    proj.qzeros = torch.zeros((out_features, HIDDEN // group_size // 2), dtype=torch.uint8)
    proj.g_idx = g_idx
    proj.in_features, proj.out_features, proj.bits, proj.group_size = HIDDEN, out_features, 4, group_size
    return proj


def _qwen35(extra_options=None):
    model = _model(Qwen35TextModel, extra_options or {})
    model.make_attention_init(config=None)
    return model


def _apply_quant_config(model, quant_config):
    model.quant_config = quant_config
    model.quant_type = None
    model.make_quant_init(SimpleNamespace())


def _emit_input_proj(model, attention, layer_id=3, stub_packing=True):
    matmuls, splits = [], []
    if stub_packing:
        model.make_packed_matmul = lambda q, k, v, basename, root_input: matmuls.append((basename, None)) or basename
    model.make_matmul = lambda proj, basename, root_input, **kwargs: matmuls.append((basename, proj)) or basename
    model.make_split = lambda name, inputs, outputs, dtypes, shapes, axis=-1, **_: splits.append((name, inputs))
    model.make_reshape = lambda *args, **kwargs: None

    model.make_attention_input_proj(layer_id, attention, "root")

    return matmuls, splits


def _projection_names(matmuls):
    return [name.split("/")[-2] for name, _ in matmuls]


def test_modelopt_loaded_qwen35_layer_splits_packed_qkv_by_q_weight_rows():
    prefix = "model.language_model.layers.3.self_attn"
    rows = {"q_proj": 2 * Q_SIZE, "k_proj": KV_SIZE, "v_proj": KV_SIZE, "o_proj": HIDDEN}
    tensors = {f"{prefix}.{name}.weight": torch.randn(n, HIDDEN, dtype=torch.bfloat16) for name, n in rows.items()}
    loader = object.__new__(ModeloptModel)
    loader.get_tensor = tensors.get
    loader.quant_type = "modelopt"
    loader.num_experts = None
    attention = loader.make_layer(3).self_attn
    assert not hasattr(attention.q_proj, "out_features")

    matmuls, splits = _emit_input_proj(_qwen35(), attention, stub_packing=False)

    [(name, packed)] = matmuls
    assert name == "/model/layers.3/attn/qkv_proj/MatMul"
    assert tuple(packed.weight.shape) == (2 * Q_SIZE + 2 * KV_SIZE, HIDDEN)
    assert splits[0][1][1] == f"/model/constants/INT64/[{2 * Q_SIZE}, {KV_SIZE}, {KV_SIZE}]"


@pytest.mark.parametrize(
    "quant_config",
    [
        {"nodes_to_exclude": ["/model/layers.3/attn/k_proj/MatMul"]},
        {
            "weights": {
                "type": "int4",
                "overrides": [{"match": {"name": "/model/layers.3/attn/v_proj/MatMul"}, "type": "int8"}],
            }
        },
        {
            "weights": {
                "type": "int4",
                "overrides": [{"match": {"name": "/model/layers.3/attn/q_proj/MatMul"}, "exclude": True}],
            }
        },
    ],
    ids=["legacy_exclusion", "exact_override", "exact_exclusion"],
)
def test_settings_naming_one_projection_keep_qkv_separate(quant_config):
    model = _qwen35()
    if "weights" in quant_config:
        config = base_module.QuantConfig.from_dict(quant_config)
    else:
        config = base_module.QuantConfig.from_extra_options(quant_config, precision="int4")
    _apply_quant_config(model, config)

    assert _projection_names(_emit_input_proj(model, _float_attention())[0]) == SEPARATE
    assert _projection_names(_emit_input_proj(model, _float_attention(), layer_id=7)[0]) == PACKED


@pytest.mark.parametrize(
    ("k_g_idx", "k_group_size", "expected"),
    [
        (torch.arange(HIDDEN) // 4, 4, PACKED),
        (torch.arange(HIDDEN).flip(0) // 4, 4, SEPARATE),
        (None, 4, SEPARATE),
        (torch.arange(HIDDEN) // 8, 8, SEPARATE),
    ],
    ids=["same_g_idx", "different_g_idx", "missing_g_idx", "different_group_size"],
)
def test_gptq_projections_pack_only_with_matching_group_mapping(k_g_idx, k_group_size, expected):
    q_g_idx = torch.arange(HIDDEN) // 4
    attention = SimpleNamespace(
        q_proj=_gptq_proj(2 * Q_SIZE, q_g_idx),
        k_proj=_gptq_proj(KV_SIZE, k_g_idx, group_size=k_group_size),
        v_proj=_gptq_proj(KV_SIZE, q_g_idx.clone()),
    )

    assert _projection_names(_emit_input_proj(_qwen35(), attention)[0]) == expected


@pytest.mark.parametrize(("use_qdq", "expected"), [(True, "make_matmul_nbits_qdq"), (False, "make_matmul_nbits")])
def test_packed_prequantized_qkv_honors_qdq_format(use_qdq, expected):
    model = _qwen35()
    model.onnx_dtype = ir.DataType.INT4
    model.quant_attrs["use_qdq"] = use_qdq
    emitted = []
    for method in ("make_matmul_nbits_qdq", "make_matmul_nbits"):
        setattr(
            model, method, lambda matmul, basename, root_input, m=method, **_: emitted.append((m, matmul)) or basename
        )
    attention = SimpleNamespace(q_proj=_gptq_proj(2 * Q_SIZE), k_proj=_gptq_proj(KV_SIZE), v_proj=_gptq_proj(KV_SIZE))

    model.make_packed_matmul(attention.q_proj, attention.k_proj, attention.v_proj, "/qkv_proj/MatMul", "root")

    [(method, packed)] = emitted
    assert method == expected
    assert packed.out_features == 2 * Q_SIZE + 2 * KV_SIZE
    assert packed.qweight.shape[0] == 2 * Q_SIZE + 2 * KV_SIZE


def test_qwen35_mixed_layers_keeps_v_only_upgrade():
    model = _qwen35()
    _apply_quant_config(
        model, base_module.QuantConfig.from_extra_options({"matmul_mixed_precision": "mixed_layers:int8"})
    )
    assert "/model/layers.3/attn/v_proj/MatMul" in model.int4_customized_weight_config
    assert "/model/layers.5/attn/v_proj/MatMul" not in model.int4_customized_weight_config

    assert _projection_names(_emit_input_proj(model, _float_attention(), layer_id=3)[0]) == SEPARATE
    assert _projection_names(_emit_input_proj(model, _float_attention(), layer_id=5)[0]) == PACKED
    # Other models keep upgrading the whole packed projection.
    assert Model.is_qkv_projection_packable(model, 3, _float_attention())

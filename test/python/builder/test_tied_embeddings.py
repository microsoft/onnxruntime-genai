from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir
import pytest

BUILDERS_DIR = Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "builders"
sys.path.insert(0, str(BUILDERS_DIR.parents[1]))


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
Model = base_module.Model


def _make_model_for_tied_embeddings(
    *,
    shared_embeddings=None,
    tie_word_embeddings=None,
    onnx_dtype=ir.DataType.FLOAT16,
    op_types=("MatMul", "Gather"),
    nodes_to_exclude=(),
    exclude_embeds=False,
    exclude_lm_head=False,
    prune_lm_head=False,
    int4_algo_config="default",
):
    model = Model.__new__(Model)
    model.extra_options = {"int4_algo_config": int4_algo_config}
    if shared_embeddings is not None:
        model.extra_options["shared_embeddings"] = shared_embeddings
    model.onnx_dtype = onnx_dtype
    model.quant_attrs = {
        "op_types_to_quantize": op_types,
        "nodes_to_exclude": list(nodes_to_exclude),
    }
    model.exclude_embeds = exclude_embeds
    model.exclude_lm_head = exclude_lm_head
    model.prune_lm_head = prune_lm_head

    config = types.SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
    model.make_tied_embeddings_init(config)
    return model


def test_shared_embeddings_option_overrides_config_tie_word_embeddings():
    model = _make_model_for_tied_embeddings(
        shared_embeddings=False,
        tie_word_embeddings=True,
        onnx_dtype=ir.DataType.INT4,
    )

    assert model.quantized_embeds is True
    assert model.quantized_lm_head is True
    assert model.tied_quantized_embeddings is False
    assert model.tied_unquantized_embeddings is False


def test_tie_word_embeddings_defaults_to_false_when_unset_or_none():
    model_unset = _make_model_for_tied_embeddings()
    model_none = _make_model_for_tied_embeddings(tie_word_embeddings=None)

    assert model_unset.tied_quantized_embeddings is False
    assert model_unset.tied_unquantized_embeddings is False
    assert model_none.tied_quantized_embeddings is False
    assert model_none.tied_unquantized_embeddings is False


@pytest.mark.parametrize(
    "onnx_dtype, op_types, nodes_to_exclude, exclude_embeds, exclude_lm_head, prune_lm_head, expected_embeds, expected_lm_head",
    [
        (ir.DataType.INT4, ("MatMul", "Gather"), (), False, False, False, True, True),
        (ir.DataType.UINT4, ("MatMul", "Gather"), (), False, False, False, True, True),
        (ir.DataType.FLOAT16, ("MatMul", "Gather"), (), False, False, False, False, False),
        (ir.DataType.INT4, ("MatMul",), (), False, False, False, False, True),
        (ir.DataType.INT4, ("Gather",), (), False, False, False, True, False),
        (ir.DataType.INT4, ("MatMul", "Gather"), ("/model/embed_tokens/Gather",), False, False, False, False, True),
        (ir.DataType.INT4, ("MatMul", "Gather"), ("/lm_head/MatMul",), False, False, False, True, False),
        (ir.DataType.INT4, ("MatMul", "Gather"), (), True, False, False, False, True),
        (ir.DataType.INT4, ("MatMul", "Gather"), (), False, True, False, True, False),
        (ir.DataType.INT4, ("MatMul", "Gather"), (), False, False, True, True, False),
    ],
)
def test_quantization_eligibility_for_embeddings_and_lm_head(
    onnx_dtype,
    op_types,
    nodes_to_exclude,
    exclude_embeds,
    exclude_lm_head,
    prune_lm_head,
    expected_embeds,
    expected_lm_head,
):
    model = _make_model_for_tied_embeddings(
        shared_embeddings=True,
        tie_word_embeddings=False,
        onnx_dtype=onnx_dtype,
        op_types=op_types,
        nodes_to_exclude=nodes_to_exclude,
        exclude_embeds=exclude_embeds,
        exclude_lm_head=exclude_lm_head,
        prune_lm_head=prune_lm_head,
    )

    assert model.quantized_embeds is expected_embeds
    assert model.quantized_lm_head is expected_lm_head


@pytest.mark.parametrize(
    "quantized_embeds, quantized_lm_head, expected_tied_quantized, expected_tied_unquantized",
    [
        (True, True, True, False),
        (True, False, False, True),
        (False, True, False, True),
        (False, False, False, True),
    ],
)
def test_shared_embeddings_prefers_quantized_path_only_when_both_layers_are_quantized(
    quantized_embeds,
    quantized_lm_head,
    expected_tied_quantized,
    expected_tied_unquantized,
):
    op_types = tuple(op for enabled, op in ((quantized_embeds, "Gather"), (quantized_lm_head, "MatMul")) if enabled)
    model = _make_model_for_tied_embeddings(
        shared_embeddings=True,
        tie_word_embeddings=False,
        onnx_dtype=ir.DataType.INT4,
        op_types=op_types,
    )

    assert model.tied_quantized_embeddings is expected_tied_quantized
    assert model.tied_unquantized_embeddings is expected_tied_unquantized


@pytest.mark.parametrize(
    "int4_algo_config, expected_int4_lm_head, expected_int8_lm_head",
    [
        ("default", False, False),
        ("rtn", True, False),
        ("k_quant", True, False),
        ("k_quant_mixed", False, True),
        ("k_quant_last", False, True),
        ("k_quant_linear", False, True),
        ("rtn_last", False, True),
    ],
)
def test_lm_head_quantized_dtype_flags_derive_from_int4_algo_config(
    int4_algo_config,
    expected_int4_lm_head,
    expected_int8_lm_head,
):
    model = _make_model_for_tied_embeddings(
        shared_embeddings=True,
        tie_word_embeddings=True,
        onnx_dtype=ir.DataType.INT4,
        int4_algo_config=int4_algo_config,
    )

    assert model.int4_lm_head is expected_int4_lm_head
    assert model.int8_lm_head is expected_int8_lm_head


def _make_minimal_model_for_int4_matmul():
    model = Model.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model.quant_attrs = {"accuracy_level": 0}

    model._float_called = False
    model._initializers = []
    model._nodes = []
    model._values = []

    def _make_matmul_float(_matmul, _basename, _root_input, **_kwargs):
        model._float_called = True
        return "float_fallback"

    def _make_initializer(tensor, name, to=None):
        model._initializers.append((name, to, tensor))

    def _make_node(op_type, **kwargs):
        model._nodes.append((op_type, kwargs))

    def _make_value(name, dtype, shape):
        model._values.append((name, dtype, shape))

    model.make_matmul_float = _make_matmul_float
    model.make_initializer = _make_initializer
    model.make_node = _make_node
    model.make_value = _make_value
    return model


def test_int4_matmul_uses_float_fallback_when_model_not_already_quantized():
    model = _make_minimal_model_for_int4_matmul()

    matmul = types.SimpleNamespace(weight=object())
    result = model.make_matmul_int4(matmul, "/lm_head/MatMul", "hidden_states")

    assert result == "float_fallback"
    assert model._float_called is True
    assert model._nodes == []


def test_int4_matmul_emits_matmul_nbits_when_model_already_quantized():
    model = _make_minimal_model_for_int4_matmul()

    matmul = types.SimpleNamespace(
        qweight=object(),
        scales=object(),
        qzeros=object(),
        g_idx=object(),
        bits=4,
        group_size=32,
        in_features=64,
        out_features=128,
    )
    result = model.make_matmul_int4(matmul, "/lm_head/MatMul", "hidden_states")

    assert result == "/lm_head/MatMulNBits"
    assert model._float_called is False
    assert any(op_type == "MatMulNBits" for op_type, _ in model._nodes)
    assert any(name == "lm_head.MatMulNBits.qweight" for name, _, _ in model._initializers)
    assert any(name == "lm_head.MatMulNBits.scales" for name, _, _ in model._initializers)
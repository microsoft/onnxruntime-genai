from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import onnx_ir as ir
import pytest
import torch

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
    "exclude_embeds, exclude_lm_head, prune_lm_head",
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_shared_embeddings_are_disabled_when_embeddings_or_lm_head_are_excluded(
    exclude_embeds,
    exclude_lm_head,
    prune_lm_head,
):
    model = _make_model_for_tied_embeddings(
        shared_embeddings=True,
        tie_word_embeddings=False,
        onnx_dtype=ir.DataType.INT4,
        exclude_embeds=exclude_embeds,
        exclude_lm_head=exclude_lm_head,
        prune_lm_head=prune_lm_head,
    )

    assert model.tied_quantized_embeddings is False
    assert model.tied_unquantized_embeddings is False


@pytest.mark.parametrize(
    "onnx_dtype, op_types, nodes_to_exclude, exclude_embeds, exclude_lm_head, prune_lm_head, int4_algo_config, expected_tied_quantized, expected_tied_unquantized",
    [
        (ir.DataType.INT4, ("MatMul", "Gather"), (), False, False, False, "default", True, False),
        (ir.DataType.INT4, ("MatMul", "Gather"), (), False, False, False, "rtn", True, False),
        (ir.DataType.UINT4, ("MatMul", "Gather"), (), False, False, False, "default", True, False),
        (ir.DataType.UINT4, ("MatMul", "Gather"), (), False, False, False, "k_quant", True, False),
        (ir.DataType.FLOAT16, ("MatMul", "Gather"), (), False, False, False, "default", False, True),
        (ir.DataType.INT4, ("MatMul",), (), False, False, False, "rtn", False, False),
        (ir.DataType.INT4, ("Gather",), (), False, False, False, "rtn", False, False),
        (
            ir.DataType.INT4,
            ("MatMul", "Gather"),
            ("/model/embed_tokens/Gather",),
            False,
            False,
            False,
            "rtn",
            False,
            False,
        ),
        (ir.DataType.INT4, ("MatMul", "Gather"), ("/lm_head/MatMul",), False, False, False, "rtn", False, False),
        (ir.DataType.INT4, ("MatMul", "Gather"), (), True, False, False, "rtn", False, False),
        (ir.DataType.INT4, ("MatMul", "Gather"), (), False, True, False, "rtn", False, False),
        (ir.DataType.INT4, ("MatMul", "Gather"), (), False, False, True, "rtn", False, False),
    ],
)
def test_tied_embedding_path_selection_matches_current_base_logic(
    onnx_dtype,
    op_types,
    nodes_to_exclude,
    exclude_embeds,
    exclude_lm_head,
    prune_lm_head,
    int4_algo_config,
    expected_tied_quantized,
    expected_tied_unquantized,
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
        int4_algo_config=int4_algo_config,
    )

    assert model.tied_quantized_embeddings is expected_tied_quantized
    assert model.tied_unquantized_embeddings is expected_tied_unquantized


@pytest.mark.parametrize(
    "onnx_dtype, op_types, nodes_to_exclude",
    [
        (ir.DataType.INT4, ("MatMul",), ("/lm_head/MatMul",)),
        (ir.DataType.UINT4, ("MatMul",), ("/lm_head/MatMul",)),
        (ir.DataType.INT4, ("MatMul", "Gather"), ("/lm_head/MatMul", "/model/embed_tokens/Gather")),
        (ir.DataType.UINT4, ("MatMul", "Gather"), ("/lm_head/MatMul", "/model/embed_tokens/Gather")),
    ],
)
def test_tied_unquantized_embeddings_can_be_true_in_int4_mode_when_both_quant_paths_are_disabled(
    onnx_dtype,
    op_types,
    nodes_to_exclude,
):
    model = _make_model_for_tied_embeddings(
        shared_embeddings=True,
        tie_word_embeddings=False,
        onnx_dtype=onnx_dtype,
        op_types=op_types,
        nodes_to_exclude=nodes_to_exclude,
        int4_algo_config="rtn",
    )

    assert model.tied_quantized_embeddings is False
    assert model.tied_unquantized_embeddings is True


@pytest.mark.parametrize(
    "quantized_embeds, quantized_lm_head, int4_algo_config, expected_tied_quantized, expected_tied_unquantized",
    [
        (True, True, "default", True, False),
        (True, True, "rtn", True, False),
        (True, True, "rtn_last", True, False),
        (True, True, "k_quant", True, False),
        (True, True, "k_quant_last", True, False),
        (True, True, "k_quant_mixed", True, False),
        (True, True, "k_quant_linear", True, False),
        (True, False, "rtn", False, False),
        (False, True, "rtn", False, False),
        (False, False, "rtn", False, True),
    ],
)
def test_shared_embeddings_prefers_quantized_path_only_when_both_layers_are_quantized(
    quantized_embeds,
    quantized_lm_head,
    int4_algo_config,
    expected_tied_quantized,
    expected_tied_unquantized,
):
    op_types = tuple(op for enabled, op in ((quantized_embeds, "Gather"), (quantized_lm_head, "MatMul")) if enabled)
    model = _make_model_for_tied_embeddings(
        shared_embeddings=True,
        tie_word_embeddings=False,
        onnx_dtype=ir.DataType.INT4,
        op_types=op_types,
        int4_algo_config=int4_algo_config,
    )

    assert model.tied_quantized_embeddings is expected_tied_quantized
    assert model.tied_unquantized_embeddings is expected_tied_unquantized


# fmt: off
_TIED_QUANTIZED_EMBEDDING_WEIGHT_NAME_CASES = [
    ("default", 32, True, 4, "lm_head.MatMul.weight_Q4", "lm_head.MatMul.weight_scales", ""),
    ("default", 32, False, 4, "lm_head.MatMul.weight", "", ""),
    ("rtn", 32, True, 4, "lm_head.MatMul.weight_Q4G32", "lm_head.MatMul.weight_scale", ""),
    ("rtn", 32, False, 4, "lm_head.MatMul.weight_Q4G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("rtn_last", 32, True, 8, "lm_head.MatMul.weight_Q8G32", "lm_head.MatMul.weight_scale", ""),
    ("rtn_last", 32, False, 8, "lm_head.MatMul.weight_Q8G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("k_quant", 32, True, 4, "lm_head.MatMul.weight_Q4G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("k_quant", 32, False, 4, "lm_head.MatMul.weight_Q4G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("k_quant_last", 32, True, 8, "lm_head.MatMul.weight_Q8G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("k_quant_last", 32, False, 8, "lm_head.MatMul.weight_Q8G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("k_quant_mixed", 32, True, 8, "lm_head.MatMul.weight_Q8G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("k_quant_mixed", 32, False, 8, "lm_head.MatMul.weight_Q8G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("k_quant_linear", 32, True, 8, "lm_head.MatMul.weight_Q8G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
    ("k_quant_linear", 32, False, 8, "lm_head.MatMul.weight_Q8G32", "lm_head.MatMul.weight_scale", "lm_head.MatMul.weight_zp"),
]
# fmt: on


@pytest.mark.parametrize(
    "int4_algo_config, matmul_block_size, is_symmetric, expected_bits, expected_weight, expected_scale, expected_zp",
    _TIED_QUANTIZED_EMBEDDING_WEIGHT_NAME_CASES,
)
def test_tied_quantized_embedding_weight_names_cover_all_supported_algorithms(
    int4_algo_config,
    matmul_block_size,
    is_symmetric,
    expected_bits,
    expected_weight,
    expected_scale,
    expected_zp,
):
    model = Model.__new__(Model)
    model.extra_options = {"int4_algo_config": int4_algo_config}
    model.algo_config_name = int4_algo_config
    model.matmul_block_size = matmul_block_size
    model.quant_attrs = {"is_symmetric": is_symmetric}

    bits, weight_name, scale_name, zp_name = model.make_tied_quantized_embedding_input_names()

    assert bits == expected_bits
    assert weight_name == expected_weight
    assert scale_name == expected_scale
    assert zp_name == expected_zp


def test_tied_quantized_embedding_weight_names_raise_for_unknown_algorithm():
    model = Model.__new__(Model)
    model.extra_options = {"int4_algo_config": "unexpected"}
    model.algo_config_name = "unexpected"
    model.matmul_block_size = 32
    model.quant_attrs = {"is_symmetric": True}

    with pytest.raises(AssertionError, match="Unknown quantization algo config name detected"):
        model.make_tied_quantized_embedding_input_names()


def _make_minimal_model_for_quantized_tied_embedding(*, int4_algo_config, is_symmetric=True, quant_type=None):
    model = Model.__new__(Model)
    model.extra_options = {"int4_algo_config": int4_algo_config}
    model.algo_config_name = int4_algo_config
    model.matmul_block_size = 32
    model.hidden_size = 64
    model.vocab_size = 32000
    model.io_dtype = ir.DataType.FLOAT16
    model.quant_attrs = {"is_symmetric": is_symmetric}
    model.quant_type = quant_type
    model.input_names = {"input_ids": "input_ids"}
    model.embed_attrs = {"scale": 1}
    model.layernorm_attrs = {
        "cast": {"use_fp32": False},
        "root_input": "",
        "skip_input": "",
    }
    model.tied_quantized_embeddings = True
    model.tied_unquantized_embeddings = False

    model._reshape_calls = []
    model._node_calls = []

    def _make_reshape(name, inputs, dtype, shape):
        model._reshape_calls.append((name, inputs, dtype, shape))

    def _make_node(op_type, inputs, outputs, name, **kwargs):
        model._node_calls.append((op_type, inputs, outputs, name, kwargs))

    def _make_value(_name, _dtype, shape=None):
        return shape

    model.make_reshape = _make_reshape
    model.make_node = _make_node
    model.make_value = _make_value

    return model


@pytest.mark.parametrize(
    "int4_algo_config, is_symmetric, quant_type, expected_weight_name, expected_scale_name, expected_zp_name, expect_zp_input",
    [
        ("default", True, None, "lm_head.MatMul.weight_Q4", "lm_head.MatMul.weight_scales", None, False),
        ("default", False, None, "lm_head.MatMul.weight", "", None, False),
        ("rtn", True, None, "lm_head.MatMul.weight_Q4G32", "lm_head.MatMul.weight_scale", None, False),
        (
            "rtn",
            False,
            None,
            "lm_head.MatMul.weight_Q4G32",
            "lm_head.MatMul.weight_scale",
            "lm_head.MatMul.weight_zp",
            True,
        ),
        (
            "k_quant",
            True,
            None,
            "lm_head.MatMul.weight_Q4G32",
            "lm_head.MatMul.weight_scale",
            "lm_head.MatMul.weight_zp",
            True,
        ),
        (
            "k_quant",
            False,
            None,
            "lm_head.MatMul.weight_Q4G32",
            "lm_head.MatMul.weight_scale",
            "lm_head.MatMul.weight_zp",
            True,
        ),
    ],
)
def test_make_embedding_uses_algo_specific_lm_head_initializer_names_for_tied_quantized_embeddings(
    int4_algo_config,
    is_symmetric,
    quant_type,
    expected_weight_name,
    expected_scale_name,
    expected_zp_name,
    expect_zp_input,
):
    model = _make_minimal_model_for_quantized_tied_embedding(
        int4_algo_config=int4_algo_config,
        is_symmetric=is_symmetric,
        quant_type=quant_type,
    )

    model.make_embedding(embedding=None)

    assert model._reshape_calls[0][1][0] == expected_weight_name

    gather_calls = [call for call in model._node_calls if call[0] == "GatherBlockQuantized"]
    assert len(gather_calls) == 1
    gather_inputs = gather_calls[0][1]
    if expected_scale_name:  # Only check scale name if it's not empty
        assert expected_scale_name in gather_inputs
    if expected_zp_name is not None:
        assert (expected_zp_name in gather_inputs) is expect_zp_input
    else:
        assert "lm_head.MatMul.weight_zp" not in gather_inputs
        assert "lm_head.MatMul.weight_zero_points" not in gather_inputs


def _make_minimal_model_for_embedding_branches(*, tied_quantized_embeddings=False, tied_unquantized_embeddings=False):
    model = Model.__new__(Model)
    model.hidden_size = 64
    model.vocab_size = 32000
    model.io_dtype = ir.DataType.FLOAT16
    model.input_names = {"input_ids": "input_ids"}
    model.embed_attrs = {"scale": 1}
    model.layernorm_attrs = {
        "cast": {"use_fp32": False},
        "root_input": "",
        "skip_input": "",
    }
    model.tied_quantized_embeddings = tied_quantized_embeddings
    model.tied_unquantized_embeddings = tied_unquantized_embeddings

    model._transpose_calls = []
    model._initializer_calls = []
    model._node_calls = []
    model._value_calls = []

    def _make_transpose(name, root_input, dtype, shape, perm):
        model._transpose_calls.append((name, root_input, dtype, shape, perm))

    def _make_initializer(tensor, name, to=None):
        model._initializer_calls.append((tensor, name, to))

    def _make_node(op_type, **kwargs):
        model._node_calls.append((op_type, kwargs))

    def _make_value(name, dtype, shape):
        model._value_calls.append((name, dtype, shape))

    model.make_transpose = _make_transpose
    model.make_initializer = _make_initializer
    model.make_node = _make_node
    model.make_value = _make_value
    return model


def test_make_embedding_unquantized_tied_path_emits_transpose_and_gather():
    model = _make_minimal_model_for_embedding_branches(
        tied_quantized_embeddings=False,
        tied_unquantized_embeddings=True,
    )

    model.make_embedding(embedding=None)

    assert len(model._transpose_calls) == 1
    transpose_call = model._transpose_calls[0]
    assert transpose_call[0] == "/model/embed_tokens/Transpose"
    assert transpose_call[1] == "lm_head.MatMul.weight"

    gather_calls = [call for call in model._node_calls if call[0] == "Gather"]
    assert len(gather_calls) == 1
    gather_inputs = gather_calls[0][1]["inputs"]
    assert gather_inputs[0] == "/model/embed_tokens/Transpose/output_0"
    assert gather_inputs[1] == "input_ids"

    assert model._initializer_calls == []


def test_make_embedding_non_tied_path_uses_embed_tokens_initializer_and_gather():
    model = _make_minimal_model_for_embedding_branches(
        tied_quantized_embeddings=False,
        tied_unquantized_embeddings=False,
    )
    embedding = object()

    model.make_embedding(embedding=embedding)

    assert len(model._initializer_calls) == 1
    initializer_call = model._initializer_calls[0]
    assert initializer_call[0] is embedding
    assert initializer_call[1] == "model.embed_tokens.weight"
    assert initializer_call[2] == ir.DataType.FLOAT16

    gather_calls = [call for call in model._node_calls if call[0] == "Gather"]
    assert len(gather_calls) == 1
    gather_inputs = gather_calls[0][1]["inputs"]
    assert gather_inputs[0] == "model.embed_tokens.weight"
    assert gather_inputs[1] == "input_ids"

    assert model._transpose_calls == []


def _make_minimal_model_for_int4_matmul():
    model = Model.__new__(Model)
    model.io_dtype = ir.DataType.FLOAT16
    model.quant_attrs = {"accuracy_level": 0, "is_symmetric": True}
    model.matmul_block_size = 32

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


def test_nbits_matmul_defers_float_weight_to_graph_quantizer():
    # Raw float weights are not quantized inside make_matmul_nbits. They are emitted as a
    # float MatMul and quantized later by the graph-level `to_nbits` pass (which honors
    # `int4_algo_config`). Because base.py runs for every EP, the CUDA-only CudaQuantizer
    # must never be invoked here.
    model = _make_minimal_model_for_int4_matmul()

    matmul = types.SimpleNamespace(weight=torch.zeros((128, 64), dtype=torch.float16), in_features=64, out_features=128)
    result = model.make_matmul_nbits(matmul, "/lm_head/MatMul", "hidden_states")

    assert result == "float_fallback"
    assert model._float_called is True
    assert not any(op_type == "MatMulNBits" for op_type, _ in model._nodes)


def test_nbits_matmul_emits_matmul_nbits_when_model_already_quantized():
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
    result = model.make_matmul_nbits(matmul, "/lm_head/MatMul", "hidden_states")

    assert result == "/lm_head/MatMulNBits"
    assert model._float_called is False
    assert any(op_type == "MatMulNBits" for op_type, _ in model._nodes)
    assert any(name == "lm_head.MatMulNBits.qweight" for name, _, _ in model._initializers)
    assert any(name == "lm_head.MatMulNBits.scales" for name, _, _ in model._initializers)

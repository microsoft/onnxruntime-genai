# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from types import SimpleNamespace

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
import torch

from models.builders.base import Model
from models.builders.qwen3_8 import Qwen4ExpEngramModel
from models.builders.qwen3_8.graph import Qwen4ExpGraphModel


@pytest.fixture
def engram_weights():
    embedding = SimpleNamespace(
        layer_multipliers=torch.tensor([1, 3, 5], dtype=torch.int64),
        ngram_heads_vocab_sizes=torch.tensor([7, 11, 13, 17], dtype=torch.int64),
        ngram_heads_offsets=torch.tensor([0, 7, 18, 31], dtype=torch.int64),
        eos_token_id=[0, 99],
        ngram_embedding=SimpleNamespace(
            weight=(torch.arange(96).reshape(48, 2).float() - 48).to(torch.float8_e4m3fn),
            weight_scale=torch.tensor([0.125]),
        ),
    )
    config = SimpleNamespace(ngram_size=3, heads_per_ngram=2, ple_embed_dim=8, ple_layer_ids=[5])
    return SimpleNamespace(ple_embedding=embedding), config


@pytest.fixture(params=[ir.DataType.FLOAT, ir.DataType.FLOAT16], ids=["fp32", "fp16"])
def engram_models(request, engram_weights, tmp_path):
    ple, config = engram_weights
    models = []
    for paged in (False, True):
        model = Qwen4ExpEngramModel(ple, config, request.param, use_paged_attention=paged)
        directory = tmp_path / ("paged" if paged else "dense")
        directory.mkdir()
        model.save_model(directory)
        models.append((model, directory / "engram.onnx"))
    return models


def make_session(path):
    try:
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except (ort.capi.onnxruntime_pybind11_state.Fail, ort.capi.onnxruntime_pybind11_state.NotImplemented) as error:
        if "not a registered function/op" in str(error) or "Could not find an implementation" in str(error):
            pytest.skip(f"Installed ORT lacks standalone Engram kernels: {error}")
        raise


@pytest.fixture
def engram_sessions(engram_models):
    return [make_session(path) for _, path in engram_models]


def run_engram(session, tokens, history):
    return session.run(None, {"input_ids": tokens, "past.4.ple_tokens": history})


@pytest.mark.parametrize("paged", [False, True], ids=["dense", "paged"])
def test_standalone_engram_abi_metadata_and_external_data(engram_models, engram_weights, paged):
    model, path = engram_models[int(paged)]
    assert isinstance(model, Qwen4ExpGraphModel)
    assert isinstance(model, Model)
    assert [value.name for value in model.graph.inputs] == ["input_ids", "past.4.ple_tokens"]
    assert [value.name for value in model.graph.outputs] == ["engram_embeddings", "present.4.ple_tokens"]
    assert [list(value.shape) for value in model.graph.inputs] == [
        ["batch_size", "sequence_length"],
        ["batch_size", 2],
    ]
    assert [list(value.shape) for value in model.graph.outputs] == [
        ["batch_size", "sequence_length", 8],
        ["batch_size", 2],
    ]
    assert [value.dtype for value in model.graph.inputs] == [ir.DataType.INT64, ir.DataType.INT64]
    assert [value.dtype for value in model.graph.outputs] == [model.io_dtype, ir.DataType.INT64]
    exported = onnx.load(path, load_external_data=False)
    assert exported.producer_name == "onnxruntime-genai"
    assert {op.domain: op.version for op in exported.opset_import} == {"": 22, "com.microsoft": 1}
    hash_op = "VarlenNGramHashMapping" if paged else "NGramHashMapping"
    hash_node = next(node for node in exported.graph.node if node.op_type == hash_op)
    assert hash_node.domain == "com.microsoft"
    assert {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in hash_node.attribute} == {
        "max_ngram_size": 3,
        "n_head_per_ngram": 2,
        "pad_id": 0,
        "reset_on_eos": 1,
    }
    assert list(hash_node.output) == [f"/model/ple/{hash_op}/output_0", "present.4.ple_tokens"]
    if paged:
        assert hash_node.input[0] == "/model/ple/packed/Flatten/output_0"
        assert hash_node.input[3] == "/model/ple/packed/CumulativeLengths/output_0"
        assert {"Shape", "Range", "Mul", "Concat"} <= {node.op_type for node in exported.graph.node}
        assert model.values[hash_node.input[3]].dtype == ir.DataType.INT32
    else:
        assert hash_node.input[0] == "input_ids"
    assert list(hash_node.input[-3:]) == [
        "past.4.ple_tokens",
        "model.ple.head_offsets",
        "model.ple.eos_token_id",
    ]
    gather = next(node for node in exported.graph.node if node.op_type == "GatherBlockQuantized")
    assert list(gather.input) == [
        "model.ple.ngram_embedding.weight",
        hash_node.output[0],
        "model.ple.ngram_embedding.weight_scale",
    ]
    assert {prop.key: prop.value for prop in gather.metadata_props} == {"layer_ann": "cpu_embedding"}
    assert not any("Gate" in node.op_type or "Conv" in node.op_type for node in exported.graph.node)
    assert {
        entry.value
        for tensor in exported.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    } == {"engram.onnx.data"}
    assert all(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in exported.graph.initializer)
    assert {file.name for file in path.parent.iterdir()} == {"engram.onnx", "engram.onnx.data"}
    reloaded = ir.load(path)
    for name, value in model.graph.initializers.items():
        np.testing.assert_array_equal(reloaded.graph.initializers[name].const_value.numpy(), value.const_value.numpy())
    table = reloaded.graph.initializers["model.ple.ngram_embedding.weight"]
    expected = ir.tensor(engram_weights[0].ple_embedding.ngram_embedding.weight).numpy()
    np.testing.assert_array_equal(table.const_value.numpy(), expected)
    assert table.dtype == ir.DataType.FLOAT8E4M3FN
    np.testing.assert_array_equal(
        reloaded.graph.initializers["model.ple.ngram_embedding.weight_scale"].const_value.numpy(), [[0.125]]
    )
    onnx.checker.check_model(str(path))


@pytest.mark.parametrize(
    ("tokens", "history"),
    [
        ([[2, 3, 4, 5], [7, 8, 9, 10]], [[11, 12], [13, 14]]),
        ([[2, 0, 4, 5], [0, 8, 0, 10]], [[0, 12], [13, 0]]),
        ([[0], [9]], [[11, 12], [0, 14]]),
    ],
    ids=["history", "eos-reset", "single-token"],
)
def test_standalone_engram_paged_matches_dense_uniform_batch(engram_sessions, tokens, history):
    tokens = np.asarray(tokens, dtype=np.int64)
    history = np.asarray(history, dtype=np.int64)
    dense, paged = [run_engram(session, tokens, history) for session in engram_sessions]
    assert paged[0].shape == (2, tokens.shape[1], 8)
    assert paged[1].shape == (2, 2)
    np.testing.assert_array_equal(paged[0], dense[0])
    np.testing.assert_array_equal(paged[1], dense[1])
    for row in range(2):
        independent = run_engram(engram_sessions[0], tokens[row : row + 1], history[row : row + 1])
        np.testing.assert_array_equal(paged[0][row : row + 1], independent[0])
        np.testing.assert_array_equal(paged[1][row : row + 1], independent[1])


def test_standalone_engram_variable_requests_use_engine_rank_two_abi(engram_sessions):
    requests = [
        ([3], [[7, 8]]),
        ([4, 0, 5, 6, 9], [[0, 12]]),
        ([0, 10], [[13, 0]]),
    ]
    for tokens, history in requests:
        histories = [np.asarray(history, dtype=np.int64) for _ in engram_sessions]
        for chunk in (tokens, [11], [0, 12, 13]):
            input_ids = np.asarray([chunk], dtype=np.int64)
            results = [
                run_engram(session, input_ids, past) for session, past in zip(engram_sessions, histories, strict=True)
            ]
            assert results[1][0].shape == (1, len(chunk), 8)
            assert results[1][1].shape == (1, 2)
            np.testing.assert_array_equal(results[1][0], results[0][0])
            np.testing.assert_array_equal(results[1][1], results[0][1])
            histories = [result[1] for result in results]


def test_standalone_engram_dense_empty_sequence_preserves_history(engram_sessions):
    history = np.asarray([[7, 8], [0, 12]], dtype=np.int64)
    embeddings, present = run_engram(engram_sessions[0], np.empty((2, 0), dtype=np.int64), history)
    assert embeddings.shape == (2, 0, 8)
    np.testing.assert_array_equal(present, history)


def test_standalone_engram_paged_empty_sequence_if_supported(engram_sessions):
    history = np.asarray([[7, 8], [0, 12]], dtype=np.int64)
    try:
        embeddings, present = run_engram(engram_sessions[1], np.empty((2, 0), dtype=np.int64), history)
    except ort.capi.onnxruntime_pybind11_state.Fail as error:
        if "every request must contain a token" in str(error):
            pytest.skip("VarlenNGramHashMapping requires at least one token per request")
        raise
    assert embeddings.shape == (2, 0, 8)
    np.testing.assert_array_equal(present, history)


@pytest.mark.parametrize("scale_dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16], ids=["fp32-scale", "fp16-scale"])
def test_standalone_engram_float_table_and_explicit_scale(engram_weights, tmp_path, scale_dtype):
    ple, config = engram_weights
    ple.ple_embedding.ngram_embedding.weight = ple.ple_embedding.ngram_embedding.weight.float()
    tokens = np.asarray([[2, 3, 0, 4], [5, 0, 6, 7]], dtype=np.int64)
    history = np.asarray([[8, 9], [10, 11]], dtype=np.int64)
    results = []
    for paged in (False, True):
        model = Qwen4ExpEngramModel(ple, config, ir.DataType.FLOAT, scale_dtype=scale_dtype, use_paged_attention=paged)
        scale = model.graph.initializers["model.ple.ngram_embedding.weight_scale"]
        assert scale.dtype == scale_dtype
        assert model.graph.initializers["model.ple.ngram_embedding.weight"].dtype == ir.DataType.FLOAT8E4M3FN
        directory = tmp_path / ("paged" if paged else "dense")
        directory.mkdir()
        model.save_model(directory)
        results.append(run_engram(make_session(directory / "engram.onnx"), tokens, history))
    np.testing.assert_array_equal(results[1][0], results[0][0])
    np.testing.assert_array_equal(results[1][1], results[0][1])


def test_standalone_engram_default_constructor_keeps_dense_hash(engram_weights):
    model = Qwen4ExpEngramModel(*engram_weights, ir.DataType.FLOAT)
    assert [node.op_type for node in model.graph] == [
        "NGramHashMapping",
        "GatherBlockQuantized",
        "Constant",
        "Reshape",
    ]

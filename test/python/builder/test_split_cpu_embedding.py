# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

spec = importlib.util.spec_from_file_location(
    "split_cpu_embedding", Path(__file__).parents[3] / "src" / "python" / "py" / "models" / "split_cpu_embedding.py"
)
split_cpu_embedding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split_cpu_embedding)
convert = split_cpu_embedding.convert
split_graph = split_cpu_embedding.split_graph


def lookup_model(quantized=False, retain_ids=False):
    width = 32
    if quantized:
        weight = TensorProto(name="embedding.weight", data_type=TensorProto.INT4, dims=[8, width])
        weight.raw_data = np.arange(8 * width // 2, dtype=np.uint8).tobytes()
        scales = numpy_helper.from_array(np.full((8, 2), 0.125, dtype=np.float16), "embedding.scales")
        initializers = [weight, scales]
        lookup = helper.make_node(
            "GatherBlockQuantized",
            [weight.name, "input_ids", scales.name],
            ["lookup"],
            domain="com.microsoft",
            gather_axis=0,
            quantize_axis=1,
            block_size=16,
        )
        dtype = TensorProto.FLOAT16
    else:
        weight = numpy_helper.from_array(np.arange(8 * width, dtype=np.float32).reshape(8, width), "embedding.weight")
        initializers = [weight]
        lookup = helper.make_node("Gather", [weight.name, "input_ids"], ["lookup"])
        dtype = TensorProto.FLOAT
    nodes = [lookup, helper.make_node("Identity", ["lookup"], ["hidden"])]
    outputs = [helper.make_tensor_value_info("hidden", dtype, ["tokens", width])]
    if retain_ids:
        nodes.append(helper.make_node("Identity", ["input_ids"], ["selector_ids"]))
        outputs.append(helper.make_tensor_value_info("selector_ids", TensorProto.INT64, ["tokens"]))
    return helper.make_model(
        helper.make_graph(
            nodes,
            "test",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["tokens"])],
            outputs,
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
        ir_version=10,
    )


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("retain_ids", [False, True])
@pytest.mark.parametrize("lookup_output", ["lookup", "inputs_embeds"])
def test_split_preserves_results_and_selector_ids(quantized, retain_ids, lookup_output):
    model = lookup_model(quantized, retain_ids)
    model.graph.node[0].output[0] = lookup_output
    model.graph.node[1].input[0] = lookup_output
    original = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    embedding, removed = split_graph(model, "input_ids", 32)
    onnx.checker.check_model(model)
    onnx.checker.check_model(embedding)
    assert removed == {i.name for i in embedding.graph.initializer}
    assert not model.graph.initializer
    assert ("input_ids" in {i.name for i in model.graph.input}) == retain_ids
    lookup = ort.InferenceSession(embedding.SerializeToString(), providers=["CPUExecutionProvider"])
    consumer = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    for ids in ([0], [7, 3, 3, 0], list(range(8))):
        tokens = np.array(ids, dtype=np.int64)
        expected = original.run(None, {"input_ids": tokens})
        feed = {"inputs_embeds": lookup.run(None, {"input_ids": tokens})[0]}
        if retain_ids:
            feed["input_ids"] = tokens
        actual = consumer.run(None, feed)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)


def test_rejects_embedding_weights_used_by_other_nodes():
    model = lookup_model()
    model.graph.node.append(helper.make_node("Identity", ["embedding.weight"], ["tied_head"]))
    with pytest.raises(ValueError, match="other consumers"):
        split_graph(model, "input_ids", 32)


@pytest.mark.parametrize("captured", ["lookup", "input_ids", "embedding.weight"])
def test_split_handles_nested_graph_captures(captured):
    model = lookup_model()
    dtype = TensorProto.INT64 if captured == "input_ids" else TensorProto.FLOAT
    shape = ["tokens"] if captured == "input_ids" else [8, 32] if captured == "embedding.weight" else ["tokens", 32]
    branch = helper.make_graph(
        [helper.make_node("Identity", [captured], ["branch_output"])],
        "branch", [], [helper.make_tensor_value_info("branch_output", dtype, shape)],
    )
    model.graph.input.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
    model.graph.node.append(helper.make_node("If", ["condition"], ["conditional"], then_branch=branch, else_branch=branch))
    model.graph.output.append(helper.make_tensor_value_info("conditional", dtype, shape))
    if captured == "embedding.weight":
        with pytest.raises(ValueError, match="other consumers"):
            split_graph(model, "input_ids", 32)
        return
    split_graph(model, "input_ids", 32)
    onnx.checker.check_model(model)
    assert ("input_ids" in {value.name for value in model.graph.input}) == (captured == "input_ids")
    expected = "inputs_embeds" if captured == "lookup" else captured
    for attribute in model.graph.node[-1].attribute:
        assert attribute.g.node[0].input[0] == expected


def test_split_ir_contract_and_metadata():
    proto = lookup_model()
    proto.doc_string = "model documentation"
    proto.metadata_props.add(key="model_key", value="model_value")
    proto.graph.doc_string = "graph documentation"
    proto.graph.node[0].doc_string = "lookup documentation"
    proto.graph.initializer[0].doc_string = "weight documentation"
    model = ir.serde.deserialize_model(proto)
    embedding, removed = split_graph(model, "input_ids", 32, output_name="custom_embeds")
    assert isinstance(embedding, ir.Model)
    assert removed == {"embedding.weight"}
    assert model.graph.inputs[0].name == "custom_embeds"
    result = ir.serde.serialize_model(model)
    extracted = ir.serde.serialize_model(embedding)
    assert result.doc_string == proto.doc_string
    assert result.metadata_props == proto.metadata_props
    assert result.graph.doc_string == proto.graph.doc_string
    assert extracted.graph.node[0].doc_string == proto.graph.node[0].doc_string
    assert extracted.graph.initializer[0] == proto.graph.initializer[0]


@pytest.mark.parametrize("collision", ["input", "node", "initializer"])
def test_rejects_embedding_output_name_collisions(collision):
    model = lookup_model()
    if collision == "input":
        model.graph.input.append(helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, ["tokens", 32]))
    elif collision == "node":
        model.graph.node.append(helper.make_node("Identity", ["lookup"], ["inputs_embeds"]))
    else:
        model.graph.initializer.append(numpy_helper.from_array(np.zeros((1, 32), np.float32), "inputs_embeds"))
    with pytest.raises(ValueError, match="Tensor name already exists"):
        split_graph(model, "input_ids", 32)


@pytest.mark.parametrize("decoder_filename", ["model.onnx", "embedding.onnx"])
@pytest.mark.parametrize("rename_drafter_weight", [False, True])
def test_conversion_owns_external_data_and_preserves_runtime_shared_head(
    tmp_path, decoder_filename, rename_drafter_weight
):
    source = tmp_path / "source"
    source.mkdir()
    target = lookup_model(retain_ids=False)
    target.graph.initializer.append(numpy_helper.from_array(np.arange(6, dtype=np.float32), "lm_head.weight"))
    target.metadata_props.add(key="source_key", value="source_value")
    onnx.save_model(
        target,
        source / decoder_filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    drafter = onnx.load(source / decoder_filename, load_external_data=False)
    (source / "alternate.onnx").write_bytes(drafter.SerializeToString())
    (source / "orphan.data").write_bytes(b"stale data")
    (source / "tokenizer.json").write_text("{}")
    if rename_drafter_weight:
        drafter.graph.initializer[0].name = "draft.embedding.weight"
        drafter.graph.node[0].input[0] = "draft.embedding.weight"
    drafter.graph.node.append(helper.make_node("Identity", ["input_ids"], ["selector_ids"]))
    drafter.graph.output.append(helper.make_tensor_value_info("selector_ids", TensorProto.INT64, ["tokens"]))
    (source / "dflash2.onnx").write_bytes(drafter.SerializeToString())
    head = next(value for value in drafter.graph.initializer if value.name == "lm_head.weight")
    head_storage = {entry.key: entry.value for entry in head.external_data}
    shared_head = {
        "name": head.name, "data_file": head_storage["location"],
        "offset": head_storage["offset"], "length": head_storage["length"],
        "data_type": head.data_type, "shape": list(head.dims),
    }
    shared = [{"name": "embedding.weight"}, shared_head]
    config = {
        "model": {
            "decoder": {"filename": decoder_filename, "hidden_size": 32, "shared_initializers": shared},
            "dflash2": {
                "filename": "dflash2.onnx",
                "shared_initializers": [{"name": drafter.graph.initializer[0].name}, shared_head],
            },
        },
        "engine": {"dynamic_batching": {"max_batch_size": 1}},
    }
    (source / "genai_config.json").write_text(json.dumps(config))
    before = (source / decoder_filename).read_bytes()
    original_data = (source / "weights.data").read_bytes()
    destination = convert(source, tmp_path / "split")
    result = json.loads((destination / "genai_config.json").read_text())
    expected_head = {**shared_head, "data_file": decoder_filename + ".data", "offset": "0"}
    assert result["model"]["decoder"]["shared_initializers"] == [expected_head]
    assert result["model"]["dflash2"]["shared_initializers"] == [expected_head]
    assert (source / decoder_filename).read_bytes() == before
    assert (source / "weights.data").read_bytes() == original_data
    assert (destination / "weights.data").read_bytes() == original_data
    assert (destination / "weights.data").stat().st_ino != (source / "weights.data").stat().st_ino
    assert not (destination / "orphan.data").exists()
    assert (destination / "tokenizer.json").stat().st_ino == (source / "tokenizer.json").stat().st_ino
    graph_paths = list(destination.glob("*.onnx"))
    assert {path.name for path in destination.glob("*.data")} == {
        *(path.name + ".data" for path in graph_paths if path.name != "alternate.onnx"), "weights.data"
    }
    for path in graph_paths:
        if path.name == "alternate.onnx":
            assert path.read_bytes() == (source / path.name).read_bytes()
            np.testing.assert_array_equal(
                ir.load(path).graph.initializers["lm_head.weight"].const_value.numpy(), np.arange(6, dtype=np.float32)
            )
            continue
        graph = onnx.load(path, load_external_data=False)
        assert all(
            next(entry.value for entry in tensor.external_data if entry.key == "location") == path.name + ".data"
            for tensor in graph.graph.initializer
        )
        loaded = onnx.load(path)
        for tensor in loaded.graph.initializer:
            if tensor.name == "lm_head.weight":
                np.testing.assert_array_equal(numpy_helper.to_array(tensor), np.arange(6, dtype=np.float32))
        if path.name != result["model"]["embedding"]["filename"]:
            assert graph.metadata_props == target.metadata_props
    embedding_filename = result["model"]["embedding"]["filename"]
    assert embedding_filename != decoder_filename
    session = ort.InferenceSession(str(destination / embedding_filename), providers=["CPUExecutionProvider"])
    rows = session.run(None, {"input_ids": np.array([0, 7], np.int64)})[0]
    consumer = ort.InferenceSession(str(destination / decoder_filename), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(consumer.run(None, {"inputs_embeds": rows})[0], rows)
    # Same-shaped but independently stored drafter weights cannot silently become shared.
    altered = copy.deepcopy(drafter)
    for entry in altered.graph.initializer[0].external_data:
        if entry.key == "offset":
            entry.value = "4"
    (source / "dflash2.onnx").write_bytes(altered.SerializeToString())
    with pytest.raises(ValueError, match="share identical"):
        convert(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_conversion_rejects_data_collision_with_unchanged_prebuilt_graph(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = lookup_model(retain_ids=False)
    onnx.save_model(
        model, source / "model.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
        location="model.onnx.data", size_threshold=0,
    )
    (source / "prebuilt.onnx").write_bytes((source / "model.onnx").read_bytes())
    (source / "genai_config.json").write_text(json.dumps({
        "model": {"decoder": {"filename": "model.onnx", "hidden_size": 32}},
        "engine": {"dynamic_batching": {"max_batch_size": 1}},
    }))
    original_model = (source / "prebuilt.onnx").read_bytes()
    original_data = (source / "model.onnx.data").read_bytes()

    with pytest.raises(ValueError, match="overwrite unchanged model data"):
        convert(source, tmp_path / "split")

    assert not (tmp_path / "split").exists()
    assert not (tmp_path / "split.saving").exists()
    assert (source / "prebuilt.onnx").read_bytes() == original_model
    assert (source / "model.onnx.data").read_bytes() == original_data


def test_rejects_unsupported_embedding_output_type():
    model = lookup_model()
    model.graph.initializer[0].CopyFrom(
        numpy_helper.from_array(np.arange(8 * 32, dtype=np.int64).reshape(8, 32), "embedding.weight")
    )
    with pytest.raises(ValueError, match="output must be float32, float16, or bfloat16"):
        split_graph(model, "input_ids", 32)


def test_rejects_missing_quantized_scales():
    model = lookup_model(quantized=True)
    del model.graph.node[0].input[2:]
    with pytest.raises(ValueError, match="scales must be an initializer"):
        split_graph(model, "input_ids", 32)

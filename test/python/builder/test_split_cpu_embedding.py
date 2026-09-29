# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
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
def test_conversion_preserves_external_offsets_and_shared_head(tmp_path, decoder_filename, rename_drafter_weight):
    source = tmp_path / "source"
    source.mkdir()
    target = lookup_model(retain_ids=False)
    onnx.save_model(
        target,
        source / decoder_filename,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    drafter = onnx.load(source / decoder_filename, load_external_data=False)
    if rename_drafter_weight:
        drafter.graph.initializer[0].name = "draft.embedding.weight"
        drafter.graph.node[0].input[0] = "draft.embedding.weight"
    drafter.graph.node.append(helper.make_node("Identity", ["input_ids"], ["selector_ids"]))
    drafter.graph.output.append(helper.make_tensor_value_info("selector_ids", TensorProto.INT64, ["tokens"]))
    (source / "dflash2.onnx").write_bytes(drafter.SerializeToString())
    shared = [{"name": "embedding.weight"}, {"name": "lm_head.weight"}]
    config = {
        "model": {
            "decoder": {"filename": decoder_filename, "hidden_size": 32, "shared_initializers": shared},
            "dflash2": {
                "filename": "dflash2.onnx",
                "shared_initializers": [{"name": drafter.graph.initializer[0].name}, {"name": "lm_head.weight"}],
            },
        },
        "engine": {"dynamic_batching": {"max_batch_size": 1}},
    }
    (source / "genai_config.json").write_text(json.dumps(config))
    before = (source / decoder_filename).read_bytes()
    destination = convert(source, tmp_path / "split")
    result = json.loads((destination / "genai_config.json").read_text())
    assert result["model"]["decoder"]["shared_initializers"] == [{"name": "lm_head.weight"}]
    assert result["model"]["dflash2"]["shared_initializers"] == [{"name": "lm_head.weight"}]
    assert (source / decoder_filename).read_bytes() == before
    assert (destination / "weights.data").stat().st_ino == (source / "weights.data").stat().st_ino
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

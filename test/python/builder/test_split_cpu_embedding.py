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
def test_split_preserves_results_and_selector_ids(quantized, retain_ids):
    model = lookup_model(quantized, retain_ids)
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


def test_conversion_preserves_external_offsets_and_shared_head(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = lookup_model(retain_ids=False)
    onnx.save_model(
        target,
        source / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )
    drafter = onnx.load(source / "model.onnx", load_external_data=False)
    drafter.graph.node.append(helper.make_node("Identity", ["input_ids"], ["selector_ids"]))
    drafter.graph.output.append(helper.make_tensor_value_info("selector_ids", TensorProto.INT64, ["tokens"]))
    (source / "dflash2.onnx").write_bytes(drafter.SerializeToString())
    shared = [{"name": "embedding.weight"}, {"name": "lm_head.weight"}]
    config = {
        "model": {
            "decoder": {"filename": "model.onnx", "hidden_size": 32, "shared_initializers": shared},
            "dflash2": {"filename": "dflash2.onnx", "shared_initializers": shared},
        },
        "engine": {"dynamic_batching": {"max_batch_size": 1}},
    }
    (source / "genai_config.json").write_text(json.dumps(config))
    before = (source / "model.onnx").read_bytes()
    destination = convert(source, tmp_path / "split")
    result = json.loads((destination / "genai_config.json").read_text())
    assert result["model"]["decoder"]["shared_initializers"] == [{"name": "lm_head.weight"}]
    assert result["model"]["dflash2"]["shared_initializers"] == [{"name": "lm_head.weight"}]
    assert (source / "model.onnx").read_bytes() == before
    assert (destination / "weights.data").stat().st_ino == (source / "weights.data").stat().st_ino
    session = ort.InferenceSession(str(destination / "embedding.onnx"), providers=["CPUExecutionProvider"])
    assert session.run(None, {"input_ids": np.array([0, 7], np.int64)})[0].shape == (2, 32)
    # Same-shaped but independently stored drafter weights cannot silently become shared.
    altered = copy.deepcopy(drafter)
    for entry in altered.graph.initializer[0].external_data:
        if entry.key == "offset":
            entry.value = "4"
    (source / "dflash2.onnx").write_bytes(altered.SerializeToString())
    with pytest.raises(ValueError, match="share identical"):
        convert(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()

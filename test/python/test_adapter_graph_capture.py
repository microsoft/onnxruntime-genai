# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import re
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_genai as og
import pytest


@pytest.fixture
def capture_messages():
    messages = []
    og.set_log_options(enabled=True, graph_capture=True)
    og.set_log_callback(messages.append)
    try:
        yield messages
    finally:
        og.set_log_callback(None)
        og.set_log_options(enabled=False, graph_capture=False)


@pytest.mark.graph_capture
@pytest.mark.parametrize("device", ["cuda", "webgpu"])
@pytest.mark.parametrize("capture", [False, True])
def test_activate_adapter_after_decode(tmp_path, qwen_graph_for, device, capture, capture_messages):
    if not getattr(og, f"is_{device}_available")():
        pytest.skip(f"{device} is unavailable")
    source = qwen_graph_for(device)
    model_path = tmp_path / "adapters"
    shutil.copytree(source, model_path)
    config_path = model_path / "genai_config.json"
    config = json.loads(config_path.read_text())
    decoder = config["model"]["decoder"]
    logits_name = decoder["outputs"]["logits"]
    proto_path = model_path / decoder["filename"]
    proto = onnx.load(proto_path, load_external_data=False)
    logits_type = next(output for output in proto.graph.output if output.name == logits_name).type.tensor_type.elem_type
    dtype = onnx.helper.tensor_dtype_to_np_dtype(logits_type)
    vocab_size = config["model"]["vocab_size"]
    for node in proto.graph.node:
        for index, name in enumerate(node.output):
            if name == logits_name:
                node.output[index] = "base_logits"
    for index, name in enumerate(("adapter_a", "adapter_b")):
        proto.graph.input.append(onnx.helper.make_tensor_value_info(name, logits_type, [vocab_size]))
        proto.graph.initializer.append(onnx.numpy_helper.from_array(np.zeros(vocab_size, dtype=dtype), name))
        bias = np.full(vocab_size, 0.25 * (index + 1), dtype=dtype)
        adapter = ort.AdapterFormat()
        adapter.set_adapter_version(1)
        adapter.set_model_version(1)
        adapter.set_parameters({name: ort.OrtValue.ortvalue_from_numpy(bias)})
        adapter.export_adapter(str(model_path / f"{name}.onnx_adapter"))
    proto.graph.node.extend(
        [
            onnx.helper.make_node("Add", ["base_logits", "adapter_a"], ["adapted_logits"]),
            onnx.helper.make_node("Add", ["adapted_logits", "adapter_b"], [logits_name]),
        ]
    )
    onnx.save(proto, proto_path)
    session_options = decoder.setdefault("session_options", {})
    session_options["provider_options"] = [
        {device: {("enableGraphCapture" if device == "webgpu" else "enable_cuda_graph"): str(int(capture))}}
    ]
    config_path.write_text(json.dumps(config))
    model = og.Model(str(model_path))
    adapters = og.Adapters(model)
    for name in ("adapter_a", "adapter_b"):
        adapters.load(str(model_path / f"{name}.onnx_adapter"), name)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=32, batch_size=1, do_sample=False)

    def logits(generator):
        generator.append_tokens(np.array([1, 2, 3, 4], dtype=np.int32))
        rows = []
        for step in range(6):
            rows.append(generator.get_logits().copy())
            if step != 5:
                generator.append_tokens(np.array([5 + step], dtype=np.int32))
        return np.stack(rows)

    # Capture the base path before activation, then capture/replay each new
    # adapter set on the same generator. Forced tokens keep every prefix equal.
    generator = og.Generator(model, params)
    base = logits(generator)
    active = []
    for name in ("adapter_a", "adapter_b"):
        generator.rewind_to(0)
        generator.set_active_adapter(adapters, name)
        active.append(name)
        actual = logits(generator)
        reference = og.Generator(model, params)
        for active_name in active:
            reference.set_active_adapter(adapters, active_name)
        expected = logits(reference)
        np.testing.assert_array_equal(actual, expected)
        assert not np.array_equal(actual, base)
        generator.rewind_to(0)
        np.testing.assert_array_equal(logits(generator), expected)
        del reference
    # A new base generator sharing the session must not reuse either active capture.
    base_generator = og.Generator(model, params)
    np.testing.assert_array_equal(logits(base_generator), base)
    del base_generator, generator
    for name in ("adapter_a", "adapter_b"):
        adapters.unload(name)
    if capture:
        entries = re.findall(r"State graph id=(\d+) length=1 variant=0 adapters=(\d+)", "".join(capture_messages))
        # The late-activation generator, two active reference generators, and
        # a new base generator each reserve a base ID. Active sets add four IDs.
        assert len({graph_id for graph_id, _ in entries}) == 8
        assert [int(count) for _, count in entries].count(0) == 4
        assert [int(count) for _, count in entries].count(1) == 2
        assert [int(count) for _, count in entries].count(2) == 2

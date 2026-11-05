# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import json
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime_genai as og
import pytest


def add_recurrent_state(proto, decoder, logits_type, dtype, vocab_size):
    # Both states advance on every forward and affect logits, exposing extra capture-time runs.
    state_vectors = []
    for kind, shape in (("conv", ["batch_size", 1, 1]), ("recurrent", ["batch_size", 1, 1, 1])):
        past = f"past_key_values.0.{kind}_state"
        present = f"present.0.{kind}_state"
        decoder["inputs"][f"past_{kind}_names"] = f"past_key_values.%d.{kind}_state"
        decoder["outputs"][f"present_{kind}_names"] = f"present.%d.{kind}_state"
        proto.graph.input.append(onnx.helper.make_tensor_value_info(past, logits_type, shape))
        proto.graph.output.append(onnx.helper.make_tensor_value_info(present, logits_type, shape))
        vector = f"{kind}_state_vector"
        proto.graph.node.extend(
            [
                onnx.helper.make_node("Add", [past, "state_increment"], [present]),
                onnx.helper.make_node("Flatten", [present], [vector], axis=1),
            ]
        )
        state_vectors.append(vector)
    proto.graph.initializer.extend(
        [
            onnx.numpy_helper.from_array(np.array(0.125, dtype=dtype), "state_increment"),
            onnx.numpy_helper.from_array(np.linspace(-1, 1, vocab_size, dtype=dtype).reshape(1, -1), "state_weights"),
            onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "state_logits_axes"),
        ]
    )
    proto.graph.node.extend(
        [
            onnx.helper.make_node("Add", state_vectors, ["combined_state"]),
            onnx.helper.make_node("MatMul", ["combined_state", "state_weights"], ["state_logits_2d"]),
            onnx.helper.make_node("Unsqueeze", ["state_logits_2d", "state_logits_axes"], ["state_logits"]),
            onnx.helper.make_node("Add", ["base_logits", "state_logits"], ["recurrent_logits"]),
        ]
    )


def create_adapter_model(tmp_path, source, recurrent=False):
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
    if recurrent:
        add_recurrent_state(proto, decoder, logits_type, dtype, vocab_size)
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
            onnx.helper.make_node(
                "Add", ["recurrent_logits" if recurrent else "base_logits", "adapter_a"], ["adapted_logits"]
            ),
            onnx.helper.make_node("Add", ["adapted_logits", "adapter_b"], [logits_name]),
        ]
    )
    onnx.save(proto, proto_path)
    config.setdefault("search", {})["past_present_share_buffer"] = True
    config_path.write_text(json.dumps(config))
    return model_path


def create_model(model_path, device, capture):
    config = og.Config(str(model_path))
    config.clear_providers()
    config.append_provider(device)
    option = "enableGraphCapture" if device == "webgpu" else "enable_cuda_graph"
    config.set_provider_option(device, option, str(int(capture)))
    return og.Model(config)


def load_adapters(model, model_path):
    adapters = og.Adapters(model)
    for name in ("adapter_a", "adapter_b"):
        adapters.load(str(model_path / f"{name}.onnx_adapter"), name)
    return adapters


def logits(generator, tokens=(1, 2, 3, 4)):
    generator.append_tokens(np.array(tokens, dtype=np.int32))
    rows = []
    for step in range(6):
        rows.append(generator.get_logits().copy())
        if step != 5:
            generator.append_tokens(np.array([5 + step], dtype=np.int32))
    return np.stack(rows)


@pytest.mark.graph_capture
@pytest.mark.parametrize("device", ["cuda", "webgpu"])
@pytest.mark.parametrize("capture", [False, True])
def test_activate_adapter_after_decode(tmp_path, qwen_graph_for, device, capture):
    if not getattr(og, f"is_{device}_available")():
        pytest.skip(f"{device} is unavailable")
    model_path = create_adapter_model(tmp_path, qwen_graph_for(device))
    model = create_model(model_path, device, capture)
    adapters = load_adapters(model, model_path)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=32, batch_size=1, do_sample=False)

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


@pytest.mark.graph_capture
@pytest.mark.parametrize("device", ["cuda", "webgpu"])
@pytest.mark.parametrize("recurrent", [False, True], ids=["kv_cache", "recurrent_state"])
def test_activate_adapter_without_rewind(tmp_path, qwen_graph_for, device, recurrent, monkeypatch):
    if not getattr(og, f"is_{device}_available")():
        pytest.skip(f"{device} is unavailable")
    # CUDA aliases recurrent inputs/outputs, exercising save/restore on every new capture ID.
    # WebGPU instead exercises both directions of its recurrent-state double buffers.
    monkeypatch.setenv("ORTGENAI_SHARE_RECURRENT_STATE_UNDER_GRAPH_CAPTURE", "1")
    model_path = create_adapter_model(tmp_path, qwen_graph_for(device), recurrent)
    eager_model = create_model(model_path, device, False)
    captured_model = create_model(model_path, device, True)
    eager_adapters = load_adapters(eager_model, model_path)
    captured_adapters = load_adapters(captured_model, model_path)
    eager_params = og.GeneratorParams(eager_model)
    captured_params = og.GeneratorParams(captured_model)
    for params in (eager_params, captured_params):
        params.set_search_options(max_length=32, batch_size=1, do_sample=False, past_present_share_buffer=True)
    eager = og.Generator(eager_model, eager_params)
    captured = og.Generator(captured_model, captured_params)

    # Both references build the same base prefix; activation preserves those existing caches.
    np.testing.assert_array_equal(logits(captured), logits(eager))
    for name in ("adapter_a", "adapter_b"):
        eager.set_active_adapter(eager_adapters, name)
        captured.set_active_adapter(captured_adapters, name)
        np.testing.assert_array_equal(logits(captured, (10,)), logits(eager, (10,)))

    del captured, eager
    for name in ("adapter_a", "adapter_b"):
        eager_adapters.unload(name)
        captured_adapters.unload(name)

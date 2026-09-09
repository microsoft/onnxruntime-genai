# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""Real GQA cache, asynchronous lifetime, and actual GPU capture workloads.

Run with ORTGENAI_MULTIMODAL_TEST_EPS=cpu,cuda (or cpu,webgpu). An explicitly
requested missing provider fails through the common EP helper. The native
MultimodalDeviceTests target supplements these numerical/partition checks with
allocator, pointer identity, decoder identity, and internal capture assertions.
"""

from __future__ import annotations

import gc
import json

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
from _test_utils import MULTIMODAL_EP_NAMES, multimodal_test_devices, require_execution_provider
from create.create_multimodal_gqa_model import CPU_METADATA_NODES, GPU_DEVICES, create_model, supported_dtypes
from create.create_multimodal_turn_test_model import (
    EOS_TOKEN_ID,
    IMAGE_TOKEN_ID,
    QWEN_FAMILIES,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
)
from onnx import numpy_helper
from test_multimodal_turns import _arrays, _generator, _History, _image_turn, _last_logits, _named

CASES = [
    pytest.param(device, dtype, id=f"{device}-{dtype}")
    for device in multimodal_test_devices()
    for dtype in supported_dtypes(device)
]


def _tolerance(dtype):
    return {"rtol": 0.004, "atol": 0.004} if dtype == "fp16" else {"rtol": 2e-5, "atol": 2e-5}


@pytest.fixture
def gqa_model_factory(tmp_path, device, dtype, request):
    require_execution_provider(device)
    if device not in ("cpu", *GPU_DEVICES):
        pytest.skip(f"GQA multimodal fixture kernel compatibility is not established for {device}")
    models = {}

    def make(family="phi3v", *, shared=False, capture=False, asynchronous=False):
        directory = create_model(
            tmp_path / str(len(models)),
            family,
            device=device,
            dtype=dtype,
            shared=shared,
            capture=capture,
            profile=True,
        )
        if asynchronous:
            config_path = directory / "genai_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for role in ("vision", "embedding", "decoder"):
                config["model"][role]["run_options"] = {"disable_synchronize_execution_providers": "1"}
            config_path.write_text(json.dumps(config), encoding="utf-8")
        models[directory] = og.Model(str(directory))
        return directory, models[directory]

    failures = request.session.testsfailed
    yield make
    directories = list(models)
    models.clear()
    gc.collect()
    # Failed-test tracebacks can retain models and delay profile finalization.
    if request.session.testsfailed == failures:
        for directory in directories:
            assert_profile_partitions(directory, device)


def _pixels_dtype(arrays, dtype):
    if "pixel_values" in arrays:
        arrays["pixel_values"] = arrays["pixel_values"].astype(np.float16 if dtype == "fp16" else np.float32)
    return arrays


def assert_profile_partitions(directory, device):
    """Reject arbitrary CPU numerical fallback; transfers are not model kernels."""
    expected_provider = MULTIMODAL_EP_NAMES[device][1]
    for role in ("vision", "embedding", "decoder"):
        profiles = list(directory.glob(f"profile_{role}_*.json"))
        assert profiles, f"No execution profile for {role}"
        kernels = []
        for path in profiles:
            events = json.loads(path.read_text(encoding="utf-8"))
            kernels.extend(e for e in events if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
        assert kernels, f"No executed {role} kernels"
        numerical = 0
        for event in kernels:
            args = event["args"]
            name = event["name"].removesuffix("_kernel_time")
            provider = args["provider"]
            op = args["op_name"]
            if op in ("MemcpyFromHost", "MemcpyToHost"):
                continue
            if role == "decoder" and CPU_METADATA_NODES.get(name) == op:
                assert provider in ("CPUExecutionProvider", expected_provider), event
                continue
            assert provider == expected_provider, f"Unexpected numerical partition: {role}: {event}"
            numerical += 1
        assert numerical, f"No numerical {role} work on {expected_provider}"
        if role == "decoder":
            assert any(e["args"]["op_name"] == "GroupQueryAttention" for e in kernels), "GQA did not execute"


def test_partition_audit_rejects_numerical_cpu_fallback(tmp_path):
    def event(name, op, provider="CUDAExecutionProvider"):
        return {"cat": "Node", "name": name + "_kernel_time", "args": {"op_name": op, "provider": provider}}

    for role in ("vision", "embedding", "decoder"):
        kernels = [event("projection", "MatMul")]
        if role == "decoder":
            kernels += [
                event("GQA_0", "GroupQueryAttention"),
                event("metadata.mask_shape", "Shape", "CPUExecutionProvider"),
                event("decoder.q", "MatMul", "CPUExecutionProvider"),
            ]
        (tmp_path / f"profile_{role}_test.json").write_text(json.dumps(kernels), encoding="utf-8")
    with pytest.raises(AssertionError, match="Unexpected numerical partition: decoder"):
        assert_profile_partitions(tmp_path, "cuda")


def _numpy_last_logits(directory, history):
    """Independent single-query causal GQA oracle (no ORT/reference replay)."""
    weights = {
        value.name: numpy_helper.to_array(value).astype(np.float32)
        for value in onnx.load(directory / "decoder.onnx").graph.initializer
    }
    embeddings = []
    for ids, pixels in history.turns:
        values = np.broadcast_to((np.asarray(ids) * 0.02 + 0.03)[:, None], (len(ids), 64)).copy()
        if pixels is not None:
            features = (pixels.sum(axis=1) + 11)[:, None] * np.linspace(-0.01, 0.02, 64)[None, :]
            mask = np.asarray(ids) < 0 if history.family == "phi3v" else np.asarray(ids) == IMAGE_TOKEN_ID
            values[mask] = features
        embeddings.extend(values)
    _, positions = history.embeddings_and_positions()
    pos = positions @ np.array([1, 2, 4]) if history.family in QWEN_FAMILIES else positions[:, 0]
    hidden = np.asarray(embeddings, dtype=np.float32) + pos[:, None] * weights["position_projection"]
    query = (hidden[-1:] @ weights["q.weight"]).reshape(4, 16)
    key = (hidden @ weights["k.weight"]).reshape(-1, 2, 16)
    value = (hidden @ weights["v.weight"]).reshape(-1, 2, 16)
    heads = []
    for head in range(4):
        scores = key[:, head // 2] @ query[head] / 4
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        heads.append(probabilities @ value[:, head // 2])
    return np.concatenate(heads) @ weights["lm_head.weight"] + weights["lm_head.bias"]


@pytest.mark.parametrize("device", GPU_DEVICES)
def test_gqa_fixture_is_deterministic_nonrecurrent_and_disables_child_capture(tmp_path, device):
    directories = [
        create_model(tmp_path / name, device=device, dtype="fp16", shared=True, capture=True)
        for name in ("first", "second")
    ]
    for role in ("vision", "embedding", "decoder"):
        assert (directories[0] / f"{role}.onnx").read_bytes() == (directories[1] / f"{role}.onnx").read_bytes()
    decoder = onnx.load(directories[0] / "decoder.onnx").graph
    attention = [node for node in decoder.node if node.op_type == "GroupQueryAttention"]
    assert len(attention) == 1
    assert attention[0].domain == "com.microsoft"
    assert list(attention[0].output[1:]) == ["present.0.key", "present.0.value"]
    assert all(node.op_type != "Concat" for node in decoder.node)
    assert not any("state" in value.name for value in (*decoder.input, *decoder.output))
    mask = next(value for value in decoder.input if value.name == "attention_mask")
    assert mask.type.tensor_type.shape.dim[1].dim_value == 192
    config = json.loads((directories[0] / "genai_config.json").read_text(encoding="utf-8"))
    capture_key = "enable_cuda_graph" if device == "cuda" else "enableGraphCapture"
    for role in ("vision", "embedding", "decoder"):
        options = config["model"][role]["session_options"]["provider_options"][0][device]
        assert options[capture_key] == ("1" if role == "decoder" else "0")


@pytest.mark.parametrize("device,dtype", CASES)
@pytest.mark.parametrize("family", ("phi3v", "qwen2_5_vl", "mistral3"))
@pytest.mark.parametrize("shared", (False, True), ids=("dynamic", "shared"))
def test_gqa_retained_turns_match_numpy_and_full_prefix(gqa_model_factory, dtype, family, shared):
    directory, model = gqa_model_factory(family, shared=shared)
    generator = _generator(model)
    history = _History(family)
    for value in (1, 7, 13):
        generator.set_inputs(_named(_pixels_dtype(history.image(value), dtype)))
        np.testing.assert_allclose(_last_logits(generator), _numpy_last_logits(directory, history), **_tolerance(dtype))
        reference = _generator(model, past_present_share_buffer=not shared)
        reference.set_inputs(_named(_pixels_dtype(history.prefix_arrays(), dtype)))
        np.testing.assert_allclose(_last_logits(generator), _last_logits(reference), **_tolerance(dtype))
        del reference
        for _ in range(4):
            generator.generate_next_token()
            history.text([int(generator.get_sequence(0)[-1])])
        generator.append_tokens(np.array([[5, 9]], dtype=np.int32))
        history.text([5, 9])
        np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
        np.testing.assert_allclose(_last_logits(generator), _numpy_last_logits(directory, history), **_tolerance(dtype))


def _unequal_grid_turn(value, dtype, family="qwen2_5_vl"):
    grids = np.array([[1, 1, 2], [1, 2, 3]], dtype=np.int64)
    pixels = np.arange(24, dtype=np.float32).reshape(8, 3) + value
    if family == "mistral3":
        padded = np.zeros((2, 3, 2, 3), dtype=np.float32)
        padded[0, :, :1, :2] = pixels[:2].reshape(1, 2, 3).transpose(2, 0, 1)
        padded[1] = pixels[2:].reshape(2, 3, 3).transpose(2, 0, 1)
        return _pixels_dtype(
            {
                "input_ids": np.array([[2, *([IMAGE_TOKEN_ID] * 8), 3]], dtype=np.int32),
                "pixel_values": padded,
                "image_sizes": grids[:, 1:].copy(),
                "num_image_tokens": np.array([2, 6], dtype=np.int64),
            },
            dtype,
        )
    ids = [2]
    for count in (2, 6):
        ids += [VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * count), VISION_END_TOKEN_ID]
    ids += [3]
    return _pixels_dtype(
        {
            "input_ids": np.array([ids], dtype=np.int32),
            "pixel_values": pixels,
            "image_grid_thw": grids,
            "num_image_tokens": np.array([2, 6], dtype=np.int64),
        },
        dtype,
    )


@pytest.mark.parametrize("device,dtype", CASES)
@pytest.mark.parametrize("family", ("phi3v", "qwen2_5_vl", "mistral3"))
def test_shared_gqa_eos_image_resume_and_suffix_rewind(gqa_model_factory, dtype, family):
    directory, model = gqa_model_factory(family, shared=True)
    generator, history = _generator(model), _History(family)
    generator.set_inputs(_named(_pixels_dtype(history.image(2), dtype)))
    for _ in range(3):
        generator.generate_next_token()
        history.text([int(generator.get_sequence(0)[-1])])
    forced = np.full_like(generator.get_logits(), -10000)
    forced[..., EOS_TOKEN_ID] = 10000
    generator.set_logits(forced)
    generator.generate_next_token()
    assert generator.is_done()
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    cache = {kind: generator.get_output(f"present.0.{kind}").copy() for kind in ("key", "value")}
    np.testing.assert_array_equal(generator.get_logits(), forced)
    for kind, expected in cache.items():
        np.testing.assert_array_equal(generator.get_output(f"present.0.{kind}"), expected)

    generator.set_inputs(_named(_pixels_dtype(history.image(9), dtype)))
    boundary = len(history.tokens)
    assert not generator.is_done()
    np.testing.assert_allclose(_last_logits(generator), _numpy_last_logits(directory, history), **_tolerance(dtype))
    for _ in range(4):
        generator.generate_next_token()
        history.text([int(generator.get_sequence(0)[-1])])
    expected = _last_logits(generator)
    generator.rewind_to(boundary + 2)
    generator.append_tokens(np.asarray([history.tokens[-2:]], dtype=np.int32))
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    np.testing.assert_allclose(_last_logits(generator), expected, **_tolerance(dtype))
    np.testing.assert_allclose(_last_logits(generator), _numpy_last_logits(directory, history), **_tolerance(dtype))


@pytest.mark.parametrize("device,dtype", CASES)
@pytest.mark.parametrize("family", ("phi3v", "qwen2_5_vl", "mistral3"))
@pytest.mark.parametrize("shared", (False, True), ids=("dynamic", "shared"))
def test_gqa_consecutive_suffix_rewinds_match_direct_rewind(gqa_model_factory, dtype, family, shared):
    directory, model = gqa_model_factory(family, shared=shared)
    generator, reference = _generator(model), _generator(model)
    history = _History(family)
    inputs = _named(_pixels_dtype(history.image(2), dtype))
    boundary = len(history.tokens)
    for current in (generator, reference):
        current.set_inputs(inputs)
        current.append_tokens(np.asarray([[4, 5, 6, 7, 8, 9, 10]], dtype=np.int32))

    generator.rewind_to(boundary + 5)
    generator.rewind_to(boundary + 3)
    reference.rewind_to(boundary + 3)
    history.text([4, 5, 6, 11, 12, 13])
    for current in (generator, reference):
        current.append_tokens(np.asarray([[11, 12, 13]], dtype=np.int32))
        np.testing.assert_array_equal(current.get_sequence(0), history.tokens)
    for kind in ("key", "value"):
        actual = generator.get_output(f"present.0.{kind}")
        expected = reference.get_output(f"present.0.{kind}")
        np.testing.assert_allclose(
            actual[:, :, : len(history.tokens)], expected[:, :, : len(history.tokens)], **_tolerance(dtype)
        )
    np.testing.assert_allclose(_last_logits(generator), _last_logits(reference), **_tolerance(dtype))
    np.testing.assert_allclose(_last_logits(generator), _numpy_last_logits(directory, history), **_tolerance(dtype))

    generator.set_inputs(_named(_pixels_dtype(history.image(9), dtype)))
    np.testing.assert_allclose(_last_logits(generator), _numpy_last_logits(directory, history), **_tolerance(dtype))


@pytest.mark.parametrize("device,dtype", CASES)
@pytest.mark.parametrize("family", ("phi3v", "qwen2_5_vl", "mistral3"))
@pytest.mark.parametrize("shared", (False, True), ids=("dynamic", "shared"))
def test_gqa_rewind_discards_unforwarded_response_token(gqa_model_factory, dtype, family, shared):
    directory, model = gqa_model_factory(family, shared=shared)
    generator, history = _generator(model), _History(family)
    generator.set_inputs(_named(_pixels_dtype(history.image(2), dtype)))
    history.text([4, 5, 6])
    generator.append_tokens(np.asarray([[4, 5, 6]], dtype=np.int32))
    cached_length = len(history.tokens)
    prefix = {kind: generator.get_output(f"present.0.{kind}")[:, :, :cached_length].copy() for kind in ("key", "value")}
    generator.generate_next_token()
    assert len(generator.get_sequence(0)) == cached_length + 1
    # Reading logits here would forward the sampled token and hide the equality case.
    generator.rewind_to(cached_length)
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    history.text([7, 8, 9])
    generator.append_tokens(np.asarray([[7, 8, 9]], dtype=np.int32))
    for kind, expected in prefix.items():
        np.testing.assert_allclose(
            generator.get_output(f"present.0.{kind}")[:, :, :cached_length], expected, **_tolerance(dtype)
        )
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    np.testing.assert_allclose(_last_logits(generator), _numpy_last_logits(directory, history), **_tolerance(dtype))

    generator.set_inputs(_named(_pixels_dtype(history.image(9), dtype)))
    np.testing.assert_allclose(_last_logits(generator), _numpy_last_logits(directory, history), **_tolerance(dtype))


@pytest.mark.parametrize("device,dtype", CASES)
@pytest.mark.parametrize("shared", (False, True), ids=("dynamic", "shared"))
@pytest.mark.parametrize("family", ("qwen2_5_vl", "mistral3"))
def test_gqa_no_readback_unequal_grid_lifetime_stress(gqa_model_factory, dtype, shared, family):
    _, model = gqa_model_factory(family, shared=shared, asynchronous=True)

    def turn_arrays(turn):
        if turn % 2 == 0:
            return _unequal_grid_turn(turn + 1, dtype, family)
        return _pixels_dtype(_History(family).image(turn + 1), dtype)

    generator = _generator(model)
    for turn in range(8):
        arrays = turn_arrays(turn)
        inputs = _named(arrays)
        generator.set_inputs(inputs)
        del inputs, arrays
        gc.collect()
        # Separate generators churn the same model/device allocator. No sequence,
        # logits, cache, or profiling readback occurs in this workload's turn loop.
        churn = [_generator(model) for _ in range(3)]
        for other in churn:
            other.append_tokens(np.array([[4, 6, 8]], dtype=np.int32))
            other.generate_next_token()
        del other, churn
        for _ in range(4):
            generator.generate_next_token()
    sequence = generator.get_sequence(0).copy()
    actual = _last_logits(generator)
    # Replay only in the test oracle, with the now-known generated tokens.
    reference = _generator(model, past_present_share_buffer=not shared)
    offset = 0
    for turn in range(8):
        arrays = turn_arrays(turn)
        reference.set_inputs(_named(arrays))
        offset += arrays["input_ids"].size
        reference.append_tokens(sequence[offset : offset + 4].reshape(1, -1))
        offset += 4
    np.testing.assert_array_equal(reference.get_sequence(0), sequence)
    np.testing.assert_allclose(actual, _last_logits(reference), **_tolerance(dtype))


@pytest.mark.parametrize("device,dtype", CASES)
def test_gqa_actual_capture_rejects_image_atomically_and_resumes(gqa_model_factory, device, dtype):
    if device not in GPU_DEVICES:
        pytest.skip("CPU has no actual GPU graph capture; not capture execution coverage")
    _, model = gqa_model_factory(shared=True, capture=True)
    _, eager_model = gqa_model_factory(shared=True)
    generator, reference = _generator(model), _generator(eager_model)
    ids, pixels = _image_turn("phi3v", 1)
    for current in (generator, reference):
        current.set_inputs(_named(_pixels_dtype(_arrays("phi3v", ids, [pixels]), dtype)))
        # First sample uses prefill; following steps capture and then replay decode.
        for _ in range(8):
            current.generate_next_token()
    np.testing.assert_array_equal(generator.get_sequence(0), reference.get_sequence(0))
    before = {
        "sequence": generator.get_sequence(0).copy(),
        "logits": _last_logits(generator),
        **{kind: generator.get_output(f"present.0.{kind}").copy() for kind in ("key", "value")},
    }
    ids, pixels = _image_turn("phi3v", 7)
    with pytest.raises(RuntimeError, match="graph capture"):
        generator.set_inputs(_named(_pixels_dtype(_arrays("phi3v", ids, [pixels]), dtype)))
    np.testing.assert_array_equal(generator.get_sequence(0), before["sequence"])
    np.testing.assert_array_equal(_last_logits(generator), before["logits"])
    for kind in ("key", "value"):
        np.testing.assert_array_equal(generator.get_output(f"present.0.{kind}"), before[kind])
    for _ in range(4):
        generator.generate_next_token()
        reference.generate_next_token()
    np.testing.assert_array_equal(generator.get_sequence(0), reference.get_sequence(0))
    np.testing.assert_allclose(_last_logits(generator), _last_logits(reference), **_tolerance(dtype))

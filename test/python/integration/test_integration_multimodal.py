# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pinned Phi Vision retained-cache continuation, full-prefix oracles, and EP audits."""

from __future__ import annotations

import gc
import json
import struct
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest
from _test_utils import MULTIMODAL_EP_NAMES, require_execution_provider

from . import resolver
from .fetch_public_models import verify_artifact

pytestmark = pytest.mark.multimodal

_ROLES = ("vision", "embedding", "decoder")
# LongRoPE switches above 4096; replay would then rotate cached keys differently.
_MAX_LENGTH = 4096
# Fixed before running either EP: FP32 CPU and FP16 CUDA, both with INT4 weights.
_TOLERANCES = {"cpu": {"atol": 2e-3, "rtol": 2e-4}, "cuda": {"atol": 6e-2, "rtol": 2e-3}}
_METADATA_OPS = {
    "Gather",
    "Slice",
    "Concat",
    "Cast",
    "Unsqueeze",
    "Squeeze",
    "Reshape",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "ReduceProd",
    "Less",
    "Greater",
    "And",
    "NonZero",
    "Transpose",
    "ConstantOfShape",
    "Equal",
    "Where",
    "Expand",
    "SequenceAt",
}
_INTEGER_SEQUENCE_OPS = {"SequenceConstruct", "SplitToSequence"}
_INTEGER_TYPES = {"int32", "int64", "bool"}


def _audit_profile(path: Path, role: str, device: str, control_nodes=None) -> None:
    events = json.loads(path.read_text(encoding="utf-8"))
    nodes = [event for event in events if event.get("cat") == "Node" and event.get("args", {}).get("provider")]
    assert nodes, f"No actual partition evidence for {role}: {path}"
    expected = MULTIMODAL_EP_NAMES[device][1]
    numerical = set()
    for event in nodes:
        args = event["args"]
        op = args.get("op_name")
        if op in {"MemcpyFromHost", "MemcpyToHost"}:
            continue
        if args["provider"] == expected:
            numerical.add(op)
            continue
        # Allow CPU shape/index work, but not floating-point fallback (including Gather).
        inputs = args.get("input_type_shape", [])
        outputs = args.get("output_type_shape", [])
        metadata = op in {"Shape", "Size"} or (
            op in _METADATA_OPS
            and bool(inputs)
            and bool(outputs)
            and all(value and set(value) <= _INTEGER_TYPES for value in [*inputs, *outputs])
        )
        # ORT omits sequence outputs; these ops preserve their input element types.
        metadata |= (
            op in _INTEGER_SEQUENCE_OPS
            and bool(inputs)
            and all(value and set(value) <= _INTEGER_TYPES for value in inputs)
        )
        # Allow pinned host dispatch; its body kernels are audited individually.
        control = op in {"If", "Loop"} and (control_nodes or {}).get(event["name"].removesuffix("_kernel_time")) == op
        assert args["provider"] == "CPUExecutionProvider" and (metadata or control), (
            f"{role}: unapproved fallback node {event['name']}: {args}"
        )
    required_ops = {
        "vision": {"Conv", "MatMul", "FusedMatMul", "Attention", "MultiHeadAttention"},
        "embedding": {"Gather"},
        "decoder": {"GroupQueryAttention"},
    }
    assert numerical & required_ops[role], f"{role} did not execute its numerical workload on {expected}"


def _write_image(path: Path, height: int, width: int, index: int) -> None:
    y, x = np.indices((height, width))
    pixels = np.stack(
        ((x * (index + 1) + 50 * index) % 256, (y * 3 + 80 * index) % 256, ((x + y) * 2 + 30 * index) % 256),
        axis=-1,
    ).astype(np.uint8)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\0" + row.tobytes() for row in pixels)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


@dataclass
class _Bundle:
    model: og.Model
    tokenizer: og.Tokenizer
    processor: object
    device: str
    config: dict
    directory: Path
    images: list[Path]
    control_nodes: dict
    overrides: dict


def _control_nodes(graph):
    result = {}
    for node in graph.node:
        if node.op_type in {"If", "Loop"}:
            result[node.name] = node.op_type
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                result.update(_control_nodes(attribute.g))
    return result


def _session_overrides(device, directory, *, default_kernels=False):
    roles = {}
    for role in _ROLES:
        provider_options = {}
        if device in ("cuda", "webgpu"):
            provider_options["device_filtering_options"] = {"hardware_device_type": "gpu"}
        roles[role] = {
            "session_options": {
                "provider_options": [] if device == "cpu" else [{device: provider_options}],
                "enable_profiling": str(directory / role),
                "intra_op_num_threads": 8,
            }
        }
    # This export predates the explicit processor filename field.
    roles["vision"]["config_filename"] = "processor_config.json"
    if device == "cpu" and not default_kernels:
        # Packed INT8 is chunk-dependent; the oracle keeps INT4 weights with FP32 accumulation.
        roles["decoder"]["session_options"]["session.disable_prepacking"] = "1"
    return roles


@pytest.fixture(scope="module")
def bundles(pytestconfig, request):
    cached = {}
    root = Path(pytestconfig.getoption("--multimodal-output-dir")) / uuid.uuid4().hex
    root.mkdir(parents=True)
    failures = request.session.testsfailed
    yield cached, root
    for bundle in cached.values():
        directory, device = bundle.directory, bundle.device
        bundle.processor = bundle.tokenizer = bundle.model = None
        gc.collect()
        if request.session.testsfailed != failures:
            # Failure tracebacks may retain Generators and prevent profile finalization.
            continue
        for role in _ROLES:
            profiles = list(directory.glob(f"{role}_*.json"))
            assert len(profiles) == 1, f"Expected a finalized {role} profile under {directory}, found {profiles}"
            _audit_profile(profiles[0], role, device, bundle.control_nodes[role])


@pytest.fixture
def bundle(device, model, pytestconfig, bundles, request):
    require_execution_provider(device)
    model_path = resolver.get_path_for(model, device, model_root=pytestconfig.getoption("--model-root"), required=True)
    cached, root = bundles
    default_kernels = getattr(request, "param", False)
    unpacked_reference = device == "cpu" and not default_kernels
    key = model, device, unpacked_reference
    if key not in cached:
        verify_artifact(model_path, model, device)
        cfg = json.loads((model_path / "genai_config.json").read_text(encoding="utf-8"))
        assert cfg["model"]["type"] == "phi3v"
        assert cfg["search"]["past_present_share_buffer"] is True
        variant = "fp32-reference" if unpacked_reference else "default-kernels"
        directory = root / f"{model}-{device}-{variant}"
        directory.mkdir()
        control_nodes = {}
        for role in _ROLES:
            graph = onnx.load(model_path / cfg["model"][role]["filename"], load_external_data=False)
            if role == "decoder":
                assert sum(node.op_type == "GroupQueryAttention" for node in graph.graph.node) == 32
            control_nodes[role] = _control_nodes(graph.graph)
            del graph
        roles = _session_overrides(device, directory, default_kernels=default_kernels)
        config = og.Config(str(model_path))
        config.overlay(json.dumps({"model": roles}))
        runtime_model = og.Model(config)
        images = [directory / f"image-{i}.png" for i in range(2)]
        for i, (height, width) in enumerate(((224, 336), (336, 224))):
            _write_image(images[i], height, width, i + 1)
        cached[key] = _Bundle(
            runtime_model,
            og.Tokenizer(runtime_model),
            runtime_model.create_multimodal_processor(),
            device,
            cfg,
            directory,
            images,
            control_nodes,
            roles,
        )
    return cached[key]


def _generator(bundle, shared):
    params = og.GeneratorParams(bundle.model)
    params.set_search_options(max_length=_MAX_LENGTH, do_sample=False, past_present_share_buffer=shared)
    return og.Generator(bundle.model, params)


def _last_logits(generator):
    logits = generator.get_logits()
    return logits.reshape(-1, logits.shape[-1])[-1].copy()


@dataclass
class _History:
    tokens: list[int] = field(default_factory=list)
    reference_tokens: list[int] = field(default_factory=list)
    fragments: list[str] = field(default_factory=list)
    images: list[Path] = field(default_factory=list)

    def text(self, tokens, text):
        self.tokens.extend(tokens)
        self.reference_tokens.extend(tokens)
        self.fragments.append(text)


def _image_turn(bundle, generator, history, image_index):
    prompt = "<|user|>\n<|image_1|>\nDescribe the colors briefly.<|end|>\n<|assistant|>\n"
    image = bundle.images[image_index]
    images = og.Images.open(str(image))
    inputs = bundle.processor(prompt, images=images)
    tokens = inputs["input_ids"].as_numpy().reshape(-1).tolist()
    assert tokens.count(-1) > 0, "Phi processor must expand local image_1 into negative image tokens"
    history.images.append(image)
    history.tokens.extend(tokens)
    history.reference_tokens.extend(-len(history.images) if token < 0 else token for token in tokens)
    history.fragments.append(prompt.replace("<|image_1|>", f"<|image_{len(history.images)}|>"))
    generator.set_inputs(inputs)
    del inputs, images
    gc.collect()
    pressure = [np.full((512, 512), value, dtype=np.float32) for value in range(8)]
    del pressure


def _text_turn(bundle, generator, history, text="Name another color."):
    prompt = f"<|user|>\n{text}<|end|>\n<|assistant|>\n"
    tokens = bundle.tokenizer.encode(prompt)
    assert len(tokens) > 1
    generator.append_tokens(np.asarray(tokens, dtype=np.int32))
    history.text(tokens, prompt)


def _respond(bundle, generator, history, count=3):
    start = len(history.tokens)
    for _ in range(count):
        if generator.is_done():
            break
        before = len(history.tokens)
        generator.generate_next_token()
        sequence = generator.get_sequence(0).tolist()
        if len(sequence) == before:
            assert generator.is_done()
            break
        assert len(sequence) == before + 1
        token = sequence[-1]
        history.text([token], bundle.tokenizer.decode([token]))
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    return history.tokens[start:]


def _reference(bundle, history, shared):
    reference = _generator(bundle, shared)
    if history.images:
        images = og.Images.open(*(str(path) for path in history.images))
        inputs = bundle.processor("".join(history.fragments), images=images)
        # Decode/encode can merge adjacent text tokens. Use the exact
        # teacher-forced token history with full-prefix image numbering.
        tokens = np.asarray([history.reference_tokens], dtype=np.int32)
        inputs["input_ids"] = og.Tensor(tokens)
        reference.set_inputs(inputs)
        del inputs, images, tokens
    else:
        reference.append_tokens(np.asarray(history.reference_tokens, dtype=np.int32))
    return reference


def _assert_reference(bundle, generator, history, shared, *, response_steps=2):
    # Terminal scores may contain caller overrides or sampling processors.
    assert not generator.is_done(), "Compare a committed prefix before sampling EOS"
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    reference = _reference(bundle, history, shared)
    for step in range(response_steps + 1):
        np.testing.assert_allclose(
            _last_logits(generator),
            _last_logits(reference),
            **_TOLERANCES[bundle.device],
            err_msg=f"{bundle.device}: retained image/KV history differs from same-EP full-prefix replay at step {step}",
        )
        if step == response_steps:
            break
        response = _respond(bundle, generator, history, count=1)
        if not response:
            break
        reference.append_tokens(np.asarray(response, dtype=np.int32))
    del reference


@pytest.mark.parametrize("shared", [False, True], ids=["dynamic-kv", "shared-gqa-kv"])
def test_image_response_image_matches_full_prefix(bundle, shared):
    generator, history = _generator(bundle, shared), _History()
    _image_turn(bundle, generator, history, 0)
    _respond(bundle, generator, history)
    _image_turn(bundle, generator, history, 1)
    _assert_reference(bundle, generator, history, shared)


@pytest.mark.parametrize("initial_text", [False, True], ids=["image-text-image", "text-image-text"])
def test_mixed_multitoken_text_and_images(bundle, initial_text):
    generator, history = _generator(bundle, True), _History()
    if initial_text:
        _text_turn(bundle, generator, history)
        _respond(bundle, generator, history)
    _image_turn(bundle, generator, history, 0)
    _respond(bundle, generator, history)
    _text_turn(bundle, generator, history)
    if not initial_text:
        _respond(bundle, generator, history)
        _image_turn(bundle, generator, history, 1)
    _assert_reference(bundle, generator, history, True)


def test_eos_image_resume_and_latest_suffix_rewind(bundle):
    generator, history = _generator(bundle, True), _History()
    _image_turn(bundle, generator, history, 0)
    _respond(bundle, generator, history)
    committed = generator.get_sequence(0).copy()
    if not generator.is_done():
        eos = bundle.config["model"]["eos_token_id"]
        forced = np.full_like(generator.get_logits(), -10000)
        forced[..., eos] = 10000
        generator.set_logits(forced)
        generator.generate_next_token()
    assert generator.is_done()
    np.testing.assert_array_equal(generator.get_sequence(0), committed)
    terminal_logits = generator.get_logits().copy()
    np.testing.assert_array_equal(generator.get_logits(), terminal_logits)
    np.testing.assert_array_equal(generator.get_sequence(0), committed)
    _image_turn(bundle, generator, history, 1)
    boundary = len(history.tokens)
    assert not generator.is_done()
    _assert_reference(bundle, generator, history, True, response_steps=4)
    # A known suffix makes rewind coverage independent of when the EP samples EOS.
    _text_turn(bundle, generator, history, text="Compare the two images.")
    sequence, logits = generator.get_sequence(0).copy(), _last_logits(generator)
    for invalid in (0, boundary - 1, boundary):
        with pytest.raises(RuntimeError, match="latest multimodal prompt boundary"):
            generator.rewind_to(invalid)
        np.testing.assert_array_equal(generator.get_sequence(0), sequence)
        np.testing.assert_array_equal(_last_logits(generator), logits)
    assert len(sequence) >= boundary + 3, "Need a text suffix to exercise valid rewind"
    generator.rewind_to(boundary + 1)
    generator.append_tokens(np.asarray(sequence[boundary + 1 :], dtype=np.int32))
    np.testing.assert_array_equal(generator.get_sequence(0), sequence)
    np.testing.assert_allclose(_last_logits(generator), logits, **_TOLERANCES[bundle.device])
    _assert_reference(bundle, generator, history, True)


def test_two_turn_lifetime_with_deferred_readback(bundle):
    generator, history = _generator(bundle, True), _History()
    # Avoid extra get_logits/get_sequence calls in the repeated-turn workload.
    # Fixed response tokens make an exact teacher-forced reference possible.
    response = bundle.tokenizer.encode("The colors are red and blue.")
    for index in range(2):
        _image_turn(bundle, generator, history, index)
        generator.append_tokens(np.asarray(response, dtype=np.int32))
        history.text(response, "The colors are red and blue.")
    _assert_reference(bundle, generator, history, True)


@pytest.mark.parametrize("bundle", [True], indirect=True, ids=["default-kernels"])
@pytest.mark.parametrize("shared", [False, True], ids=["dynamic-kv", "shared-gqa-kv"])
def test_default_kernel_retained_turns(bundle, shared):
    assert "session.disable_prepacking" not in bundle.overrides["decoder"]["session_options"]
    generator, history = _generator(bundle, shared), _History()
    _image_turn(bundle, generator, history, 0)
    first_response = _respond(bundle, generator, history)
    assert first_response
    _text_turn(bundle, generator, history)
    text_response = _respond(bundle, generator, history)
    assert text_response
    _image_turn(bundle, generator, history, 1)
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    continued_logits = _last_logits(generator)

    # Packed INT8 depends on batch width: replay identical chunks, not the full prefix.
    reference, replay = _generator(bundle, shared), _History()
    _image_turn(bundle, reference, replay, 0)
    for token in first_response:
        reference.append_tokens(np.asarray([token], dtype=np.int32))
        replay.text([token], bundle.tokenizer.decode([token]))
    _text_turn(bundle, reference, replay)
    for token in text_response:
        reference.append_tokens(np.asarray([token], dtype=np.int32))
        replay.text([token], bundle.tokenizer.decode([token]))
    _image_turn(bundle, reference, replay, 1)
    np.testing.assert_array_equal(reference.get_sequence(0), history.tokens)
    np.testing.assert_allclose(continued_logits, _last_logits(reference), **_TOLERANCES[bundle.device])

    fresh = _generator(bundle, shared)
    _image_turn(bundle, fresh, _History(), 1)
    assert not np.allclose(continued_logits, _last_logits(fresh), **_TOLERANCES[bundle.device]), (
        "Default-kernel continuation lost observable prior-context influence"
    )
    del fresh

    start = len(history.tokens)
    for _ in range(2):
        response = _respond(bundle, generator, history, count=1)
        if not response:
            break
        reference.append_tokens(np.asarray(response, dtype=np.int32))
        np.testing.assert_array_equal(reference.get_sequence(0), history.tokens)
        np.testing.assert_allclose(_last_logits(generator), _last_logits(reference), **_TOLERANCES[bundle.device])
    assert len(history.tokens) > start, "The second image produced no response"

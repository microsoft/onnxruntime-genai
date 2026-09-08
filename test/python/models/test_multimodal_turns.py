# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

"""EP-parametrized coverage of retained image, token, KV, and mRoPE history.

No tokenizer, downloaded weights, or tracked ONNX assets are needed. The fixture
graphs and independent arithmetic oracle intentionally make dropped/replayed
tokens, stale images, cache resets, and wrong positions observable in logits.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field

import numpy as np
import onnxruntime_genai as og
import pytest
from _test_utils import MULTIMODAL_EP_NAMES, multimodal_test_devices, require_execution_provider
from create.create_multimodal_turn_test_model import (
    EOS_TOKEN_ID,
    IMAGE_TOKEN_ID,
    QWEN_FAMILIES,
    VISION_END_TOKEN_ID,
    VISION_START_TOKEN_ID,
    VOCAB_SIZE,
    create_model,
    make_decoder_model,
    make_embedding_model,
    make_vision_model,
)
from onnx.reference import ReferenceEvaluator

FAMILIES = ("phi3v", "mistral3", "qwen2_5_vl")


def _image_turn(family, value):
    patches = 4 if family in QWEN_FAMILIES else 1
    pixels = np.arange(patches * 3, dtype=np.float32).reshape(patches, 3) + value
    if family in QWEN_FAMILIES:
        ids = [2, VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * 4), VISION_END_TOKEN_ID, 3]
    else:
        ids = [2, -1 if family == "phi3v" else IMAGE_TOKEN_ID, 3]
    return ids, pixels


def _arrays(family, ids, images):
    arrays = {"input_ids": np.array([ids], dtype=np.int32)}
    if not images:
        return arrays
    pixels = np.concatenate(images, axis=0)
    if family in QWEN_FAMILIES:
        arrays["pixel_values"] = pixels
        arrays["image_grid_thw"] = np.tile(np.array([[1, 2, 2]], dtype=np.int64), (len(images), 1))
    else:
        shape = (len(images), 3, 1, 1) if family == "mistral3" else (len(images), 1, 3, 1, 1)
        arrays["pixel_values"] = pixels.reshape(shape)
        arrays["image_sizes"] = np.ones((len(images), 2), dtype=np.int64)
    arrays["num_image_tokens"] = np.array([len(image) for image in images], dtype=np.int64)
    return arrays


def _named(arrays):
    inputs = og.NamedTensors()
    for name, array in arrays.items():
        inputs[name] = og.Tensor(array)
    return inputs


def _generator(model, **options):
    params = og.GeneratorParams(model)
    params.set_search_options(**{"do_sample": False, "max_length": 192, **options})
    return og.Generator(model, params)


def _last_logits(generator):
    logits = np.asarray(generator.get_logits())
    assert logits.shape[-1] == VOCAB_SIZE
    return logits.reshape(-1, VOCAB_SIZE)[-1].copy()


@dataclass
class _History:
    family: str
    turns: list = field(default_factory=list)

    def text(self, ids):
        self.turns.append((list(ids), None))

    def image(self, value):
        ids, pixels = _image_turn(self.family, value)
        self.turns.append((ids, pixels))
        return _arrays(self.family, ids, [pixels])

    @property
    def tokens(self):
        return [token for ids, _ in self.turns for token in ids]

    def prefix_arrays(self):
        ids, images = [], []
        for turn_ids, pixels in self.turns:
            prefix_ids = turn_ids
            if pixels is not None:
                images.append(pixels)
                if self.family == "phi3v":
                    prefix_ids = [-len(images) if token < 0 else token for token in turn_ids]
            ids.extend(prefix_ids)
        return _arrays(self.family, ids, images)

    def embeddings_and_positions(self):
        embeddings, positions = [], []
        next_position = 0
        for ids, pixels in self.turns:
            values = [2 * token + 3 for token in ids]
            local_positions = [[i, i, i] for i in range(len(ids))]
            if pixels is not None:
                features = pixels.sum(axis=1) + 11
                if self.family in QWEN_FAMILIES:
                    values[2:6] = features.tolist()
                    # A 2x2 image uses four physical tokens, but just two mRoPE positions.
                    local_positions = [
                        [0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2],
                        [2, 2, 3],
                        [2, 3, 2],
                        [2, 3, 3],
                        [4, 4, 4],
                        [5, 5, 5],
                    ]
                else:
                    values[1] = features[0]
            local_positions = np.asarray(local_positions, dtype=np.int64)
            positions.extend((local_positions + next_position).tolist())
            embeddings.extend(values)
            next_position += int(local_positions.max()) + 1
        return np.asarray(embeddings, dtype=np.float32), np.asarray(positions, dtype=np.int64)

    def logits(self):
        embeddings, positions = self.embeddings_and_positions()
        weighted_positions = positions @ np.array([1, 2, 4])
        history = np.sum(3 * embeddings + 4 * weighted_positions + 1)
        target = int(history) % 16 + 2
        return (-((np.arange(VOCAB_SIZE) - target) ** 2) + history / 1024).astype(np.float32)


@pytest.fixture(params=multimodal_test_devices())
def turn_device(request):
    require_execution_provider(request.param)
    return request.param


@pytest.fixture
def model_factory(tmp_path, turn_device):
    models = {}

    def make(family, *, fail_on_negative_pixels=False, asynchronous=False):
        key = (family, fail_on_negative_pixels, asynchronous)
        if key not in models:
            suffix = "-failing-vision" if fail_on_negative_pixels else "-async" if asynchronous else ""
            directory = tmp_path / (family + suffix)
            create_model(directory, family, fail_on_negative_pixels=fail_on_negative_pixels)
            config_path = directory / "genai_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for name in ("decoder", "embedding", "vision"):
                if name not in config["model"]:
                    continue
                provider_options = {}
                if turn_device in ("cuda", "webgpu"):
                    provider_options["device_filtering_options"] = {"hardware_device_type": "gpu"}
                config["model"][name]["session_options"] = {
                    "provider_options": [] if turn_device == "cpu" else [{turn_device: provider_options}],
                    "session.disable_cpu_ep_fallback": "0" if turn_device == "cpu" else "1",
                }
                if asynchronous:
                    config["model"][name]["run_options"] = {"disable_synchronize_execution_providers": "1"}
            config_path.write_text(json.dumps(config), encoding="utf-8")
            models[key] = og.Model(str(directory))
            assert models[key].device_type == MULTIMODAL_EP_NAMES[turn_device][0]
        return models[key]

    return make


def _assert_prefill(generator, history, model):
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    np.testing.assert_array_equal(_last_logits(generator), history.logits())
    reference = _generator(model)
    if history.family == "llama":
        reference.append_tokens(np.array([history.tokens], dtype=np.int32))
    else:
        reference.set_inputs(_named(history.prefix_arrays()))
    np.testing.assert_array_equal(_last_logits(generator), _last_logits(reference))


def _respond(generator, history, count=5):
    for _ in range(count):
        expected_logits = history.logits()
        assert not generator.is_done()
        np.testing.assert_array_equal(_last_logits(generator), expected_logits)
        generator.generate_next_token()
        token = int(generator.get_sequence(0)[-1])
        assert token == int(np.argmax(expected_logits))
        history.text([token])
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)


@pytest.mark.parametrize("family", (*FAMILIES, "qwen3_vl", "fara"))
def test_three_image_response_turns_match_full_prefix(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    for value in (1, 7, 13):
        inputs = _named(history.image(value))
        generator.set_inputs(inputs)
        del inputs
        gc.collect()
        _assert_prefill(generator, history, model)
        _respond(generator, history, count=12)
    if family in QWEN_FAMILIES:
        _, positions = history.embeddings_and_positions()
        assert int(positions.max()) + 1 - len(history.tokens) == -6


def test_qwen_later_images_with_different_grids_match_full_prefix(model_factory):
    family = "qwen2_5_vl"
    model = model_factory(family)
    generator = _generator(model)
    tokens, images, grids = [], [], []
    for h, w in ((2, 2), (1, 3), (3, 1)):
        pixels = np.arange(h * w * 3, dtype=np.float32).reshape(-1, 3) + len(images) * 7
        ids = [2, VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * (h * w)), VISION_END_TOKEN_ID, 3]
        grid = [1, h, w]
        arrays = _arrays(family, ids, [pixels])
        arrays["image_grid_thw"] = np.array([grid], dtype=np.int64)
        generator.set_inputs(_named(arrays))

        tokens.extend(ids)
        images.append(pixels)
        grids.append(grid)
        full_prefix = _arrays(family, tokens, images)
        full_prefix["image_grid_thw"] = np.array(grids, dtype=np.int64)
        reference = _generator(model)
        reference.set_inputs(_named(full_prefix))
        np.testing.assert_array_equal(_last_logits(generator), _last_logits(reference))
        for _ in range(3):
            generator.generate_next_token()
            reference.generate_next_token()
            np.testing.assert_array_equal(generator.get_sequence(0), reference.get_sequence(0))
        tokens = generator.get_sequence(0).tolist()


@pytest.mark.parametrize("family", QWEN_FAMILIES)
def test_unequal_grid_images_within_later_turns(model_factory, family):
    model = model_factory(family)
    generator = _generator(model)
    tokens, images, grids = [], [], []
    for turn_grids in (((2, 2),), ((1, 3), (3, 1)), ((2, 1), (1, 2))):
        turn_ids, turn_images, grid_rows = [], [], []
        for h, w in turn_grids:
            pixels = np.arange(h * w * 3, dtype=np.float32).reshape(-1, 3) + 7 * (len(images) + 1)
            turn_ids.extend([2, VISION_START_TOKEN_ID, *([IMAGE_TOKEN_ID] * (h * w)), VISION_END_TOKEN_ID, 3])
            turn_images.append(pixels)
            grid_rows.append([1, h, w])
        arrays = _arrays(family, turn_ids, turn_images)
        arrays["image_grid_thw"] = np.asarray(grid_rows, dtype=np.int64)
        inputs = _named(arrays)
        generator.set_inputs(inputs)
        del inputs, arrays
        gc.collect()
        tokens.extend(turn_ids)
        images.extend(turn_images)
        grids.extend(grid_rows)
        reference_arrays = _arrays(family, tokens, images)
        reference_arrays["image_grid_thw"] = np.asarray(grids, dtype=np.int64)
        reference = _generator(model)
        reference.set_inputs(_named(reference_arrays))
        np.testing.assert_array_equal(_last_logits(generator), _last_logits(reference))
        for _ in range(4):
            generator.generate_next_token()
            reference.generate_next_token()
        np.testing.assert_array_equal(generator.get_sequence(0), reference.get_sequence(0))
        tokens = generator.get_sequence(0).tolist()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("asynchronous", [False, True], ids=["default-run", "async-run"])
def test_turn_lifetimes_without_intermediate_readbacks(model_factory, family, turn_device, asynchronous):
    if asynchronous and turn_device not in ("cuda", "nvtensorrtrtx"):
        pytest.skip("Unsynchronized CUDA-stream execution requires CUDA or TensorRT-RTX")
    model = model_factory(family, asynchronous=asynchronous)
    generator, history = _generator(model), _History(family)
    for turn in range(10):
        inputs = _named(history.image(turn + 1))
        generator.set_inputs(inputs)
        del inputs
        gc.collect()
        for _ in range(3):
            # Only the independent CPU oracle is inspected; the live Generator is not read back.
            expected_token = int(np.argmax(history.logits()))
            generator.generate_next_token()
            history.text([expected_token])
        text = [4, 5, 6] if turn % 2 else [7, 8]
        generator.append_tokens(np.array([text], dtype=np.int32))
        history.text(text)
        # Exercise allocation/reuse while the conversation's cache remains live.
        scratch = [_named(_arrays(family, *_scratch_turn(family, turn + i))) for i in range(4)]
        del scratch
    _assert_prefill(generator, history, model)


def _scratch_turn(family, value):
    ids, pixels = _image_turn(family, value)
    return ids, [pixels]


@pytest.mark.parametrize("family", FAMILIES)
def test_image_response_text_response_image_response(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    generator.set_inputs(_named(history.image(2)))
    _respond(generator, history)
    for text in ([4, 5, 6], [7, 8]):
        history.text(text)
        generator.append_tokens(np.array([text], dtype=np.int32))
        _assert_prefill(generator, history, model)
    _respond(generator, history)
    generator.set_inputs(_named(history.image(9)))
    _assert_prefill(generator, history, model)
    _respond(generator, history)


@pytest.mark.parametrize("family", FAMILIES)
def test_initial_text_response_then_image(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    history.text([2, 4, 6])
    generator.append_tokens(np.array([history.tokens], dtype=np.int32))
    _assert_prefill(generator, history, model)
    _respond(generator, history)
    generator.set_inputs(_named(history.image(5)))
    _assert_prefill(generator, history, model)
    _respond(generator, history)


@pytest.mark.parametrize("family", FAMILIES)
def test_initial_image_inputs_can_be_staged_before_append_tokens(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    arrays = history.image(2)
    input_ids = arrays.pop("input_ids")
    inputs = _named(arrays)
    generator.set_inputs(inputs)
    del inputs, arrays
    gc.collect()
    assert len(generator.get_sequence(0)) == 0

    generator.append_tokens(input_ids)
    _assert_prefill(generator, history, model)
    _respond(generator, history)
    generator.set_inputs(_named(history.image(8)))
    _assert_prefill(generator, history, model)
    _respond(generator, history)


@pytest.mark.parametrize("family", (*FAMILIES, "llama"))
def test_text_only_multitoken_turns_preserve_history(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    for text in ([2, 3, 4], [5, 6, 7, 8], [9, 10]):
        history.text(text)
        generator.append_tokens(np.array([text], dtype=np.int32))
        _assert_prefill(generator, history, model)
        _respond(generator, history)


@pytest.mark.parametrize("family", FAMILIES)
def test_input_ids_only_set_inputs_after_image(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    generator.set_inputs(_named(history.image(3)))
    _respond(generator, history)
    history.text([4, 5, 6])
    generator.set_inputs(_named({"input_ids": np.array([[4, 5, 6]], dtype=np.int32)}))
    _assert_prefill(generator, history, model)
    generator.set_inputs(_named(history.image(10)))
    _assert_prefill(generator, history, model)


def test_mistral_text_processor_metadata_does_not_restart_image_prefill(model_factory):
    model = model_factory("mistral3")
    generator, history = _generator(model), _History("mistral3")
    generator.set_inputs(_named(history.image(3)))
    _respond(generator, history)
    history.text([4, 5, 6])
    generator.set_inputs(
        _named(
            {
                "input_ids": np.array([[4, 5, 6]], dtype=np.int32),
                "num_image_tokens": np.array([0], dtype=np.int64),
            }
        )
    )
    _assert_prefill(generator, history, model)
    _respond(generator, history)
    generator.set_inputs(_named(history.image(10)))
    _assert_prefill(generator, history, model)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("resume_with_image", [False, True], ids=["text", "image"])
def test_eos_resume_preserves_committed_history(model_factory, family, resume_with_image):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    generator.set_inputs(_named(history.image(2)))
    _respond(generator, history)
    forced_logits = np.full_like(generator.get_logits(), -10000)
    forced_logits[..., EOS_TOKEN_ID] = 10000
    generator.set_logits(forced_logits)
    generator.generate_next_token()
    assert int(np.asarray(generator.get_next_tokens()).reshape(-1)[0]) == EOS_TOKEN_ID
    assert generator.is_done()
    # Search reports EOS without appending it to the committed sequence/KV prefix.
    np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    if resume_with_image:
        generator.set_inputs(_named(history.image(8)))
    else:
        history.text([5, 6, 7])
        generator.append_tokens(np.array([[5, 6, 7]], dtype=np.int32))
    assert not generator.is_done()
    _assert_prefill(generator, history, model)
    _respond(generator, history)


@pytest.mark.parametrize("family", (*FAMILIES, "llama"))
def test_reading_logits_after_eos_does_not_execute_uncommitted_token(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    history.text([2, 4, 6])
    generator.append_tokens(np.asarray([history.tokens], dtype=np.int32))
    _respond(generator, history, count=2)
    forced = np.full_like(generator.get_logits(), -10000)
    forced[..., EOS_TOKEN_ID] = 10000
    generator.set_logits(forced)
    generator.generate_next_token()
    assert generator.is_done()
    last_output = generator.get_output("logits").copy()
    for _ in range(2):
        np.testing.assert_array_equal(generator.get_logits(), forced)
        np.testing.assert_array_equal(generator.get_output("logits"), last_output)
        np.testing.assert_array_equal(generator.get_sequence(0), history.tokens)
    if family == "llama":
        history.text([7, 8, 9])
        generator.append_tokens(np.asarray([[7, 8, 9]], dtype=np.int32))
    else:
        generator.set_inputs(_named(history.image(9)))
    _assert_prefill(generator, history, model)
    _respond(generator, history)


@pytest.mark.parametrize("family", ("phi3v", "llama"))
def test_reading_logits_at_length_limit_executes_committed_token(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model, max_length=7), _History(family)
    history.text([2, 4, 6])
    generator.append_tokens(np.asarray([history.tokens], dtype=np.int32))
    _respond(generator, history, count=4)
    assert generator.is_done()
    np.testing.assert_array_equal(_last_logits(generator), history.logits())


@pytest.mark.parametrize("family", FAMILIES)
def test_earlier_pixels_remain_observable_after_later_image(model_factory, family):
    model = model_factory(family)
    results = []
    for first_pixel in (1, 5):
        generator, history = _generator(model), _History(family)
        generator.set_inputs(_named(history.image(first_pixel)))
        # Identical intervening tokens isolate image history, rather than sampled-token differences.
        history.text([4, 5, 6])
        generator.append_tokens(np.array([[4, 5, 6]], dtype=np.int32))
        generator.set_inputs(_named(history.image(12)))
        _assert_prefill(generator, history, model)
        results.append(_last_logits(generator))
    assert not np.array_equal(*results)


@pytest.mark.parametrize("family", ("phi3v", "mistral3"))
def test_chunked_later_image_prefill_matches_full_prefix(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model, chunk_size=2), _History(family)
    for value in (2, 8, 14):
        generator.set_inputs(_named(history.image(value)))
        _assert_prefill(generator, history, model)
        _respond(generator, history)


@pytest.mark.parametrize("family", FAMILIES)
def test_later_image_execution_failure_permanently_blocks_generator(model_factory, family, turn_device):
    if turn_device != "cpu":
        pytest.skip("Out-of-range Gather is a CPU error injector, not a portable GPU kernel failure")
    model = model_factory(family, fail_on_negative_pixels=True)
    generator, history = _generator(model), _History(family)
    generator.set_inputs(_named(history.image(2)))
    _assert_prefill(generator, history, model)
    _respond(generator, history)
    ids, pixels = _image_turn(family, 8)
    valid_arrays = _arrays(family, ids, [pixels])
    failing_arrays = dict(valid_arrays)
    failing_arrays["pixel_values"] = np.full_like(valid_arrays["pixel_values"], -1)
    with pytest.raises(RuntimeError, match="Gather|indices"):
        generator.set_inputs(_named(failing_arrays))

    operations = {
        "run": generator.generate_next_token,
        "append": lambda: generator.append_tokens(np.array([[4, 5]], dtype=np.int32)),
        "set_inputs": lambda: generator.set_inputs(_named(valid_arrays)),
        "rewind": lambda: generator.rewind_to(len(history.tokens) - 1),
        "logits": generator.get_logits,
        "output": lambda: generator.get_output("logits"),
    }
    for operation in operations.values():
        with pytest.raises(RuntimeError, match="Multimodal execution failed; this Generator cannot be reused"):
            operation()

    # Poisoning one generator must not corrupt the shared model sessions.
    fresh, fresh_history = _generator(model), _History(family)
    fresh.set_inputs(_named(fresh_history.image(8)))
    _assert_prefill(fresh, fresh_history, model)
    _respond(fresh, fresh_history)


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_ids", "input_ids"),
        ("wrong_ids_dtype", "input_ids"),
        ("empty_ids", "input_ids"),
        ("missing_pixels", "pixel_values"),
        ("empty_pixels", "pixel_values"),
        ("wrong_pixels_dtype", "pixel_values"),
        ("wrong_pixels_rank", "pixel_values"),
        ("missing_sizes", "image_sizes"),
        ("wrong_count_dtype", "num_image_tokens"),
        ("negative_count", "num_image_tokens"),
        ("zero_count", "num_image_tokens"),
        ("missing_count", "num_image_tokens"),
        ("wrong_count", "num_image_tokens"),
        ("missing_placeholder", "input_ids"),
        ("nonlocal_placeholder", "input_ids"),
    ],
)
def test_invalid_later_image_is_rejected_without_mutation(model_factory, case, match):
    model = model_factory("phi3v")
    generator, history = _generator(model), _History("phi3v")
    generator.set_inputs(_named(history.image(1)))
    _respond(generator, history)
    ids, pixels = _image_turn("phi3v", 9)
    arrays = _arrays("phi3v", ids, [pixels])
    if case == "missing_ids":
        arrays.pop("input_ids")
    elif case == "wrong_ids_dtype":
        arrays["input_ids"] = arrays["input_ids"].astype(np.int64)
    elif case == "empty_ids":
        arrays["input_ids"] = np.empty((1, 0), dtype=np.int32)
    elif case == "missing_pixels":
        arrays.pop("pixel_values")
    elif case == "empty_pixels":
        arrays["pixel_values"] = np.empty((0, 1, 3, 1, 1), dtype=np.float32)
    elif case == "wrong_pixels_dtype":
        arrays["pixel_values"] = arrays["pixel_values"].astype(np.int32)
    elif case == "wrong_pixels_rank":
        arrays["pixel_values"] = pixels
    elif case == "missing_sizes":
        arrays.pop("image_sizes")
    elif case == "wrong_count_dtype":
        arrays["num_image_tokens"] = np.array([1], dtype=np.int32)
    elif case in ("negative_count", "zero_count", "wrong_count"):
        arrays["num_image_tokens"] = np.array(
            [{"negative_count": -1, "zero_count": 0, "wrong_count": 2}[case]], dtype=np.int64
        )
    elif case == "missing_count":
        arrays.pop("num_image_tokens")
    elif case == "missing_placeholder":
        arrays["input_ids"] = np.array([[2, 4, 3]], dtype=np.int32)
    elif case == "nonlocal_placeholder":
        arrays["input_ids"] = np.array([[2, -2, 3]], dtype=np.int32)
    # GetLogits would consume the pending sampled token; inspect the last model output instead.
    sequence, logits = generator.get_sequence(0).copy(), generator.get_output("logits").copy()
    with pytest.raises(Exception, match=match):
        generator.set_inputs(_named(arrays))
    np.testing.assert_array_equal(generator.get_sequence(0), sequence)
    np.testing.assert_array_equal(generator.get_output("logits"), logits)
    generator.set_inputs(_named(history.image(9)))
    _assert_prefill(generator, history, model)
    _respond(generator, history)


@pytest.mark.parametrize(
    "case,match",
    [
        ("missing_grid", "image_grid_thw"),
        ("grid_dtype", "image_grid_thw"),
        ("grid_shape", "image_grid_thw"),
        ("grid_negative", "image_grid_thw"),
        ("grid_temporal", "image_grid_thw"),
        ("grid_token_count", "image_grid_thw|input_ids|num_image_tokens"),
        ("pixel_patch_count", "pixel_values|image_grid_thw"),
        ("missing_start", "input_ids"),
        ("broken_image_block", "input_ids"),
    ],
)
def test_invalid_qwen_image_metadata_preserves_previous_turn(model_factory, case, match):
    model = model_factory("qwen2_5_vl")
    generator, history = _generator(model), _History("qwen2_5_vl")
    generator.set_inputs(_named(history.image(1)))
    _respond(generator, history)
    ids, pixels = _image_turn("qwen2_5_vl", 9)
    arrays = _arrays("qwen2_5_vl", ids, [pixels])
    if case == "missing_grid":
        arrays.pop("image_grid_thw")
    elif case == "grid_dtype":
        arrays["image_grid_thw"] = arrays["image_grid_thw"].astype(np.float32)
    elif case == "grid_shape":
        arrays["image_grid_thw"] = np.array([[1, 2]], dtype=np.int64)
    elif case in ("grid_negative", "grid_temporal", "grid_token_count"):
        grids = {"grid_negative": [1, -2, 2], "grid_temporal": [2, 2, 2], "grid_token_count": [1, 2, 3]}
        arrays["image_grid_thw"] = np.array([grids[case]], dtype=np.int64)
    elif case == "pixel_patch_count":
        arrays["pixel_values"] = arrays["pixel_values"][:-1].copy()
    elif case == "missing_start":
        arrays["input_ids"][0, 1] = 4
    elif case == "broken_image_block":
        arrays["input_ids"][0, 3] = 4
    sequence, logits = generator.get_sequence(0).copy(), generator.get_output("logits").copy()
    with pytest.raises(Exception, match=match):
        generator.set_inputs(_named(arrays))
    np.testing.assert_array_equal(generator.get_sequence(0), sequence)
    np.testing.assert_array_equal(generator.get_output("logits"), logits)
    generator.set_inputs(_named(history.image(9)))
    _assert_prefill(generator, history, model)
    _respond(generator, history)


@pytest.mark.parametrize("case", ["missing_image_token", "wrong_image_size", "wrong_image_count"])
def test_invalid_mistral_image_metadata_preserves_previous_turn(model_factory, case):
    model = model_factory("mistral3")
    generator, history = _generator(model), _History("mistral3")
    generator.set_inputs(_named(history.image(1)))
    _respond(generator, history)
    ids, pixels = _image_turn("mistral3", 9)
    arrays = _arrays("mistral3", ids, [pixels])
    if case == "missing_image_token":
        arrays["input_ids"][0, 1] = 4
    elif case == "wrong_image_size":
        arrays["image_sizes"] = np.array([[2, 1]], dtype=np.int64)
    else:
        arrays["num_image_tokens"] = np.array([2], dtype=np.int64)
    sequence, logits = generator.get_sequence(0).copy(), generator.get_output("logits").copy()
    with pytest.raises(Exception, match="input_ids|image_sizes|num_image_tokens"):
        generator.set_inputs(_named(arrays))
    np.testing.assert_array_equal(generator.get_sequence(0), sequence)
    np.testing.assert_array_equal(generator.get_output("logits"), logits)
    generator.set_inputs(_named(history.image(9)))
    _assert_prefill(generator, history, model)


@pytest.mark.parametrize("family", FAMILIES)
def test_later_image_max_length_rejection_preserves_pending_token(model_factory, family):
    model = model_factory(family)
    history = _History(family)
    arrays = history.image(2)
    generator = _generator(model, max_length=len(history.tokens) + 4)
    generator.set_inputs(_named(arrays))
    _respond(generator, history, count=2)
    ids, pixels = _image_turn(family, 7)
    sequence, logits = generator.get_sequence(0).copy(), generator.get_output("logits").copy()
    with pytest.raises(Exception, match="max length"):
        generator.set_inputs(_named(_arrays(family, ids, [pixels])))
    np.testing.assert_array_equal(generator.get_sequence(0), sequence)
    np.testing.assert_array_equal(generator.get_output("logits"), logits)
    _respond(generator, history, count=1)


def test_unsupported_later_image_is_rejected_without_mutation(model_factory):
    family = "gemma3"
    model = model_factory(family)
    generator = _generator(model)
    ids, pixels = _image_turn(family, 2)
    generator.set_inputs(_named(_arrays(family, ids, [pixels])))
    sequence, logits = generator.get_sequence(0).copy(), generator.get_logits().copy()
    with pytest.raises(Exception, match="not supported"):
        generator.set_inputs(_named(_arrays(family, ids, [pixels + 5])))
    np.testing.assert_array_equal(generator.get_sequence(0), sequence)
    np.testing.assert_array_equal(generator.get_logits(), logits)


@pytest.mark.parametrize("family", FAMILIES)
def test_rewind_guard_tracks_latest_image_boundary(model_factory, family):
    model = model_factory(family)
    generator, history = _generator(model), _History(family)
    generator.set_inputs(_named(history.image(2)))
    old_boundary = len(history.tokens)
    _respond(generator, history)
    generator.set_inputs(_named(history.image(8)))
    latest_boundary = len(history.tokens)
    history.text([4, 5, 6])
    generator.append_tokens(np.array([[4, 5, 6]], dtype=np.int32))
    sequence, logits = generator.get_sequence(0).copy(), generator.get_logits().copy()
    for target in (old_boundary, latest_boundary - 1, latest_boundary):
        with pytest.raises(Exception, match="multimodal prompt boundary"):
            generator.rewind_to(target)
        np.testing.assert_array_equal(generator.get_sequence(0), sequence)
        np.testing.assert_array_equal(generator.get_logits(), logits)
    generator.rewind_to(latest_boundary + 1)
    generator.append_tokens(np.array([[5, 6]], dtype=np.int32))
    _assert_prefill(generator, history, model)
    _respond(generator, history)


@pytest.mark.parametrize("family", FAMILIES)
def test_synthetic_graphs_are_causal_and_cache_position_pixel_sensitive(family):
    """Validate the fixture independently, including causal logits at every prefix."""
    history = _History(family)
    arrays = history.image(2)
    vision = ReferenceEvaluator(make_vision_model(family))
    features = vision.run(None, {name: arrays[name] for name in vision.input_names})[0]
    changed_pixels = {name: arrays[name] for name in vision.input_names}
    changed_pixels["pixel_values"] = changed_pixels["pixel_values"] + 1
    assert not np.array_equal(features, vision.run(None, changed_pixels)[0])
    embedding = ReferenceEvaluator(make_embedding_model(family))
    embeds = embedding.run(None, {"input_ids": arrays["input_ids"], "image_features": features})[0]
    expected_embeds, positions = history.embeddings_and_positions()
    np.testing.assert_array_equal(embeds.reshape(-1), expected_embeds)
    decoder = ReferenceEvaluator(make_decoder_model(family))
    position_ids = positions.T[:, None, :] if family in QWEN_FAMILIES else positions[None, :, 0]
    empty_cache = np.empty((1, 1, 0, 1), dtype=np.float32)
    feeds = {
        "inputs_embeds": embeds,
        "position_ids": position_ids,
        "attention_mask": np.ones((1, len(history.tokens)), dtype=np.int64),
        "past_key_values.0.key": empty_cache,
        "past_key_values.0.value": empty_cache,
    }
    full = decoder.run(None, feeds)
    np.testing.assert_array_equal(full[0][0, -1], history.logits())
    past_key = past_value = empty_cache
    for index in range(len(history.tokens)):
        step = dict(feeds)
        step.update(
            {
                "inputs_embeds": embeds[:, index : index + 1],
                "position_ids": position_ids[..., index : index + 1],
                "attention_mask": feeds["attention_mask"][:, : index + 1],
                "past_key_values.0.key": past_key,
                "past_key_values.0.value": past_value,
            }
        )
        logits, past_key, past_value = decoder.run(None, step)
        np.testing.assert_array_equal(logits[0, -1], full[0][0, index])
    for name in ("past_key_values.0.key", "past_key_values.0.value", "position_ids"):
        perturbed = dict(step)
        perturbed[name] = perturbed[name] + 1
        assert not np.array_equal(decoder.run(None, perturbed)[0], logits), name
    np.testing.assert_array_equal(past_key, full[1])
    np.testing.assert_array_equal(past_value, full[2])

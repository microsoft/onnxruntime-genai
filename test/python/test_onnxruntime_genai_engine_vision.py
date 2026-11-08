# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Real-checkpoint gates; set ORTGENAI_REQUIRE_ENGINE_VISION=1 in qualification jobs."""

import gc
import json
import os
import struct
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest
from PIL import Image, ImageDraw, ImageFont

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>\n"
VIDEO_MARKER = "<|vision_start|><|video_pad|><|vision_end|>"


def vision_modes():
    directory = os.environ.get("ORTGENAI_ENGINE_VISION_MODEL")
    if directory and Path(directory, "genai_config.json").is_file():
        manifest = json.loads(Path(directory, "genai_config.json").read_text())
        if manifest["model"]["type"] == "qwen4_exp":
            return [pytest.param((False, False), id="eager-target")]
    return [
        pytest.param((False, False), id="eager-target"),
        pytest.param((False, True), id="eager-dflash"),
        pytest.param((True, False), id="graph-target"),
        pytest.param((True, True), id="graph-dflash"),
    ]


def vision_profile(manifest):
    profile = (2, 6, 6) if manifest["model"]["type"] == "qwen4_exp" else (4, 512, 64)
    batching = manifest["engine"]["dynamic_batching"]
    defaults = {"max_batch_size": 16, "max_scheduled_tokens": 2048}
    for field, required in zip(("max_batch_size", "max_scheduled_tokens"), profile[:2], strict=True):
        assert batching.get(field, defaults[field]) >= required, f"Qualification requires {field} >= {required}"
    chunk_size = manifest["search"].get("chunk_size", 0)
    assert not chunk_size or chunk_size >= profile[2], f"Qualification requires chunk_size >= {profile[2]}"
    return profile


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen4_exp"])
@pytest.mark.parametrize(
    "omitted_fields",
    [("max_batch_size",), ("max_scheduled_tokens",), ("max_batch_size", "max_scheduled_tokens")],
)
def test_qualification_uses_runtime_defaults_for_omitted_limits(model_type, omitted_fields):
    profile = (2, 6, 6) if model_type == "qwen4_exp" else (4, 512, 64)
    manifest = {
        "model": {"type": model_type},
        "engine": {"dynamic_batching": {"max_batch_size": profile[0], "max_scheduled_tokens": profile[1]}},
        "search": {"chunk_size": profile[2]},
    }
    for field in omitted_fields:
        del manifest["engine"]["dynamic_batching"][field]
    assert vision_profile(manifest) == profile


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen4_exp"])
@pytest.mark.parametrize("field", ["max_batch_size", "max_scheduled_tokens", "chunk_size"])
def test_qualification_rejects_reduced_execution_limits(model_type, field):
    batch_size, token_cap, chunk_size = (2, 6, 6) if model_type == "qwen4_exp" else (4, 512, 64)
    manifest = {
        "model": {"type": model_type},
        "engine": {"dynamic_batching": {"max_batch_size": batch_size, "max_scheduled_tokens": token_cap}},
        "search": {"chunk_size": chunk_size},
    }
    limits = manifest["search"] if field == "chunk_size" else manifest["engine"]["dynamic_batching"]
    limits[field] -= 1
    with pytest.raises(AssertionError, match=field):
        vision_profile(manifest)
    if field != "chunk_size":
        limits[field] = 0
        with pytest.raises(AssertionError, match=field):
            vision_profile(manifest)


@pytest.fixture(scope="module")
def vision_package():
    directory = os.environ.get("ORTGENAI_ENGINE_VISION_MODEL")
    if not directory or not Path(directory, "genai_config.json").is_file():
        if os.environ.get("ORTGENAI_REQUIRE_ENGINE_VISION") == "1":
            pytest.fail("ORTGENAI_ENGINE_VISION_MODEL must name the real packed Qwen vision package")
        pytest.skip("Set ORTGENAI_ENGINE_VISION_MODEL to run real vision qualification")
    if not og.is_cuda_available():
        pytest.fail("Real Engine vision qualification requires CUDA")
    manifest = json.loads(Path(directory, "genai_config.json").read_text())
    assert manifest["model"]["type"] in {"qwen3_5", "qwen4_exp"}
    assert manifest["model"]["vision"]["filename"]
    assert manifest["model"]["embedding"]["filename"]
    return directory, manifest


@pytest.fixture(
    scope="module",
    params=vision_modes(),
)
def vision_model(request, vision_package):
    directory, manifest = vision_package
    graphs, drafts = request.param
    batch_size, token_cap, chunk_size = vision_profile(manifest)
    config = og.Config(directory)
    overlay = {
        "model": {
            "decoder": {
                "session_options": {
                    "ep.cuda.fpa_intb_gemm": "1",
                    "session.use_device_allocator_for_initializers": "1",
                    "ep.cuda.matmul_nbits_m_chunk_size": "512",
                    "ep.cuda.enable_gemm_auto_tune": "1",
                }
            },
            "embedding": {"prefault": True},
        },
        "engine": {
            "dynamic_batching": {
                "num_blocks": 64,
                "max_batch_size": batch_size,
                "max_scheduled_tokens": token_cap,
                "prefix_caching": True,
                "media_device_bytes": (1 if manifest["model"]["type"] == "qwen4_exp" else 2) * 1024 * 1024,
            }
        },
        "search": {"chunk_size": chunk_size},
    }
    if manifest["model"]["type"] == "qwen4_exp":
        del overlay["model"]["decoder"]
    if drafts:
        assert manifest["model"]["dflash2"]["filename"], "The mandatory drafter must not be silently absent"
    else:
        overlay["model"]["dflash2"] = {"filename": ""}
    config.overlay(json.dumps(overlay))
    config.set_provider_option("cuda", "enable_cuda_graph", "1" if graphs else "0")
    if manifest["model"]["type"] == "qwen4_exp":
        config.set_provider_option("cuda", "arena_extend_strategy", "kSameAsRequested")
    model = og.Model(config)
    yield model, drafts
    del model
    gc.collect()


def prompt(tokenizer, count, question="Describe the images briefly."):
    value = tokenizer.apply_chat_template(
        messages=json.dumps([{"role": "user", "content": IMAGE_MARKER * count + question}]),
        add_generation_prompt=True,
    )
    assert value.endswith("<think>\n")
    return value + "\n</think>\n\n"


def new_request(engine):
    options = og.RequestOptions()
    options.set_max_session_tokens(4096)
    return engine.create_request(options=options)


def begin(request, value, images=None, maximum=32):
    options = og.TurnOptions(request)
    options.set_do_sample(False)
    options.set_max_generated_tokens(maximum)
    return request.begin_turn(value, options, images=images)


def drain(engine, capacity=1):
    buffer = engine.create_event_buffer(capacity)
    tokens, finishes = [], []
    runs = 0
    while engine.has_pending_requests():
        for event in engine.run(buffer):
            assert not event.flags & og.EngineEventFlags.FAILED, event.error_code
            assert not event.flags & og.EngineEventFlags.CAPACITY_BLOCKED, event.error_code
            if event.flags & og.EngineEventFlags.TOKEN:
                tokens.append(event.token)
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                finishes.append(
                    (
                        event.turn_id,
                        event.usage.prompt_tokens,
                        event.usage.generated_tokens,
                        event.usage.cached_prompt_tokens,
                    )
                )
        runs += 1
        assert runs < 4096, "Media preparation must make bounded progress"
    return tokens, finishes, runs


@pytest.fixture
def video_model(vision_model, vision_package):
    if not vision_package[1]["model"]["vision"].get("video_config_filename"):
        if os.environ.get("ORTGENAI_REQUIRE_ENGINE_VIDEO") == "1":
            pytest.fail("Video qualification requires the source-bound video processor asset")
        pytest.skip("This image package has no video processor asset")
    return vision_model


@pytest.mark.parametrize("first_color", ["red", "blue"])
def test_video_frame_order_and_snapshot_ownership(video_model, first_color):
    model, _ = video_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    frames = np.empty((4, 64, 96, 3), dtype=np.uint8)
    first, last = ((255, 0, 0), (0, 0, 255)) if first_color == "red" else ((0, 0, 255), (255, 0, 0))
    frames[:2] = first
    frames[2:] = last
    videos = og.Videos.from_frames(frames, timestamps=[0.0, 1.0, 2.0, 3.0])
    frames.fill(0)
    value = prompt(
        tokenizer, 0, VIDEO_MARKER + "Which color appears FIRST in the video: red or blue? Answer just the color."
    )
    options = og.TurnOptions(request)
    options.set_do_sample(False)
    options.set_max_generated_tokens(16)
    turn = request.begin_turn(value, options, videos=videos)
    del videos, frames
    tokens, finished, runs = drain(engine)
    assert turn == 1
    assert len(finished) == 1
    assert runs >= 3, "Two distinct temporal units must be prepared before decoder execution"
    expanded = value.replace(
        VIDEO_MARKER,
        "<|vision_start|>"
        + "".join(
            f"<{time} seconds><|vision_start|>" + "<|video_pad|>" * 6 + "<|vision_end|>" for time in ("0.5", "2.5")
        )
        + "<|vision_end|>",
    )
    assert finished[0][1] == len(tokenizer.encode(expanded))
    assert tokenizer.decode(tokens).strip().lower().startswith(first_color)
    request.close()


def test_video_model_rejection_preserves_admission_identity(video_model):
    model, _ = video_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    value = prompt(tokenizer, 0, VIDEO_MARKER + "Describe the video.")
    options = og.TurnOptions(request)
    options.set_max_generated_tokens(1)
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    single = og.Videos.from_frames([frame], timestamps=[0.0])
    with pytest.raises(RuntimeError, match="frame count"):
        request.begin_turn(value, options, videos=single)
    mismatched = og.Videos.from_frames([frame, frame[:32]], timestamps=[0.0, 1.0])
    with pytest.raises(RuntimeError, match="uniform source geometry"):
        request.begin_turn(value, options, videos=mismatched)
    valid = og.Videos.from_frames([frame, frame], timestamps=[0.0, 1.0])
    turn = request.begin_turn(value, options, videos=valid)
    _, finished, _ = drain(engine)
    assert turn == 1
    assert finished[0][0] == 1
    request.close()


def test_video_input_lifetime_and_real_draft_activity(video_model):
    model, drafts = video_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    frames = np.zeros((4, 64, 96, 3), dtype=np.uint8)
    frames[:2, :, :, 0] = 255
    frames[2:, :, :, 2] = 255
    videos = og.Videos.from_frames(frames, timestamps=[0.0, 1.0, 2.0, 3.0])
    value = prompt(tokenizer, 0, VIDEO_MARKER + "Describe the color changes in this video in one sentence.")
    options = og.TurnOptions(request)
    options.set_do_sample(False)
    options.set_max_generated_tokens(32)
    request.begin_turn(value, options, videos=videos)
    del videos, frames
    tokens, finishes, runs = drain(engine, capacity=64)
    answer = tokenizer.decode(tokens).lower()
    assert "red" in answer and "blue" in answer, answer
    assert len(finishes) == 1
    stats = engine.get_speculative_stats()
    assert runs - stats["target_forward_passes"] == 2, "Expected one encoder-only Run per temporal unit"
    if drafts:
        assert stats["draft_forward_passes"] > 0
        assert stats["draft_tokens_proposed"] > 0
        assert stats["draft_tokens_evaluated"] > 0
        assert stats["dflash2_failures"] == 0
    request.close()


@pytest.mark.parametrize("spare_rows", [0, 1, 2])
def test_video_expanded_capacity_and_recovery(video_model, spare_rows):
    model, _ = video_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    value = prompt(tokenizer, 0, VIDEO_MARKER + "Describe the video.")
    expanded = value.replace(
        VIDEO_MARKER,
        "<|vision_start|>"
        + "".join(
            f"<{time} seconds><|vision_start|>" + "<|video_pad|>" * 6 + "<|vision_end|>" for time in ("0.5", "2.5")
        )
        + "<|vision_end|>",
    )
    rows = len(tokenizer.encode(expanded))
    request_options = og.RequestOptions()
    request_options.set_max_session_tokens(rows + spare_rows)
    request = engine.create_request(options=request_options)
    options = og.TurnOptions(request)
    options.set_max_generated_tokens(1)
    videos = og.Videos.from_frames(np.zeros((4, 64, 96, 3), dtype=np.uint8), timestamps=[0.0, 1.0, 2.0, 3.0])
    if spare_rows == 0:
        with pytest.raises(RuntimeError, match="remaining session capacity"):
            request.begin_turn(value, options, videos=videos)
        assert not engine.has_pending_requests()
        turn = request.begin_turn("ABC", options)
        _, finished, _ = drain(engine)
        assert turn == finished[0][0] == 1
    else:
        turn = request.begin_turn(value, options, videos=videos)
        _, finished, _ = drain(engine)
        assert turn == finished[0][0] == 1
        assert finished[0][1:3] == (rows, 1)
    request.close()


def test_video_mixed_images_and_ordered_clips(video_model):
    model, _ = video_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    frames = np.empty((4, 64, 96, 3), dtype=np.uint8)
    frames[:2] = (255, 0, 0)
    frames[2:] = (0, 0, 255)
    videos = og.Videos.from_clips(
        [frames, frames[::-1].copy()], timestamps=[[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]]
    )
    images = og.Images.open(str(IMAGE_DIR / "cars.jpg"))
    value = prompt(
        tokenizer,
        0,
        "Still image:\n"
        + IMAGE_MARKER
        + "Video 1:\n"
        + VIDEO_MARKER
        + "\nVideo 2:\n"
        + VIDEO_MARKER
        + "\n"
        + "Ignore the still image. Which color appears FIRST in the SECOND video: red or blue? Answer just the color.",
    )
    options = og.TurnOptions(request)
    options.set_do_sample(False)
    options.set_max_generated_tokens(16)
    turn = request.begin_turn(value, options, images=images, videos=videos)
    del images, videos, frames
    tokens, finished, _ = drain(engine)
    assert turn == 1
    assert len(finished) == 1
    assert tokenizer.decode(tokens).strip().lower().startswith("blue")
    request.close()


@pytest.mark.parametrize("first_result", ["FAILED", "PASSED"])
def test_video_terminal_result_order_under_feature_pressure(video_model, first_result):
    model, _ = video_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    frames = []
    font = ImageFont.load_default(size=20)
    for result in (first_result, "PASSED" if first_result == "FAILED" else "FAILED"):
        with Image.new("RGB", (384, 288), "#181818") as image:
            draw = ImageDraw.Draw(image)
            draw.text((16, 16), "$ python -m pytest -q", fill="white", font=font)
            draw.text((16, 80), "test_cache.py::test_order", fill="white", font=font)
            draw.text((16, 144), result, fill="red" if result == "FAILED" else "lime", font=font)
            frames.extend([np.asarray(image).copy()] * 2)
    videos = og.Videos.from_frames(frames, timestamps=[0.0, 1.0, 2.0, 3.0])
    value = prompt(
        tokenizer,
        0,
        VIDEO_MARKER + "Which test result occurred FIRST in the video: FAILED or PASSED? Answer just FAILED or PASSED.",
    )
    options = og.TurnOptions(request)
    options.set_do_sample(False)
    options.set_max_generated_tokens(16)
    request.begin_turn(value, options, videos=videos)
    del videos, frames
    tokens, finished, runs = drain(engine)
    assert len(finished) == 1
    assert runs >= 3
    assert tokenizer.decode(tokens).strip().upper().startswith(first_result)
    request.close()


def test_video_prefix_identity_includes_order_and_timestamp_text(video_model):
    model, _ = video_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    notes = "These surrounding notes are unrelated to the video. " * 40
    value = prompt(
        tokenizer,
        0,
        notes + VIDEO_MARKER + notes + "Which color appears FIRST in the video: red or blue? Answer just the color.",
    )
    frames = np.empty((4, 64, 96, 3), dtype=np.uint8)
    frames[:2] = (255, 0, 0)
    frames[2:] = (0, 0, 255)
    results = []
    for clip, timestamps, color in (
        (frames, [0.0, 1.0, 2.0, 3.0], "red"),
        (frames, [0.0, 1.0, 2.0, 3.0], "red"),
        (frames, [10.0, 11.0, 12.0, 13.0], "red"),
        (frames[::-1].copy(), [0.0, 1.0, 2.0, 3.0], "blue"),
    ):
        request = new_request(engine)
        options = og.TurnOptions(request)
        options.set_do_sample(False)
        options.set_max_generated_tokens(16)
        videos = og.Videos.from_frames(clip, timestamps=timestamps)
        request.begin_turn(value, options, videos=videos)
        tokens, finished, _ = drain(engine)
        answer = tokenizer.decode(tokens).strip().lower()
        assert color in answer and ("blue" if color == "red" else "red") not in answer, answer
        assert len(finished) == 1
        results.append(finished[0][3])
        request.close()
    assert results[0] == 0
    assert results[1] > results[2] > 0, "Changed timestamps must invalidate video-dependent prefix blocks"
    assert results[1] > results[3] > 0, "Changed frame order must invalidate video-dependent prefix blocks"


def test_mixed_geometry_padding_is_rejected_before_host_budget_is_exceeded(vision_package, tmp_path):
    directory, _ = vision_package
    config = og.Config(directory)
    config.overlay(
        json.dumps(
            {
                "model": {"dflash2": {"filename": ""}},
                "engine": {
                    "dynamic_batching": {
                        "num_blocks": 64,
                        "max_batch_size": 1,
                        "media_device_bytes": 32 * 1024 * 1024,
                    }
                },
            }
        )
    )
    config.set_provider_option("cuda", "enable_cuda_graph", "0")
    model = og.Model(config)
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    paths = []
    for size in (1024, 256):
        path = tmp_path / f"image-{size}.png"
        with Image.new("RGB", (size, size), "blue") as image:
            image.save(path)
        paths.append(str(path))
    images = og.Images.open(paths[0], *([paths[1]] * 11))

    with pytest.raises(RuntimeError, match="media_host_bytes"):
        begin(request, prompt(tokenizer, 12), images, maximum=1)
    turn = begin(request, prompt(tokenizer, 0), maximum=1)
    assert turn == 1
    request.close()


def test_token_admission_rejects_image_markers_and_preserves_turn_ids(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    options = og.TurnOptions(request)
    options.set_max_generated_tokens(1)
    markers = ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>", IMAGE_MARKER)
    for value in markers:
        with pytest.raises(RuntimeError, match="Image markers require"):
            request.begin_turn(tokenizer.encode(value), options)
    images = og.Images.open(str(IMAGE_DIR / "cars.jpg"))
    first = begin(request, prompt(tokenizer, 1), images, maximum=1)
    assert first == 1
    drain(engine)
    for value in markers:
        with pytest.raises(RuntimeError, match="Image markers require"):
            request.begin_turn(tokenizer.encode(value), options)
    second = request.begin_turn(tokenizer.encode("ABC"), options)
    assert second == 2
    drain(engine)
    request.close()


def test_video_markers_reject_without_frames_on_image_packages(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    options = og.TurnOptions(request)
    options.set_max_generated_tokens(1)
    for value in ("<|video_pad|>", VIDEO_MARKER):
        for prompt_value in (value, tokenizer.encode(value)):
            with pytest.raises(RuntimeError, match="Video markers require"):
                request.begin_turn(prompt_value, options)
    first = request.begin_turn(tokenizer.encode("ABC"), options)
    assert first == 1
    drain(engine)
    request.close()


def test_formatted_admission_reclaims_abandoned_media_before_host_preflight(vision_package):
    directory, _ = vision_package
    config = og.Config(directory)
    config.overlay(
        json.dumps(
            {
                "model": {"dflash2": {"filename": ""}},
                "engine": {
                    "dynamic_batching": {"num_blocks": 64, "max_batch_size": 1, "media_host_bytes": 21 * 1024 * 1024}
                },
            }
        )
    )
    config.set_provider_option("cuda", "enable_cuda_graph", "0")
    model = og.Model(config)
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    abandoned, survivor = new_request(engine), new_request(engine)
    value = prompt(tokenizer, 1)
    images = og.Images.open(str(IMAGE_DIR / "cars.jpg"))
    abandoned_turn = begin(abandoned, value, images, maximum=1)
    assert abandoned_turn == 1
    with pytest.raises(RuntimeError, match="host capacity"):
        begin(survivor, value, images, maximum=1)
    del abandoned
    gc.collect()
    survivor_turn = begin(survivor, value, images, maximum=1)
    assert survivor_turn == 1
    survivor.close()


def test_text_only_export_rejects_images_without_consuming_the_turn(vision_package):
    directory, manifest = vision_package
    config = og.Config(directory)
    config.overlay(
        json.dumps(
            {
                "model": {
                    "type": manifest["model"]["type"] + "_text",
                    "vision": {"filename": ""},
                    "dflash2": {"filename": ""},
                },
                "engine": {"dynamic_batching": {"num_blocks": 64, "max_batch_size": 1}},
            }
        )
    )
    config.set_provider_option("cuda", "enable_cuda_graph", "0")
    model = og.Model(config)
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    images = og.Images.open(str(IMAGE_DIR / "cars.jpg"))
    with pytest.raises(RuntimeError, match="does not support image inputs"):
        begin(request, prompt(tokenizer, 1), images, maximum=1)
    turn = begin(request, prompt(tokenizer, 0), maximum=1)
    assert turn == 1
    _, finishes, _ = drain(engine)
    assert len(finishes) == 1
    request.close()


def test_vision_export_requires_multimodal_position_bindings(vision_package):
    directory, manifest = vision_package
    bindings = [("position_ids", "three-axis int64 decoder position IDs")]
    if manifest["model"]["type"] == "qwen4_exp":
        bindings.append(("rotary_position_ids", "three-axis int64 sparse-indexer rotary history"))
    for binding, diagnostic in bindings:
        config = og.Config(directory)
        config.overlay(json.dumps({"model": {"decoder": {"inputs": {binding: "missing_position_input"}}}}))
        config.set_provider_option("cuda", "enable_cuda_graph", "0")
        with pytest.raises(RuntimeError, match=diagnostic):
            og.Model(config)


def test_image_input_lifetime_and_real_draft_activity(vision_model):
    model, drafts = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    images = og.Images.open(str(IMAGE_DIR / "cars.jpg"))
    turn = begin(request, prompt(tokenizer, 1, "What kind of vehicles are shown? Answer briefly."), images)
    assert turn == 1
    del images
    gc.collect()
    tokens, finishes, runs = drain(engine, capacity=64)
    answer = tokenizer.decode(tokens).lower()
    assert "car" in answer or "vehicle" in answer, answer
    assert len(finishes) == 1
    stats = engine.get_speculative_stats()
    assert runs - stats["target_forward_passes"] == 1, "Expected exactly one encoder-only Run"
    if drafts:
        assert stats["draft_forward_passes"] > 0
        assert stats["draft_tokens_proposed"] > 0
        assert stats["draft_tokens_evaluated"] > 0
        assert stats["dflash2_failures"] == 0
    request.close()


def test_controlled_image_substitution_changes_grounded_answer(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    value = prompt(
        tokenizer, 1, "Classify the image as either vehicles or a spreadsheet. Reply with only the category."
    )
    answers, streams = [], []
    for filename in ("cars.jpg", "sheet.png"):
        request = new_request(engine)
        begin(request, value, og.Images.open(str(IMAGE_DIR / filename)))
        tokens, finishes, _ = drain(engine)
        assert len(finishes) == 1
        answers.append(tokenizer.decode(tokens).lower())
        streams.append(tokens)
        request.close()
    assert "car" in answers[0] or "vehicle" in answers[0], answers[0]
    assert "spreadsheet" in answers[1], answers[1]
    assert streams[0] != streams[1], "Identical prompts must respond to the substituted image content"


def test_two_image_pressure_prefix_identity_and_rewind(vision_model, tmp_path):
    from PIL import Image, ImageOps  # noqa: PLC0415

    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    value = prompt(tokenizer, 2, "What type of document is in the second image? Answer briefly.")
    paths = [str(IMAGE_DIR / "cars.jpg"), str(IMAGE_DIR / "sheet.png")]
    first = new_request(engine)
    begin(first, value, og.Images.open(*paths))
    expected, finishes, runs = drain(engine, capacity=64)
    assert len(finishes) == 1
    assert runs - engine.get_speculative_stats()["target_forward_passes"] == 2, "Both images need scheduled encoding"
    answer = tokenizer.decode(expected).lower()
    assert any(word in answer for word in ("spreadsheet", "table", "financial")), answer
    first.close()

    repeated = new_request(engine)
    turn = begin(repeated, value, og.Images.open(*paths))
    actual, finishes, _ = drain(engine)
    assert actual == expected
    cached = finishes[0][3]
    assert cached > 0, "The hybrid target must restore actual media-dependent prefix state"
    repeated.rewind_to_start_of_turn(turn)
    replay_turn = begin(repeated, value, og.Images.open(*paths))
    assert replay_turn > turn
    replayed, _, _ = drain(engine)
    assert replayed == expected
    repeated.close()

    changed_path = tmp_path / "changed.png"
    with Image.open(paths[1]) as image:
        ImageOps.invert(image.convert("RGB")).save(changed_path)
    changed = new_request(engine)
    begin(changed, value, og.Images.open(paths[0], str(changed_path)))
    _, changed_finishes, _ = drain(engine)
    assert changed_finishes[0][1] == finishes[0][1], "Different pixels retain identical placeholder token counts"
    assert changed_finishes[0][3] < cached, "Different pixels must not adopt media-dependent state"
    changed.close()


def test_admission_failures_preserve_turn_id_and_preflight_allocations(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    value = prompt(tokenizer, 1)
    with pytest.raises(RuntimeError, match="marker count"):
        begin(request, value)
    with pytest.raises((ValueError, RuntimeError), match="NUL"):
        begin(request, "bad\0prompt")
    encoded = bytearray((IMAGE_DIR / "sheet.png").read_bytes())
    encoded[16:24] = struct.pack(">II", 16384, 16384)
    oversized = og.Images.open_bytes(bytes(encoded))
    with pytest.raises(RuntimeError, match="media_host_bytes|device media budget"):
        begin(request, value, oversized)
    turn = begin(request, value, og.Images.open(str(IMAGE_DIR / "cars.jpg")))
    assert turn == 1
    drain(engine)
    request.close()


def test_text_api_continuation_after_media_and_cancellation(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    first = begin(request, prompt(tokenizer, 1), og.Images.open(str(IMAGE_DIR / "sheet.png")), maximum=16)
    drain(engine)
    suffix = (
        "<|im_end|>\n<|im_start|>user\nWhat is the document type?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    next_turn = begin(request, suffix)
    assert next_turn > first
    tokens, finishes, _ = drain(engine)
    assert tokens and len(finishes) == 1
    request.close()

    canceled = new_request(engine)
    turn = begin(canceled, prompt(tokenizer, 1), og.Images.open(str(IMAGE_DIR / "cars.jpg")))
    was_canceled = canceled.cancel_turn(turn)
    assert was_canceled
    drain(engine)
    canceled.close()
    peer = new_request(engine)
    begin(peer, prompt(tokenizer, 0, "What is 2 + 2?"))
    tokens, _, _ = drain(engine)
    assert "4" in tokenizer.decode(tokens)
    peer.close()


def test_mixed_text_image_batch_and_later_image_turn(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    image_request, text_request = new_request(engine), new_request(engine)
    begin(
        image_request,
        prompt(tokenizer, 1, "What kind of vehicles are shown? Answer briefly."),
        og.Images.open(str(IMAGE_DIR / "cars.jpg")),
    )
    begin(text_request, prompt(tokenizer, 0, "What is 2 + 2? Answer briefly."), maximum=16)
    outputs = {image_request: [], text_request: []}
    buffer = engine.create_event_buffer(1)
    runs = 0
    while engine.has_pending_requests():
        for event in engine.run(buffer):
            assert not event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED)
            if event.flags & og.EngineEventFlags.TOKEN:
                outputs[event.request].append(event.token)
        runs += 1
        assert runs < 4096
    assert "car" in tokenizer.decode(outputs[image_request]).lower()
    assert "4" in tokenizer.decode(outputs[text_request])
    suffix = (
        "<|im_end|>\n<|im_start|>user\n"
        + IMAGE_MARKER
        + "What type of document is this new image? Answer briefly.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    begin(image_request, suffix, og.Images.open(str(IMAGE_DIR / "sheet.png")))
    tokens, _, _ = drain(engine)
    answer = tokenizer.decode(tokens).lower()
    assert any(word in answer for word in ("spreadsheet", "table", "financial")), answer
    image_request.close()
    text_request.close()


def test_cancel_after_encoding_and_replay_full_history(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    value = prompt(tokenizer, 1, "What is shown? Answer briefly.")
    turn = begin(request, value, og.Images.open(str(IMAGE_DIR / "cars.jpg")))
    buffer = engine.create_event_buffer(1)
    for _ in range(32):
        before = engine.get_speculative_stats()["target_forward_passes"]
        events = list(engine.run(buffer))
        after = engine.get_speculative_stats()["target_forward_passes"]
        if before == after:
            assert not events
            break
    else:
        pytest.fail("The image must be encoded in a preparation-only Run")
    was_canceled = request.cancel_turn(turn)
    assert was_canceled
    drain(engine)
    request.rewind_to_start_of_turn(turn)
    replay_turn = begin(request, value, og.Images.open(str(IMAGE_DIR / "cars.jpg")), maximum=16)
    tokens, _, _ = drain(engine)
    messages = [
        {"role": "user", "content": IMAGE_MARKER + "What is shown? Answer briefly."},
        {"role": "assistant", "content": tokenizer.decode(tokens)},
        {
            "role": "user",
            "content": "What general category of vehicles are they? Do not name a make or model.",
        },
    ]
    full = tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True) + "\n</think>\n\n"
    retained = list(tokenizer.encode(value)) + tokens
    assert list(tokenizer.encode(full))[: len(retained)] == retained, "This plain user append must preserve history"
    messages[0]["content"] = IMAGE_MARKER + "Identify these vehicles briefly."
    full = tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True) + "\n</think>\n\n"
    assert list(tokenizer.encode(full))[: len(retained)] != retained, "An edited earlier user requires explicit replay"
    request.rewind_to_start_of_turn(replay_turn)
    begin(request, full, og.Images.open(str(IMAGE_DIR / "cars.jpg")))
    replayed, _, _ = drain(engine)
    answer = tokenizer.decode(replayed).lower()
    assert "car" in answer or "vehicle" in answer, answer
    request.close()


def test_media_preserves_seeded_sampling_and_decoded_stops(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    value = prompt(tokenizer, 1, "What kind of vehicles are shown? Answer briefly.")
    sampled = []
    for _ in range(2):
        request = new_request(engine)
        options = og.TurnOptions(request)
        options.set_do_sample(True)
        options.set_top_k(8)
        options.set_top_p(0.9)
        options.set_seed(42)
        options.set_min_generated_tokens(4)
        options.set_max_generated_tokens(8)
        request.begin_turn(value, options, images=og.Images.open(str(IMAGE_DIR / "cars.jpg")))
        tokens, _, _ = drain(engine)
        assert 4 <= len(tokens) <= 8
        sampled.append(tokens)
        request.close()
    assert sampled[0] == sampled[1]

    reference = new_request(engine)
    begin(reference, value, og.Images.open(str(IMAGE_DIR / "cars.jpg")), maximum=12)
    expected, _, _ = drain(engine)
    reference.close()
    stop = tokenizer.decode(expected[:3])
    assert stop
    request = new_request(engine)
    options = og.TurnOptions(request)
    options.set_do_sample(False)
    options.set_stop_strings([stop])
    options.set_max_generated_tokens(12)
    request.begin_turn(value, options, images=og.Images.open(str(IMAGE_DIR / "cars.jpg")))
    buffer = engine.create_event_buffer(1)
    tokens, finishes = [], []
    for _ in range(4096):
        for event in engine.run(buffer):
            assert not event.flags & (og.EngineEventFlags.FAILED | og.EngineEventFlags.CAPACITY_BLOCKED)
            if event.flags & og.EngineEventFlags.TOKEN:
                tokens.append(event.token)
            if event.flags & og.EngineEventFlags.TURN_FINISHED:
                finishes.append((event.finish_reason, event.matched_stop_string_index))
        if not engine.has_pending_requests():
            break
    else:
        pytest.fail("A decoded stop must finish the media turn")
    assert tokens == expected[: len(tokens)]
    assert finishes == [(og.FinishReason.STOP_STRING, 0)]
    request.close()


def test_reference_tool_formatting_retains_image_history(vision_model):
    model, _ = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    messages = [
        {"role": "user", "content": IMAGE_MARKER + "Identify the vehicles. Use the inspection result."},
        {
            "role": "assistant",
            "content": "I will inspect the image.",
            "tool_calls": [{"function": {"name": "inspect", "arguments": {"image": 1}}}],
        },
        {"role": "tool", "content": "The inspected vehicles are cars."},
    ]
    initial = prompt(tokenizer, 1, "Identify the vehicles. Use the inspection result.") + "This reply will be replaced."
    full = tokenizer.apply_chat_template(messages=json.dumps(messages), add_generation_prompt=True) + "\n</think>\n\n"
    initial_ids, full_ids = list(tokenizer.encode(initial)), list(tokenizer.encode(full))
    assert "<tool_call>\n<function=inspect>" in full
    assert "<tool_response>\nThe inspected vehicles are cars." in full
    request = new_request(engine)
    turn = begin(request, initial, og.Images.open(str(IMAGE_DIR / "cars.jpg")), maximum=1)
    generated, _, _ = drain(engine)
    retained = initial_ids + generated
    assert full_ids[: len(retained)] != retained, "Replacing a partial assistant reply requires explicit replay"
    request.rewind_to_start_of_turn(turn)
    begin(request, full, og.Images.open(str(IMAGE_DIR / "cars.jpg")))
    tokens, _, _ = drain(engine)
    assert "car" in tokenizer.decode(tokens).lower()
    request.close()

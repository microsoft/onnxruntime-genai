# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Real-checkpoint gates; set ORTGENAI_REQUIRE_ENGINE_VISION=1 in qualification jobs."""

import gc
import json
import os
import struct
from pathlib import Path

import onnxruntime_genai as og
import pytest

IMAGE_DIR = Path(__file__).resolve().parents[1] / "images"
IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>\n"


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
    assert manifest["model"]["type"] == "qwen3_5"
    assert manifest["model"]["vision"]["filename"]
    assert manifest["model"]["embedding"]["filename"]
    return directory, manifest


@pytest.fixture(
    scope="module",
    params=[(False, False), (False, True), (True, False), (True, True)],
    ids=["eager-target", "eager-dflash", "graph-target", "graph-dflash"],
)
def vision_model(request, vision_package):
    directory, manifest = vision_package
    graphs, drafts = request.param
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
                "max_batch_size": 4,
                "max_scheduled_tokens": 512,
                "prefix_caching": True,
                "media_device_bytes": 2 * 1024 * 1024,
            }
        },
        "search": {"chunk_size": 64},
    }
    if drafts:
        assert manifest["model"]["dflash2"]["filename"], "The mandatory drafter must not be silently absent"
    else:
        overlay["model"]["dflash2"] = {"filename": ""}
    config.overlay(json.dumps(overlay))
    config.set_provider_option("cuda", "enable_cuda_graph", "1" if graphs else "0")
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


def test_formatted_admission_reclaims_abandoned_media_before_host_preflight(vision_package):
    directory, _ = vision_package
    config = og.Config(directory)
    config.overlay(
        json.dumps(
            {
                "model": {"dflash2": {"filename": ""}},
                "engine": {
                    "dynamic_batching": {"num_blocks": 64, "max_batch_size": 1, "media_host_bytes": 16 * 1024 * 1024}
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
    assert begin(abandoned, value, images, maximum=1) == 1
    with pytest.raises(RuntimeError, match="host capacity"):
        begin(survivor, value, images, maximum=1)
    del abandoned
    gc.collect()
    assert begin(survivor, value, images, maximum=1) == 1
    survivor.close()


def test_text_only_export_rejects_images_without_consuming_the_turn(vision_package):
    directory, _ = vision_package
    config = og.Config(directory)
    config.overlay(
        json.dumps(
            {
                "model": {"type": "qwen3_5_text", "vision": {"filename": ""}, "dflash2": {"filename": ""}},
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
    assert begin(request, prompt(tokenizer, 0), maximum=1) == 1
    _, finishes, _ = drain(engine)
    assert len(finishes) == 1
    request.close()


def test_image_input_lifetime_and_real_draft_activity(vision_model):
    model, drafts = vision_model
    engine, tokenizer = og.Engine(model), og.Tokenizer(model)
    request = new_request(engine)
    images = og.Images.open(str(IMAGE_DIR / "cars.jpg"))
    assert begin(request, prompt(tokenizer, 1, "What kind of vehicles are shown? Answer briefly."), images) == 1
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
    assert begin(repeated, value, og.Images.open(*paths)) > turn
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
    assert begin(request, value, og.Images.open(str(IMAGE_DIR / "cars.jpg"))) == 1
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
    assert begin(request, suffix) > first
    tokens, finishes, _ = drain(engine)
    assert tokens and len(finishes) == 1
    request.close()

    canceled = new_request(engine)
    turn = begin(canceled, prompt(tokenizer, 1), og.Images.open(str(IMAGE_DIR / "cars.jpg")))
    assert canceled.cancel_turn(turn)
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
    assert request.cancel_turn(turn)
    drain(engine)
    request.rewind_to_start_of_turn(turn)
    replay_turn = begin(request, value, og.Images.open(str(IMAGE_DIR / "cars.jpg")), maximum=16)
    tokens, _, _ = drain(engine)
    messages = [
        {"role": "user", "content": IMAGE_MARKER + "What is shown? Answer briefly."},
        {"role": "assistant", "content": tokenizer.decode(tokens)},
        {"role": "user", "content": "What kind of vehicles are they? Answer briefly."},
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
    initial = prompt(tokenizer, 1, "Identify the vehicles. Use the inspection result.") + messages[1]["content"]
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

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Real CUDA graph ownership tests (set ORT_QWEN38_TEST_MODEL to an exported package)."""

import gc
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import onnxruntime_genai as og
import pytest


@pytest.fixture(scope="module")
def model_path():
    path = os.environ.get("ORT_QWEN38_TEST_MODEL")
    if not path:
        pytest.skip("ORT_QWEN38_TEST_MODEL not provided")
    path = Path(path)
    assert path.is_dir()
    config = json.loads((path / "genai_config.json").read_text())
    assert config["model"]["qwen38_graph"].get("warmup", True), "ownership tests require graph warmup"
    providers = config["model"]["decoder"]["session_options"]["provider_options"]
    assert any(str(provider.get("cuda", {}).get("enable_cuda_graph")) == "1" for provider in providers)
    plugin = os.environ.get("ORT_QWEN38_TEST_CUDA_PLUGIN")
    if plugin:
        og.register_execution_provider_library("CUDAExecutionProvider", plugin)
    return path


def generate(model):
    # More than one speculative verify is needed to exercise graph replay, not only prefill.
    tokenizer = og.Tokenizer(model)
    prompt = np.asarray(
        tokenizer.encode(
            "<|im_start|>user\nWrite the numbers from one to fifty, separated by commas."
            "<|im_end|>\n<|im_start|>assistant\n"
        ),
        dtype=np.int32,
    )
    engine = og.Engine(model)
    options = og.RequestOptions()
    options.set_max_session_tokens(engine.get_capabilities().max_request_length)
    request = engine.create_request(options=options)
    turn = og.TurnOptions(request)
    turn.set_do_sample(False)
    turn.set_max_generated_tokens(64)
    request.begin_turn(prompt, turn)
    events = engine.create_event_buffer(1)
    tokens = []
    try:
        while engine.has_pending_requests():
            for event in engine.run(events):
                assert not event.flags & og.EngineEventFlags.FAILED, event.error_code
                if event.flags & og.EngineEventFlags.TOKEN:
                    tokens.append(int(event.token))
    finally:
        request.close()
    assert len(tokens) >= 16, tokens
    return tokens


@pytest.mark.parametrize("load_on_worker", [False, True])
def test_serialized_graph_handoffs_preserve_output(model_path, load_on_worker):
    if load_on_worker:
        with ThreadPoolExecutor(max_workers=1) as loader:
            model = loader.submit(og.Model, str(model_path)).result()
        # The loading thread has exited before any request is admitted.
    else:
        model = og.Model(str(model_path))
    try:
        if load_on_worker:
            with ThreadPoolExecutor(max_workers=1) as first_worker:
                expected = first_worker.submit(generate, model).result()
            assert generate(model) == expected
        else:
            expected = generate(model)
        with ThreadPoolExecutor(max_workers=1) as worker:
            for _ in range(2):
                assert worker.submit(generate, model).result() == expected
                assert generate(model) == expected
    finally:
        del model
        gc.collect()


def test_handoff_does_not_reuse_borrowed_path_graph_buffers(model_path):
    def load():
        config = og.Config(str(model_path))
        buffers = {path.name: bytearray(path.read_bytes()) for path in model_path.glob("*.onnx")}
        assert buffers
        for name, buffer in buffers.items():
            config.add_model_data(name, buffer)
        model = og.Model(config)
        for buffer in buffers.values():
            buffer[:] = bytes(len(buffer))
        return model, buffers

    with ThreadPoolExecutor(max_workers=1) as loader:
        model, buffers = loader.submit(load).result()
    try:
        # A new worker may reuse the exited loader's OS ID, but not its lifetime token.
        with ThreadPoolExecutor(max_workers=1) as worker:
            expected = worker.submit(generate, model).result()
        assert generate(model) == expected
    finally:
        del model, buffers
        gc.collect()


def test_failed_warmup_discards_captures_before_retry(model_path, monkeypatch):
    failure = "ORT_QWEN38_TEST_FAIL_AFTER_WARMUP"
    monkeypatch.setenv(failure, "1")
    with pytest.raises(RuntimeError, match="injected initialization failure after warmup"):
        og.Model(str(model_path))
    monkeypatch.delenv(failure)

    model = og.Model(str(model_path))
    try:
        expected = generate(model)
        with ThreadPoolExecutor(max_workers=1) as worker:
            monkeypatch.setenv(failure, "1")
            with pytest.raises(RuntimeError, match="injected initialization failure after warmup"):
                worker.submit(generate, model).result()
            monkeypatch.delenv(failure)
            assert worker.submit(generate, model).result() == expected
        assert generate(model) == expected
    finally:
        del model
        gc.collect()

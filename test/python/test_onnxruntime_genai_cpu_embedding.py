# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Opt-in real-model CPU embedding parity, including target and DFlash graph replay.

Set ORTGENAI_CPU_EMBEDDING_SOURCE_MODEL and ORTGENAI_CPU_EMBEDDING_MODEL to the
original packed model and an independently exported ``exclude_embeds=true`` package.
"""

import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnx_ir as ir
import onnxruntime_genai as og
import pytest
import torch
import transformers

SOURCE = os.environ.get("ORTGENAI_CPU_EMBEDDING_SOURCE_MODEL")
EXPORTED = os.environ.get("ORTGENAI_CPU_EMBEDDING_MODEL")
_DEVICES = ["cpu"] + (["cuda"] if og.is_cuda_available() else [])
real_model = pytest.mark.skipif(
    not SOURCE or not EXPORTED or not og.is_cuda_available(),
    reason="Requires original and CPU-embedding CUDA Engine models",
)


def generate(path, capture, batch_size, prompt_length):
    config = og.Config(path)
    config.set_provider_option("cuda", "enable_cuda_graph", "1" if capture else "0")
    config.overlay(
        json.dumps(
            {
                "engine": {"dynamic_batching": {"max_batch_size": batch_size}},
                "search": {"chunk_size": 128, "do_sample": False},
            }
        )
    )
    model = og.Model(config)
    tokenizer = og.Tokenizer(model)
    unit = list(tokenizer.encode("The quick brown fox jumps over the lazy dog. "))
    outputs = []
    # Recreate the Engine on the same Model to check graph release and shared CPU session lifetime.
    for repetition in range(2):
        engine = og.Engine(model)
        buffer = engine.create_event_buffer(16)
        requests = []
        for slot in range(batch_size):
            count = prompt_length + slot * 9 + repetition
            tokens = (unit * (count // len(unit) + 1))[:count]
            options = og.RequestOptions()
            options.set_max_session_tokens(count + 136)
            request = engine.create_request(options=options)
            turn = og.TurnOptions(request)
            turn.set_do_sample(False)
            turn.set_max_generated_tokens(128)
            request.begin_turn(np.asarray(tokens, dtype=np.int32), turn)
            requests.append(request)
        streams = {request: [] for request in requests}
        steps = 0
        while engine.has_pending_requests():
            steps += 1
            assert steps < 4096
            for event in engine.run(buffer):
                assert not (event.flags & og.EngineEventFlags.FAILED), event.error_code
                if event.flags & og.EngineEventFlags.TOKEN:
                    streams[event.request].append(event.token)
        stats = engine.get_speculative_stats()
        assert not any(value for key, value in stats.items() if "dflash2" in key and "fail" in key), stats
        if json.loads((Path(path) / "genai_config.json").read_text())["model"].get("dflash2"):
            assert stats["acceptance_rate"] > 0, stats
        outputs.append([streams[request] for request in requests])
        assert all(outputs[-1])
        for request in requests:
            request.close()
        del buffer, engine, requests, streams, request, turn, options
        gc.collect()
    del tokenizer, model
    gc.collect()
    return outputs


@pytest.mark.parametrize("batch_size,prompt_length", [(1, 37), (1, 769), (2, 37)])
@real_model
def test_cpu_embedding_matches_gpu_and_eager(batch_size, prompt_length):
    expected = generate(SOURCE, True, batch_size, prompt_length)
    eager = generate(EXPORTED, False, batch_size, prompt_length)
    captured = generate(EXPORTED, True, batch_size, prompt_length)
    assert captured == eager == expected


@pytest.mark.parametrize("device", _DEVICES)
def test_cpu_embedding_engine_synthetic(tmp_path, device, monkeypatch):
    """Exercise the native lookup and embedding-only target binding on a tiny model."""
    original = Path(__file__).resolve().parent.parent / "models" / "engine" / "synthetic-paged"
    exported = tmp_path / "exported"
    shutil.copytree(original, exported)
    graph = onnx.load(original / "decoder.onnx")
    for node in graph.graph.node:
        for index, name in enumerate(node.input):
            if name == "input_ids":
                node.input[index] = "embedded_ids"
    ids = next(value for value in graph.graph.input if value.name == "input_ids")
    graph.graph.input.remove(ids)
    graph.graph.input.append(
        onnx.helper.make_tensor_value_info("inputs_embeds", onnx.TensorProto.FLOAT16, ["num_tokens", 1])
    )
    graph.graph.initializer.append(onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), "embedding.axis"))
    for node in reversed(
        [
            onnx.helper.make_node("Squeeze", ["inputs_embeds", "embedding.axis"], ["embedding.ids"]),
            onnx.helper.make_node("Cast", ["embedding.ids"], ["embedded_ids"], to=onnx.TensorProto.INT64),
        ]
    ):
        graph.graph.node.insert(0, node)
    onnx.save(graph, exported / "decoder.onnx")

    builder_root = Path(__file__).resolve().parents[2] / "src" / "python" / "py" / "models"
    monkeypatch.syspath_prepend(str(builder_root))
    from builders.base import Model  # noqa: PLC0415

    builder_config = transformers.LlamaConfig(
        hidden_size=1,
        intermediate_size=2,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        vocab_size=64,
        max_position_embeddings=128,
        architectures=["LlamaForCausalLM"],
    )
    builder = Model(
        builder_config,
        ir.DataType.FLOAT16,
        ir.DataType.FLOAT16,
        "cuda",
        str(tmp_path / "cache"),
        {"exclude_embeds": True, "use_paged_attention": True},
    )
    embedding = builder.make_embedding_component(torch.arange(64, dtype=torch.float32).reshape(64, 1))
    embedding.save_model(str(exported))
    config_path = exported / "genai_config.json"
    config = json.loads(config_path.read_text())
    del config["model"]["decoder"]["inputs"]["input_ids"]
    config["model"]["decoder"]["inputs"]["inputs_embeds"] = "inputs_embeds"
    config["model"]["embedding"] = {
        "filename": embedding.filename,
        "inputs": {"input_ids": "input_ids"},
        "outputs": {"inputs_embeds": "inputs_embeds"},
    }
    config_path.write_text(json.dumps(config))

    def run(path):
        config = og.Config(str(path))
        config.clear_providers()
        if device == "cuda":
            config.append_provider("cuda")
        model = og.Model(config)
        results = []
        for _ in range(2):
            engine = og.Engine(model)
            buffer = engine.create_event_buffer(8)
            requests = []
            for prompt in ([5, 9, 13], [7, 2, 20, 4]):
                request = engine.create_request()
                turn = og.TurnOptions(request)
                turn.set_max_generated_tokens(16)
                request.begin_turn(np.asarray(prompt, dtype=np.int32), turn)
                requests.append(request)
            streams = {request: [] for request in requests}
            while engine.has_pending_requests():
                for event in engine.run(buffer):
                    if event.flags & og.EngineEventFlags.FAILED:
                        engine.create_request()  # Rethrow the detailed native failure.
                        pytest.fail(str(event.error_code))
                    if event.flags & og.EngineEventFlags.TOKEN:
                        streams[event.request].append(event.token)
            results.append([streams[r] for r in requests])
            for request in requests:
                request.close()
            del buffer, engine, requests, streams, request, turn
            gc.collect()
        del model
        gc.collect()
        return results

    expected = run(original)
    assert run(exported) == expected
    config = json.loads(config_path.read_text())
    config["model"]["embedding"]["outputs"]["inputs_embeds"] = "missing_output"
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="one dynamic int64"):
        og.Model(str(exported))

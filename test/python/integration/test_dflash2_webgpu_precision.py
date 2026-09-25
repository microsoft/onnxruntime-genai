# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Normal-output tests for a builder-exported paged Qwen3.8 WebGPU DFlash2 model.

Run with --dflash2-webgpu-model MODEL_DIR. Engine acceptance is independently
validated by run_qwen38_webgpu_engine.py; this test computes fresh target
auxiliary outputs, without recorded Engine inputs or extra graph outputs.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from transformers import AutoTokenizer

from .run_qwen38_webgpu_engine import PROMPTS, register_webgpu_plugin, validate_dflash2_activity


@pytest.fixture(scope="module")
def drafter_session(pytestconfig, tmp_path_factory):
    model_dir = pytestconfig.getoption("--dflash2-webgpu-model")
    if model_dir is None:
        pytest.skip("Pass --dflash2-webgpu-model to test an exported WebGPU DFlash2 model")
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "genai_config.json").read_text())
    if "WebGpuExecutionProvider" not in ort.get_available_providers():
        assert register_webgpu_plugin(), "WebGPU is required for this model test"
    options = ort.SessionOptions()
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path_factory.mktemp("dflash2-profile") / "profile")
    drafter = config["model"]["dflash2"]
    session = ort.InferenceSession(
        str(model_dir / drafter["filename"]), sess_options=options, providers=["WebGpuExecutionProvider"]
    )
    target = ort.InferenceSession(
        str(model_dir / config["model"]["decoder"]["filename"]), providers=["WebGpuExecutionProvider"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    yield session, target, tokenizer, config
    events = json.loads(Path(session.end_profiling()).read_text())
    required = {"PagedAttention", "MatMulNBits", "SkipSimplifiedLayerNormalization"}
    gpu_ops = {
        e.get("args", {}).get("op_name")
        for e in events
        if e.get("args", {}).get("provider") == "WebGpuExecutionProvider"
    }
    cpu_ops = {
        e.get("args", {}).get("op_name") for e in events if e.get("args", {}).get("provider") == "CPUExecutionProvider"
    }
    assert required <= gpu_ops
    assert not required & cpu_ops


def make_target_inputs(session, config, token_ids):
    """A fresh single-sequence prefill, using the target's normal paged/state inputs."""
    names = config["model"]["decoder"]["inputs"]
    length = len(token_ids)
    values = {
        names["input_ids"]: np.asarray(token_ids, dtype=np.int64),
        names["position_ids"]: np.tile(np.arange(length, dtype=np.int64), (3, 1)),
        names["cumulative_sequence_lengths"]: np.array([0, length], dtype=np.int32),
        names["past_sequence_lengths"]: np.array([0], dtype=np.int32),
        names["block_table"]: np.zeros((1, 1), dtype=np.int32),
        names["attention_metadata"]: np.array([length, length, length], dtype=np.int32),
        names["state_update_capture_count"]: np.zeros(1, dtype=np.int32),
        names["state_update_active"]: np.zeros(1, dtype=np.int32),
    }
    block_size = config["engine"]["dynamic_batching"]["block_size"]
    assert length <= block_size
    dimensions = {"batch_size": 1, "num_blocks": 1, "block_size": block_size}
    dtypes = {"tensor(float16)": np.float16, "tensor(float)": np.float32}
    for value in session.get_inputs():
        if value.name not in values:
            assert value.name.startswith(("past.", "past_key_values.")), value.name
            shape = [dimensions[dim] if isinstance(dim, str) else dim for dim in value.shape]
            values[value.name] = np.zeros(shape, dtype=dtypes[value.type])
    return values


@pytest.mark.parametrize("prompt", [prompt for prompt, _ in PROMPTS])
def test_drafter_outputs_stay_finite_across_steps(drafter_session, prompt):
    session, target, tokenizer, config = drafter_session
    drafter = config["model"]["dflash2"]
    names, outputs = drafter["inputs"], drafter["outputs"]
    block_rows, width, top_k = drafter["block_size"], drafter["num_draft_tokens"], drafter["selector_top_k"]
    inputs = {value.name: value for value in session.get_inputs()}
    cache_shape = inputs[names["past_key_names"] % 0].shape
    block_size = cache_shape[1]
    assert inputs[names["aux_hidden_states"]].type == "tensor(float16)"
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]
    feeds = {}
    for layer in range(drafter["num_hidden_layers"]):
        for kind in ("key", "value"):
            feeds[names[f"past_{kind}_names"] % layer] = np.zeros((1, *cache_shape[1:]), dtype=np.float16)
    past_length = 0
    for _ in range(4):
        target_outputs = config["model"]["decoder"]["outputs"]
        logits, auxiliary = target.run(
            [target_outputs["logits"], target_outputs["aux_hidden_states"]],
            make_target_inputs(target, config, token_ids),
        )
        assert np.isfinite(logits).all() and np.isfinite(auxiliary).all()
        context = auxiliary[past_length:]
        context_rows = len(context)
        assert past_length + context_rows + block_rows <= block_size
        anchor = int(np.argmax(logits[-1]))
        block_ids = np.full(block_rows, drafter["mask_token_id"], dtype=np.int64)
        block_ids[0] = anchor
        feeds.update(
            {
                names["aux_hidden_states"]: context,
                names["input_ids"]: block_ids,
                names["q_row_map"]: np.concatenate(
                    (np.zeros(context_rows, dtype=np.int32), np.arange(block_rows, dtype=np.int32))
                ),
                names["qkv_row_map"]: np.concatenate(
                    (block_rows + np.arange(context_rows, dtype=np.int32), np.arange(block_rows, dtype=np.int32))
                ),
                names["block_row_index"]: context_rows + np.arange(block_rows, dtype=np.int32),
                names["cumulative_sequence_lengths"]: np.array([0, context_rows + block_rows], dtype=np.int32),
                names["past_sequence_lengths"]: np.array([past_length], dtype=np.int32),
                names["block_table"]: np.zeros((1, 1), dtype=np.int32),
            }
        )
        values = dict(zip((value.name for value in session.get_outputs()), session.run(None, feeds), strict=True))
        candidate_ids, scores = values[outputs["candidate_ids"]], values[outputs["scores"]]
        assert candidate_ids.shape == (1, width, top_k)
        assert scores.shape == (1, width, top_k, top_k)
        assert np.all((candidate_ids >= 0) & (candidate_ids < config["model"]["vocab_size"]))
        assert np.isfinite(scores).all()
        for layer in range(drafter["num_hidden_layers"]):
            for kind in ("key", "value"):
                cache = values[outputs[f"present_{kind}_names"] % layer]
                assert np.isfinite(cache).all()
                feeds[names[f"past_{kind}_names"] % layer] = cache
        # Only the context rows are committed; the next call overwrites the speculative block.
        past_length += context_rows
        token_ids.append(anchor)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rounds", 0),
        ("completed_rounds", 1),
        ("draft_tokens_proposed", 0),
        ("draft_tokens_accepted", 0),
        ("dflash2_failures", 1),
        ("dflash2_disables", 1),
        ("standard_fallback_steps", 1),
    ],
)
def test_validation_rejects_inactive_or_failed_speculation(field, value):
    stats = {
        "rounds": 3,
        "completed_rounds": 3,
        "draft_tokens_proposed": 12,
        "draft_tokens_accepted": 6,
        "dflash2_failures": 0,
        "dflash2_disables": 0,
        "standard_fallback_steps": 0,
    }
    validate_dflash2_activity(stats)
    stats[field] = value
    with pytest.raises(RuntimeError):
        validate_dflash2_activity(stats)

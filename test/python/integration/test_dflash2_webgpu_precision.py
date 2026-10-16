# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Output/activity and same-shape target tests for paged Qwen3.8 WebGPU DFlash2.

Run with --dflash2-webgpu-model MODEL_DIR. The runner's answer/activity smoke is
not a token-equivalence test. Here, ordinary graph inputs/outputs compare eager
and verification target execution from identical state without Engine hooks.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from transformers import AutoTokenizer

from .run_qwen38_webgpu_engine import (
    PROMPTS,
    register_webgpu_plugin,
    run_multi_request_activity_smoke,
    validate_dflash2_activity,
    validate_partial_replay_round,
)


@pytest.fixture(scope="module")
def drafter_session(pytestconfig, tmp_path_factory):
    model_dir = pytestconfig.getoption("--dflash2-webgpu-model")
    if model_dir is None:
        pytest.skip("Pass --dflash2-webgpu-model to test an exported WebGPU DFlash2 model")
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "genai_config.json").read_text())
    if "WebGpuExecutionProvider" not in ort.get_available_providers():
        assert register_webgpu_plugin(), "WebGPU is required for this model test"
        import onnxruntime_ep_webgpu as webgpu_ep  # noqa: PLC0415

        ort.register_execution_provider_library(webgpu_ep.get_ep_name(), webgpu_ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == "WebGpuExecutionProvider"]
    assert devices, "Registered WebGPU plugin exposes no devices"
    options = ort.SessionOptions()
    options.add_provider_for_devices([devices[0]], {})
    options.enable_profiling = True
    options.profile_file_prefix = str(tmp_path_factory.mktemp("dflash2-profile") / "profile")
    drafter = config["model"]["dflash2"]
    session = ort.InferenceSession(str(model_dir / drafter["filename"]), sess_options=options, providers=[])
    assert devices[0].ep_name in session.get_providers(), "WebGPU was not selected for the drafter session"
    target_options = ort.SessionOptions()
    target_options.add_provider_for_devices([devices[0]], {"enableGraphCapture": "0"})
    target = ort.InferenceSession(
        str(model_dir / config["model"]["decoder"]["filename"]), sess_options=target_options, providers=[]
    )
    assert devices[0].ep_name in target.get_providers(), "WebGPU was not selected for the target session"
    print(f"Selected {devices[0].ep_name}, device {devices[0].device.device_id} for drafter and target")
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


def make_packed_target_inputs(session, config, sequences, past_lengths, state=None, capture=False):
    """Pack requests in caller order, using one paged block per short sequence."""
    names = config["model"]["decoder"]["inputs"]
    lengths = np.array([len(tokens) for tokens in sequences], dtype=np.int32)
    batch_size = len(sequences)
    total_lengths = lengths + past_lengths
    block_size = config["engine"]["dynamic_batching"]["block_size"]
    assert np.all(total_lengths <= block_size)
    values = {
        names["input_ids"]: np.concatenate(sequences).astype(np.int64),
        names["position_ids"]: np.tile(
            np.concatenate(
                [past + np.arange(length, dtype=np.int64) for past, length in zip(past_lengths, lengths, strict=True)]
            ),
            (3, 1),
        ),
        names["cumulative_sequence_lengths"]: np.concatenate(([0], np.cumsum(lengths))).astype(np.int32),
        names["past_sequence_lengths"]: np.array(past_lengths, dtype=np.int32),
        names["block_table"]: np.arange(batch_size, dtype=np.int32).reshape(batch_size, 1),
        names["attention_metadata"]: np.array([max(lengths), max(total_lengths), max(total_lengths)], dtype=np.int32),
        names["state_update_capture_count"]: lengths - 1 if capture else np.zeros(batch_size, dtype=np.int32),
        names["state_update_active"]: np.array([int(capture)], dtype=np.int32),
    }
    dimensions = {"batch_size": batch_size, "num_blocks": batch_size, "block_size": block_size}
    dtypes = {"tensor(float16)": np.float16, "tensor(float)": np.float32}
    for value in session.get_inputs():
        if value.name not in values:
            assert value.name.startswith(("past.", "past_key_values.")), value.name
            shape = [dimensions[dim] if isinstance(dim, str) else dim for dim in value.shape]
            values[value.name] = (
                state[value.name].copy() if state is not None else np.zeros(shape, dtype=dtypes[value.type])
            )
    return values


def make_target_inputs(session, config, token_ids):
    return make_packed_target_inputs(session, config, [token_ids], [0])


def target_state_pairs(config):
    decoder = config["model"]["decoder"]
    for group in decoder["state_groups"]:
        kinds = {"paged_kv": ("key", "value"), "fixed_conv": ("conv",), "fixed_recurrent": ("recurrent",)}
        for layer in group["layer_ids"]:
            for kind in kinds[group["kind"]]:
                yield (
                    decoder["inputs"][f"past_{kind}_names"] % layer,
                    decoder["outputs"][f"present_{kind}_names"] % layer,
                )


def logit_comparison(reference, actual):
    """Keep full-vocabulary error and exact argmax separate, including near-tie diagnostics."""
    reference, actual = np.asarray(reference, dtype=np.float32), np.asarray(actual, dtype=np.float32)
    assert reference.shape == actual.shape and reference.ndim == 2
    assert np.isfinite(reference).all() and np.isfinite(actual).all()
    difference = np.abs(reference - actual)

    def top_two(logits):
        indices = np.argsort(-logits, axis=-1, kind="stable")[:, :2]
        values = np.take_along_axis(logits, indices, axis=-1)
        return {
            "tokens": indices.tolist(),
            "logits": values.tolist(),
            "margins": (values[:, 0] - values[:, 1]).tolist(),
        }

    return {
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "argmax_agreement": (reference.argmax(axis=-1) == actual.argmax(axis=-1)).tolist(),
        "eager_top_two": top_two(reference),
        "verification_top_two": top_two(actual),
    }


def test_logit_comparison_exposes_near_tie_argmax_flip():
    reference = np.array([[1.0, 0.9999, -1.0]], dtype=np.float32)
    actual = reference[:, [1, 0, 2]]
    report = logit_comparison(reference, actual)
    np.testing.assert_allclose(reference, actual, rtol=1e-3, atol=2e-3)
    assert report["argmax_agreement"] == [False]
    assert report["eager_top_two"]["tokens"] == [[0, 1]]
    assert report["verification_top_two"]["tokens"] == [[1, 0]]
    assert 0 < report["eager_top_two"]["margins"][0] < 2e-4


def make_drafter_feeds(session, config, context, anchor, past_length=0, cache=None):
    drafter = config["model"]["dflash2"]
    names = drafter["inputs"]
    inputs = {value.name: value for value in session.get_inputs()}
    block_rows, context_rows = drafter["block_size"], len(context)
    cache_shape = inputs[names["past_key_names"] % 0].shape
    assert past_length + context_rows + block_rows <= cache_shape[1]
    feeds = {}
    for layer in range(drafter["num_hidden_layers"]):
        for kind in ("key", "value"):
            name = names[f"past_{kind}_names"] % layer
            feeds[name] = cache[name] if cache is not None else np.zeros((1, *cache_shape[1:]), dtype=np.float16)
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
    if names["attention_metadata"]:
        length = context_rows + block_rows
        feeds[names["attention_metadata"]] = np.array(
            [length, past_length + length, past_length + length], dtype=np.int32
        )
    return feeds


@pytest.mark.parametrize("prompt", [prompt for prompt, _ in PROMPTS])
def test_drafter_outputs_stay_finite_across_steps(drafter_session, prompt):
    session, target, tokenizer, config = drafter_session
    drafter = config["model"]["dflash2"]
    names, outputs = drafter["inputs"], drafter["outputs"]
    width, top_k = drafter["num_draft_tokens"], drafter["selector_top_k"]
    inputs = {value.name: value for value in session.get_inputs()}
    cache_shape = inputs[names["past_key_names"] % 0].shape
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
        anchor = int(np.argmax(logits[-1]))
        feeds = make_drafter_feeds(session, config, context, anchor, past_length, feeds)
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
        past_length += len(context)
        token_ids.append(anchor)


def test_target_verification_matches_same_shape_eager(drafter_session, tmp_path):
    """Compare the target graph contract, not a replay of unobservable Engine feeds."""
    drafter, target, tokenizer, config = drafter_session
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, return_dict=True
        )["input_ids"]
        for prompt, _ in PROMPTS[:2]
    ]
    lengths = [len(prompt) for prompt in prompts]
    prefill_feeds = make_packed_target_inputs(target, config, prompts, [0, 0])
    prefill = dict(zip((value.name for value in target.get_outputs()), target.run(None, prefill_feeds), strict=True))
    state = {input_name: prefill[output_name] for input_name, output_name in target_state_pairs(config)}
    for kind in ("past_key_values.", ".conv", ".recurrent"):
        assert any(np.any(value) for name, value in state.items() if kind in name), f"Empty prefill state: {kind}"
    output_names = config["model"]["decoder"]["outputs"]
    prefill_logits, auxiliary = prefill[output_names["logits"]], prefill[output_names["aux_hidden_states"]]
    queries, begin = [], 0
    draft_outputs = config["model"]["dflash2"]["outputs"]
    for length in lengths:
        anchor = int(prefill_logits[begin + length - 1].argmax())
        feeds = make_drafter_feeds(drafter, config, auxiliary[begin : begin + length], anchor)
        candidates, scores = drafter.run([draft_outputs["candidate_ids"], draft_outputs["scores"]], feeds)
        # The greedy lattice walk matches Dflash2Drafter::Propose, truncated to three drafts.
        proposal, previous = [], 0
        for step in range(3):
            previous = int(scores[0, step, previous].argmax())
            proposal.append(int(candidates[0, step, previous]))
        queries.append([anchor, *proposal])
        begin += length
    eager_feeds = make_packed_target_inputs(target, config, queries, lengths, state)
    verify_feeds = make_packed_target_inputs(target, config, queries, lengths, state, capture=True)
    names = config["model"]["decoder"]["inputs"]
    controls = {names["state_update_capture_count"], names["state_update_active"]}
    for name in eager_feeds:
        assert eager_feeds[name].shape == verify_feeds[name].shape
        if name not in controls:
            np.testing.assert_array_equal(eager_feeds[name], verify_feeds[name], err_msg=name)
    eager = target.run([output_names["logits"]], eager_feeds)[0]
    verification = target.run([output_names["logits"]], verify_feeds)[0]
    for name, value in state.items():
        np.testing.assert_array_equal(eager_feeds[name], value, err_msg=name)
        np.testing.assert_array_equal(verify_feeds[name], value, err_msg=name)
    report = {
        "batch_size": 2,
        "request_order": [0, 1],
        "prompt_lengths": lengths,
        "query_lengths": [len(query) for query in queries],
        "input_ids": eager_feeds[names["input_ids"]].tolist(),
        "position_ids": eager_feeds[names["position_ids"]].tolist(),
        "capture_count": verify_feeds[names["state_update_capture_count"]].tolist(),
        "same_shape": logit_comparison(eager, verification),
    }
    # Only query width changes here; the incoming state, row order and causal anchor match.
    narrow_feeds = make_packed_target_inputs(target, config, [[query[0]] for query in queries], lengths, state)
    narrow = target.run([output_names["logits"]], narrow_feeds)[0]
    report["query_width_probe"] = logit_comparison(narrow, eager[::4])
    np.savez_compressed(tmp_path / "target-logits.npz", eager=eager, verification=verification, narrow=narrow)
    (tmp_path / "target-logit-comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Target-logit comparison: {json.dumps(report, sort_keys=True)}")
    np.testing.assert_allclose(eager, verification, rtol=1e-3, atol=2e-3, err_msg=json.dumps(report))
    np.testing.assert_array_equal(eager.argmax(axis=-1), verification.argmax(axis=-1), err_msg=json.dumps(report))


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


def test_two_request_partial_replay_activity_smoke(pytestconfig):
    model_dir = pytestconfig.getoption("--dflash2-webgpu-model")
    if model_dir is None:
        pytest.skip("Pass --dflash2-webgpu-model to test an exported WebGPU DFlash2 model")
    if "WebGpuExecutionProvider" not in ort.get_available_providers():
        assert register_webgpu_plugin(), "WebGPU is required for this model test"
    result = run_multi_request_activity_smoke(Path(model_dir))
    print(
        f"Two-request replay activity smoke (cross-shape equality is diagnostic): {json.dumps(result, sort_keys=True)}"
    )


@pytest.mark.parametrize(
    "field",
    [
        "target_forward_passes",
        "completed_rounds",
        "partial_accept_rounds",
        "draft_tokens_proposed",
        "draft_tokens_accepted",
        "full_accept_rounds",
        "zero_accept_rounds",
        "acceptance_length_histogram",
        "emitted",
    ],
)
def test_validation_requires_two_distinct_partial_prefixes(field):
    before = dict.fromkeys(
        (
            "target_forward_passes",
            "completed_rounds",
            "partial_accept_rounds",
            "full_accept_rounds",
            "zero_accept_rounds",
            "draft_tokens_proposed",
            "draft_tokens_accepted",
        ),
        0,
    )
    before["acceptance_length_histogram"] = [0, 0, 0, 0]
    after = {
        **before,
        "target_forward_passes": 1,
        "completed_rounds": 2,
        "partial_accept_rounds": 2,
        "draft_tokens_proposed": 6,
        "draft_tokens_accepted": 3,
        "acceptance_length_histogram": [0, 1, 1, 0],
    }
    validate_partial_replay_round(before, after, [2, 3])
    emitted = [3, 2] if field == "emitted" else [2, 3]
    if field != "emitted":
        after[field] = [0, 0, 2, 0] if field == "acceptance_length_histogram" else after[field] + 1
    with pytest.raises(RuntimeError):
        validate_partial_replay_round(before, after, emitted)

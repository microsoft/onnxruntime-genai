# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Executable Engine coverage for continuation across window-ring wraps."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import onnx
import onnxruntime_genai as og
import pytest

_MODEL_SUBPATH = Path("engine") / "synthetic-windowed-multiwrap"

_VOCAB_SIZE = 32
_BLOCK_SIZE = 2
_WINDOW_SIZE = 3
_CHUNK_SIZE = 2
_RING_BLOCKS = 2
_RING_PERIOD = _RING_BLOCKS * _BLOCK_SIZE
_NUM_FULL_BLOCKS = 8
_MAX_LENGTH = 16
_EOS_TOKEN_ID = 1
_INVARIANT_FAILURE_TOKEN_ID = 31

_INITIAL_PROMPT = [4, 6]
_EXPECTED_FIRST_TURN = [12]
_CONTINUATION = [8, 3, 10, 5, 12, 7, 14, 9, 16]
_EXPECTED_SECOND_TURN = [28, 17]

_DEVICES = ["cpu"] + (["cuda"] if og.is_cuda_available() else [])


def _fixture_path(test_data_path) -> Path:
    if not test_data_path:
        pytest.skip("--test_models is required for the synthetic Engine fixture")
    path = Path(test_data_path) / _MODEL_SUBPATH
    if not path.exists():
        pytest.fail(f"synthetic Engine fixture is missing: {path}")
    return path


def _cache_key(token, position):
    return token * 7 + position * 3 + 1


def _cache_value(token, position):
    return token * 5 + position * 2 + 2


def _normal_token(sequence, position):
    """Mirror the graph's cache-read score for a non-EOS position."""
    first_key = _cache_key(sequence[0], 0)
    previous_key = _cache_key(sequence[position - 1], position - 1)
    current_key = _cache_key(sequence[position], position)
    previous_value = _cache_value(sequence[position - 1], position - 1)
    current_value = _cache_value(sequence[position], position)
    score = first_key + current_key + previous_key + current_key + previous_value + current_value
    return score % 28 + 2


def _wrap_count(start_position, token_count):
    positions = range(start_position, start_position + token_count)
    slots = [position % _RING_PERIOD for position in positions]
    return sum(current < previous for previous, current in pairwise(slots))


def _new_request(engine, model, prompt):
    params = og.GeneratorParams(model)
    params.set_search_options(do_sample=False, max_length=_MAX_LENGTH)
    request = og.Request(params)
    request.add_tokens(np.asarray(prompt, dtype=np.int32))
    engine.add_request(request)
    return request


def _run_turn(engine, request):
    tokens = []
    steps = 0
    while not request.is_turn_complete():
        ready = engine.step()
        steps += 1
        assert steps < 100, "synthetic turn did not reach its absolute-position EOS"
        if ready is None:
            # Partial prefill chunks commit cache progress without sampling.
            continue
        assert ready is request
        while ready.has_unseen_tokens():
            tokens.append(ready.get_unseen_token())
    return tokens


def test_windowed_multiwrap_fixture_schema(test_data_path):
    model_path = _fixture_path(test_data_path)
    config = json.loads((model_path / "genai_config.json").read_text(encoding="utf-8"))
    decoder = config["model"]["decoder"]
    dynamic = config["engine"]["dynamic_batching"]

    assert decoder["num_hidden_layers"] == 2
    assert decoder["sliding_window"] == {
        "window_size": _WINDOW_SIZE,
        "slide_key_value_cache": False,
        "slide_inputs": False,
        "layers": [1],
    }
    assert decoder["inputs"]["block_table_windowed"] == "block_table_windowed"
    assert config["model"]["vocab_size"] == _VOCAB_SIZE
    assert config["model"]["eos_token_id"] == _EOS_TOKEN_ID
    assert config["search"]["max_length"] == _MAX_LENGTH
    assert config["search"]["chunk_size"] == _CHUNK_SIZE
    assert dynamic["block_size"] == _BLOCK_SIZE
    assert dynamic["num_blocks"] == _NUM_FULL_BLOCKS
    assert dynamic["max_batch_size"] == 1

    assert (_CHUNK_SIZE + _WINDOW_SIZE - 1 + _BLOCK_SIZE - 1) // _BLOCK_SIZE == _RING_BLOCKS
    continuation_start = len(_INITIAL_PROMPT) + len(_EXPECTED_FIRST_TURN)
    assert len(_CONTINUATION) > 2 * _RING_PERIOD
    assert _wrap_count(continuation_start, len(_CONTINUATION)) == 2

    graph_model = onnx.load(model_path / "decoder.onnx", load_external_data=False)
    onnx.checker.check_model(graph_model)
    metadata = {entry.key: entry.value for entry in graph_model.metadata_props}
    assert metadata["fixture"] == "engine-windowed-multiwrap-continuation"

    input_shapes = {
        value.name: [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]
        for value in graph_model.graph.input
        if value.name.startswith("past_key_values")
    }
    assert input_shapes["past_key_values.0.key"] == [_NUM_FULL_BLOCKS, 2, 1, 1]
    assert input_shapes["past_key_values.0.value"] == [_NUM_FULL_BLOCKS, 2, 1, 1]
    assert input_shapes["past_key_values.1.key"] == [2, 2, 1, 1]
    assert input_shapes["past_key_values.1.value"] == [2, 2, 1, 1]

    node_names = {node.name for node in graph_model.graph.node}
    assert {
        "read_full_current_key",
        "read_full_previous_key",
        "read_window_previous_key",
        "read_window_current_value",
        "guard_repeated_window_block",
        "guard_stable_window_owner",
    } <= node_names


@pytest.mark.parametrize("device", _DEVICES)
def test_continuation_crosses_two_window_ring_wraps_and_matches_clean_replay(test_data_path, device):
    model_path = _fixture_path(test_data_path)
    config = og.Config(str(model_path))
    config.clear_providers()
    if device == "cuda":
        config.append_provider("cuda")
    model = og.Model(config)

    # The first normal result is followed by EOS at absolute position 2, well
    # below the session max. EOS is not part of the retained logical sequence.
    assert _normal_token(_INITIAL_PROMPT, 1) == _EXPECTED_FIRST_TURN[0]
    engine = og.Engine(model)
    request = _new_request(engine, model, _INITIAL_PROMPT)
    first_turn = _run_turn(engine, request)

    assert first_turn == _EXPECTED_FIRST_TURN
    assert request.is_turn_complete()
    first_turn_length = len(_INITIAL_PROMPT) + len(first_turn)
    assert first_turn_length == 3
    assert first_turn_length < _MAX_LENGTH

    # Positions 3..11 traverse the four-slot ring twice. The graph returns 31
    # if the repeated table, retained ring ownership, or values read through
    # the full/windowed caches disagree. Tokens at positions 11 and 12 check
    # both columns of the two-block ring before EOS at position 13.
    request.continue_with(np.asarray(_CONTINUATION, dtype=np.int32))
    assert not request.is_turn_complete()
    second_turn = _run_turn(engine, request)

    assert second_turn == _EXPECTED_SECOND_TURN
    assert _INVARIANT_FAILURE_TOKEN_ID not in second_turn
    assert request.is_turn_complete()
    retained_length = first_turn_length + len(_CONTINUATION) + len(second_turn)
    assert retained_length == 14
    assert retained_length < _MAX_LENGTH
    engine.remove_request(request)

    # Recompute the same logical context in a fresh cache. Exact parity proves
    # that continuation's retained full cache and twice-wrapped window cache
    # agree with a clean chunked replay at absolute positions 0..13.
    replay_prompt = _INITIAL_PROMPT + first_turn + _CONTINUATION
    assert _normal_token(replay_prompt, len(replay_prompt) - 1) == _EXPECTED_SECOND_TURN[0]
    replay_after_first_output = replay_prompt + _EXPECTED_SECOND_TURN[:1]
    assert _normal_token(replay_after_first_output, len(replay_after_first_output) - 1) == _EXPECTED_SECOND_TURN[1]
    replay_engine = og.Engine(model)
    replay_request = _new_request(replay_engine, model, replay_prompt)
    replay_turn = _run_turn(replay_engine, replay_request)

    assert replay_turn == second_turn == _EXPECTED_SECOND_TURN
    assert _INVARIANT_FAILURE_TOKEN_ID not in replay_turn
    replay_engine.remove_request(replay_request)

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark_qwen_context as benchmark
import run_context_sweep as sweep
from benchmark_qwen_context import generation_limit, make_review_prompt, request_stats_delta, token_throughput
from run_context_sweep import file_hash, save_table


class CharacterTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt):
        assert add_generation_prompt
        messages = json.loads(messages)
        assert [message["role"] for message in messages] == ["system", "user"]
        return messages[0]["content"] + messages[1]["content"] + "\nassistant\n"

    def encode(self, text):
        return list(text.encode("ascii"))


@pytest.mark.parametrize("length", [8192, 16384, 32768, 65536, 131072])
def test_prompt_preserves_generation_suffix_and_length(length):
    tokenizer = CharacterTokenizer()
    tokens = make_review_prompt(tokenizer, length)
    assert len(tokens) == length
    assert bytes(tokens.tolist()).endswith(b"\nassistant\n")
    assert (tokens == make_review_prompt(tokenizer, length)).all()


def test_streaming_hash(tmp_path):
    path = tmp_path / "binary"
    data = b"x" * (1024 * 1024 + 7)
    path.write_bytes(data)
    assert file_hash(path) == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("prompt", [2048, 8192, 16384, 32768, 65536, 131072])
def test_natural_generation_uses_remaining_engine_capacity(prompt):
    assert generation_limit(prompt, 0, 262144, 150000) == 150000 - prompt
    assert generation_limit(prompt, 512, 262144, 150000) == 512
    assert generation_limit(prompt, 2048, 262144, 150000) == 2048


def test_generation_is_bounded_by_model_and_rejects_invalid_lengths():
    assert generation_limit(8192, 0, 10240, 12000) == 2048
    for requested, context, capacity in [(-1, 10240, 12000), (4096, 10240, 12000),
                                         (0, 8192, 12000), (0, 10240, 8192)]:
        with pytest.raises(ValueError):
            generation_limit(8192, requested, context, capacity)


def test_throughput_uses_actual_variable_output_counts():
    assert token_throughput(2048, 10, 12) == {"decode_tps": 204.7, "end_to_end_tps": 2048 / 12}
    assert token_throughput(512, 5, 6) == {"decode_tps": 511 / 5, "end_to_end_tps": 512 / 6}
    assert token_throughput(1, 0, 1) == {"decode_tps": None, "end_to_end_tps": 1}
    assert token_throughput(3072, 15, 20, requests=3) == {
        "decode_tps": 3069 / 15, "end_to_end_tps": 3072 / 20}
    with pytest.raises(ValueError):
        token_throughput(0, 0, 1)


def test_table_labels_natural_and_exact_generation(tmp_path):
    save_table([], tmp_path)
    assert "512-token requests" in (tmp_path / "TABLE.md").read_text()
    save_table([], tmp_path, output_tokens=0)
    assert "until EOS or remaining capacity" in (tmp_path / "TABLE.md").read_text()
    save_table([], tmp_path, output_tokens=2048)
    assert "2048-token requests" in (tmp_path / "TABLE.md").read_text()


@pytest.mark.parametrize("enabled", [False, True])
def test_engine_optimization_options_are_explicit_and_persisted(tmp_path, monkeypatch, enabled):
    model = tmp_path / "model"
    model.mkdir()
    (model / "genai_config.json").write_text("{}")
    destination = tmp_path / "results"
    flags = {
        "--sparse-mtp-grouped-heads": "ORT_SPARSE_MTP_GROUPED_HEADS",
        "--sparse-prefill-async-kv": "ORT_SPARSE_PREFILL_ASYNC_KV",
        "--hyperconnection-vectorize": "ORT_HYPER_CONNECTION_VECTORIZE",
        "--nvfp4-fused-routing": "ORT_NVFP4_DECODE_FUSED_ROUTING",
        "--nvfp4-single-pass-finalize": "ORT_NVFP4_DECODE_SINGLE_PASS_FINALIZE",
    }
    for name in flags.values():
        monkeypatch.setenv(name, "1")
    command = ["run_context_sweep.py", "--model", str(model),
               "--output-dir", str(destination), "--gpu", "6"]
    if enabled:
        command.extend(flags)
    monkeypatch.setattr(sys, "argv", command)

    def output(command):
        if command[0] == "nvidia-smi":
            raise RuntimeError("stop before GPU access")
        return ""

    monkeypatch.setattr(sweep, "output", output)
    with pytest.raises(RuntimeError, match="stop before GPU access"):
        sweep.main()
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["output_tokens"] == 512
    assert all(manifest["environment"][name] == ("1" if enabled else "0") for name in flags.values())


@pytest.mark.parametrize("requested,actual", [(0, 600), (1024, 1024), (512, 512), (None, 512)])
def test_request_loop_accepts_eos_and_exact_outputs_beyond_old_cap(tmp_path, monkeypatch, requested, actual):
    model = tmp_path / "model"
    model.mkdir()
    (model / "genai_config.json").write_text(json.dumps({"model": {"context_length": 8000}}))
    destination = tmp_path / "result.json"
    engines = []

    class Options:
        def set_max_session_tokens(self, value):
            self.session_tokens = value

    class Turn:
        def __init__(self, request):
            self.request = request

        def set_do_sample(self, value):
            assert value is False

        def set_min_generated_tokens(self, value):
            self.minimum = value

        def set_max_generated_tokens(self, value):
            self.maximum = value

    class Engine:
        def __init__(self, model):
            self.count = 0
            engines.append(self)

        def get_speculative_stats(self):
            return {}

        def get_capabilities(self):
            return SimpleNamespace(max_request_length=4000)

        def create_event_buffer(self, size):
            return []

        def create_request(self, options):
            self.options = options

            def begin_turn(tokens, turn):
                self.turn = turn
                assert len(tokens) == 8

            return SimpleNamespace(begin_turn=begin_turn, close=lambda: None)

        def has_pending_requests(self):
            return self.count < actual

        def run(self, buffer):
            self.count += 1
            return [SimpleNamespace(flags=4 | (8 if self.count == actual else 0), token=self.count)]

    monkeypatch.setattr(benchmark, "og", SimpleNamespace(
        Config=lambda path: SimpleNamespace(overlay=lambda value: None),
        Model=lambda config: object(),
        Tokenizer=lambda model: SimpleNamespace(decode=lambda tokens: "synthetic"),
        Engine=Engine, RequestOptions=Options, TurnOptions=Turn,
        EngineEventFlags=SimpleNamespace(FAILED=1, CAPACITY_BLOCKED=2, TOKEN=4, TURN_FINISHED=8)))
    monkeypatch.setattr(benchmark, "make_review_prompt", lambda tokenizer, length: np.arange(8, dtype=np.int32))
    command = [
        "benchmark_qwen_context.py", "--mode", "baseline", "--model", str(model),
        "--output", str(destination), "--prompt-length", "8", "--repetitions", "1"]
    if requested is not None:
        command.extend(["--output-tokens", str(requested)])
    monkeypatch.setattr(sys, "argv", command)
    benchmark.main()
    requested = 512 if requested is None else requested
    result = json.loads(destination.read_text())
    assert [run["output_tokens"] for run in result["runs"]] == [actual, actual]
    assert result["output_policy"] == ("exact" if requested else "eos_or_capacity")
    for engine in engines:
        assert engine.turn.minimum == requested
        assert engine.turn.maximum == (requested or 3992)
        assert engine.options.session_tokens == 8 + engine.turn.maximum
    run = result["runs"][1]
    assert result["summary"]["decode_tps"] == (actual - 1) / run["decode_seconds"]
    assert result["summary"]["end_to_end_tps"] == actual / run["end_to_end_seconds"]


def test_shared_engine_statistics_exclude_previous_requests():
    before = {
        "completed_rounds": 10, "draft_tokens_accepted": 20,
        "total_draft_ms": 4.0, "acceptance_rate": 0.8,
        "effective_k": 3, "formula_supported": True,
        "acceptance_length_histogram": [2, 3],
    }
    after = {
        "completed_rounds": 14, "draft_tokens_accepted": 27,
        "total_draft_ms": 6.5, "acceptance_rate": 0.75,
        "effective_k": 5, "formula_supported": True,
        "acceptance_length_histogram": [3, 5, 1],
    }
    assert request_stats_delta(before, after) == {
        "completed_rounds": 4, "draft_tokens_accepted": 7,
        "total_draft_ms": 2.5, "effective_k": 5,
        "formula_supported": True, "acceptance_length_histogram": [1, 2, 1],
    }
    assert before["acceptance_length_histogram"] == [2, 3]


def test_cached_table_labels_excluded_cold_work(tmp_path):
    save_table([], tmp_path, prefix_cache=True)
    table = (tmp_path / "TABLE.md").read_text()
    assert "cold warmup excluded" in table
    assert "no reusable checkpoint" in table


def test_table_reports_failures(tmp_path):
    save_table([
        {"status": "failed", "requested_prompt": 131072, "mtp": 7, "peak_mib": 100000},
    ], tmp_path)
    table = (tmp_path / "TABLE.md").read_text()
    assert "| 131072 | - | 7 |" in table
    assert "failed; see log" in table
    assert "| completed |" not in table


def test_table_formats_completed_measurements(tmp_path):
    save_table([{
        "status": "completed", "requested_prompt": 8192, "prompt_length": 8190,
        "mtp": 3, "ttft_ms": 1226.31, "decode_tps": 129.52, "e2e_tps": 99.00,
        "mtp_acceptance": 71.27, "first_draft_acceptance": 75.00,
        "proposal_yield": 53.75, "within_target": False, "peak_mib": 94340,
    }], tmp_path)
    assert (
        "| 8192 | 8190 | 3 | 1226.31 | 129.52 | 99.00 | 71.27% | 75.00% "
        "| 53.75% | no | 94340 | completed |"
    ) in (tmp_path / "TABLE.md").read_text()

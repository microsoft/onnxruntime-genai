# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
from types import SimpleNamespace

import pytest
from benchmark_coding import main, parse_args, qualify, read_prompt, run_request, stats_delta


@pytest.mark.parametrize("value", [[], {}, [True], [-1], [1.5], [2147483648]])
def test_invalid_prompt(tmp_path, value):
    path = tmp_path / "prompt.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        read_prompt(path)


def test_valid_prompt(tmp_path):
    path = tmp_path / "prompt.json"
    path.write_text("[0,1,2147483647]")
    assert read_prompt(path) == [0, 1, 2147483647]


@pytest.mark.parametrize("count", ["0", "-1", "2049"])
def test_output_cap_enforced(tmp_path, count):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--model",
                str(tmp_path / "model"),
                "--prompt-ids",
                "ids.json",
                "--runtime-artifact",
                "core",
                "--num-blocks",
                "8",
                "--output",
                str(tmp_path / "out.json"),
                "--new-tokens",
                count,
            ]
        )


def test_counter_delta_excludes_gauges_and_ratios():
    before = {
        "draft_tokens_evaluated": 10,
        "draft_tokens_accepted": 5,
        "effective_k": 7,
        "acceptance_rate": 0.5,
        "total_target_ms": 1.0,
        "active_rounds": 0,
    }
    after = {
        **before,
        "draft_tokens_evaluated": 14,
        "draft_tokens_accepted": 8,
        "effective_k": 3,
        "total_target_ms": 2.5,
    }
    assert stats_delta(before, after) == {
        "draft_tokens_evaluated": 4,
        "draft_tokens_accepted": 3,
        "acceptance_rate": 0.75,
        "total_target_ms": 1.5,
    }


class FakeOptions:
    def set_max_session_tokens(self, count):
        self.limit = count

    def set_do_sample(self, value):
        self.sample = value

    def set_max_generated_tokens(self, count):
        self.limit = count


class FakeRequest:
    def __init__(self):
        self.closed = False

    def begin_turn(self, prompt, turn):
        assert not turn.sample
        assert prompt == [1, 2]

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self, flags=(1, 3), fail=False):
        self.request = FakeRequest()
        self.flags = list(flags)
        self.fail = fail

    def create_request(self, *, options):
        assert options.limit >= 3
        return self.request

    def get_speculative_stats(self):
        return {"draft_tokens_evaluated": 0}

    def create_event_buffer(self, capacity):
        assert capacity == 1
        return []

    def has_pending_requests(self):
        return bool(self.flags)

    def run(self, events):
        if self.fail:
            raise RuntimeError("injected execution failure")
        return [
            SimpleNamespace(
                request=self.request, token=42, flags=self.flags.pop(0), finish_reason="length", error_code=17
            )
        ]


def fake_runtime():
    return SimpleNamespace(
        RequestOptions=FakeOptions,
        TurnOptions=lambda request: FakeOptions(),
        EngineEventFlags=SimpleNamespace(TOKEN=1, TURN_FINISHED=2, FAILED=4, CAPACITY_BLOCKED=8, RETRYABLE=16),
    )


def test_event_pacing_and_cleanup():
    engine = FakeEngine()
    ticks = iter([0.0, 1.0, 1.5, 2.0])
    result = run_request(fake_runtime(), engine, [1, 2], 2, clock=lambda: next(ticks))
    assert engine.request.closed
    assert result["tokens"] == [42, 42]
    assert result["ttft_seconds"] == 1.0
    assert result["decode_tokens_per_second"] == 2.0


@pytest.mark.parametrize(
    "engine, cap, message",
    [
        (FakeEngine(fail=True), 2, "execution failure"),
        (FakeEngine(flags=(4,)), 2, "error_code=17"),
        (FakeEngine(flags=(1,)), 2, "turn-finished"),
        (FakeEngine(), 1, "exceeded"),
    ],
)
def test_failed_requests_close(engine, cap, message):
    with pytest.raises(RuntimeError, match=message):
        run_request(fake_runtime(), engine, [1, 2], cap)
    assert engine.request.closed


@pytest.mark.parametrize("requestless", [False, True])
@pytest.mark.parametrize(
    "flags, message",
    [
        (4, "Engine failed with error_code=17"),
        (8, "Engine capacity-blocked with error_code=17"),
        (16, "Engine retryable abort with error_code=17"),
        (4 | 16, "Engine failed with error_code=17"),
    ],
)
def test_native_error_category_precedes_request_ownership(requestless, flags, message):
    class ErrorEngine(FakeEngine):
        def run(self, events):
            batch = super().run(events)
            if requestless:
                batch[0].request = None
            return batch

    engine = ErrorEngine(flags=(flags,))
    with pytest.raises(RuntimeError, match=message):
        run_request(fake_runtime(), engine, [1, 2], 2)
    assert engine.request.closed


def test_foreign_token_event_still_rejected():
    class ForeignEngine(FakeEngine):
        def run(self, events):
            batch = super().run(events)
            batch[0].request = FakeRequest()
            return batch

    engine = ForeignEngine()
    with pytest.raises(RuntimeError, match="another request"):
        run_request(fake_runtime(), engine, [1, 2], 2)
    assert engine.request.closed


def sample_result(tokens):
    return {
        "schema_version": 1,
        "prompt_sha256": "abc",
        "prompt_tokens": 2,
        "output_token_cap": 2,
        "first_request": {"tokens": tokens},
        "warm_process_requests": [{"tokens": tokens}],
    }


def test_qualification_checks_all_outputs():
    reference = sample_result([1, 2])
    result = sample_result([1, 2])
    assert qualify(result, reference)["token_ids_equal"]
    result["warm_process_requests"][0]["tokens"] = [1, 3]
    assert not qualify(result, reference)["token_ids_equal"]
    result["prompt_sha256"] = "other"
    with pytest.raises(ValueError, match="prompt_sha256"):
        qualify(result, reference)


def test_complete_cli_uses_fresh_requests_and_read_only_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_path = model_dir / "model.onnx"
    model_path.write_bytes(b"fixture graph, not executable")
    config_path = model_dir / "genai_config.json"
    config_path.write_text('{"engine":{"dynamic_batching":{}}}')
    prompt = tmp_path / "ids.json"
    prompt.write_text("[1,2]")
    library = tmp_path / "core.so"
    library.write_bytes(b"fixture runtime, not loaded")
    output = tmp_path / "result.json"
    overlays, requests = [], []

    class Config:
        def __init__(self, path):
            assert path == str(model_dir)

        def clear_providers(self):
            pass

        def append_provider(self, provider):
            assert provider == "cpu"

        def overlay(self, text):
            overlays.append(json.loads(text))

    class Engine(FakeEngine):
        def create_request(self, *, options):
            self.request = FakeRequest()
            self.flags = [1, 3]
            requests.append(self.request)
            return super().create_request(options=options)

    runtime = fake_runtime()
    runtime.Config = Config
    runtime.Model = lambda config: object()
    runtime.Engine = lambda model: Engine()
    runtime.__file__ = str(library)
    monkeypatch.setattr("benchmark_coding.importlib.import_module", lambda name: runtime)
    main(
        [
            "--model",
            str(model_dir),
            "--prompt-ids",
            str(prompt),
            "--runtime-artifact",
            str(library),
            "--num-blocks",
            "8",
            "--output",
            str(output),
            "--new-tokens",
            "2",
            "--repeat",
            "2",
            "--warmup",
            "1",
        ]
    )
    result = json.loads(output.read_text())
    assert overlays[0]["engine"]["dynamic_batching"]["prefix_caching"] is False
    assert result["within_process_token_ids_equal"]
    assert result["model_files"]["model.onnx"]["bytes"] == len(model_path.read_bytes())
    assert len(result["warm_process_requests"]) == 2
    assert len(requests) == 4
    assert len({id(request) for request in requests}) == 4
    assert all(request.closed for request in requests)
    assert set(model_dir.iterdir()) == {model_path, config_path}

    # Profile selection happens at Model construction, after Config.overlay.
    # Reject it instead of claiming caching is disabled when a profile re-enables it.
    config_path.write_text('{"runtime_profiles":[{"id":"override"}]}')
    with pytest.raises(ValueError, match="profile-free"):
        main(
            [
                "--model",
                str(model_dir),
                "--prompt-ids",
                str(prompt),
                "--runtime-artifact",
                str(library),
                "--num-blocks",
                "8",
                "--output",
                str(tmp_path / "profile-result.json"),
            ]
        )

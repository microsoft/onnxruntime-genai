# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CPU-only simulated-clock regressions through the production benchmark loop."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


@dataclass(frozen=True)
class Request:
    token_costs_ms: tuple[float, ...]
    loop_gap_ms: float = 0


class Clock:
    def __init__(self):
        self.nanoseconds = 0

    def advance(self, milliseconds):
        self.nanoseconds += round(milliseconds * 1_000_000)

    def __call__(self):
        return self.nanoseconds / 1_000_000_000


@pytest.fixture
def benchmark(monkeypatch):
    directory = Path(__file__).resolve().parent
    monkeypatch.syspath_prepend(str(directory))
    native = ModuleType("onnxruntime_genai")
    # Neither import-time GPU discovery nor the unused memory monitor may touch hardware.
    with monkeypatch.context() as imports, patch.object(subprocess, "run", side_effect=FileNotFoundError):
        imports.setitem(sys.modules, "onnxruntime_genai", native)
        imports.setitem(sys.modules, "psutil", ModuleType("psutil"))
        spec = importlib.util.spec_from_file_location("_benchmark_e2e_decode_test", directory / "benchmark_e2e.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    monkeypatch.setattr(module, "tqdm", lambda values: values)
    monkeypatch.setattr(module.np.random, "randint", lambda high, size: np.ones(size, dtype=np.int32))
    return module


class Simulation:
    def __init__(self, module, requests, batch_size):
        self.clock = Clock()
        self.requests = list(requests)
        self.batch_size = batch_size
        self.started = []
        self.generators = []
        self.rewinds = 0
        simulation = self

        class Config:
            def __init__(self, folder):
                pass

            def overlay(self, value):
                pass

            def clear_providers(self):
                pass

        class Tokenizer:
            def __init__(self, model):
                simulation.clock.advance(19)

            def encode(self, text):
                simulation.clock.advance(11)
                return [1, 2, 3, 4]

            def decode(self, tokens):
                return "prompt"

        class GeneratorParams:
            def __init__(self, model):
                pass

            def set_search_options(self, **options):
                assert options["batch_size"] == batch_size

        class Generator:
            def __init__(self, model, params):
                simulation.clock.advance(300)
                simulation.generators.append(self)

            def rewind_to(self, length):
                assert length == 0
                simulation.rewinds += 1

            def append_tokens(self, tokens):
                assert np.asarray(tokens).shape == (batch_size, 4)
                self.request = simulation.requests[len(simulation.started)]
                simulation.started.append(self)
                self.emitted = 0
                self.observed = set()
                simulation.clock.advance(13)

            def token_count(self):
                # Account for Python/loop work only once between successive completions.
                if 0 < self.emitted < len(self.request.token_costs_ms) and self.emitted not in self.observed:
                    simulation.clock.advance(self.request.loop_gap_ms)
                    self.observed.add(self.emitted)
                return 4 + self.emitted

            def is_done(self):
                return self.emitted == len(self.request.token_costs_ms)

            def generate_next_token(self):
                assert not self.is_done()
                simulation.clock.advance(self.request.token_costs_ms[self.emitted])
                self.emitted += 1

        def model(config):
            simulation.clock.advance(23)
            return SimpleNamespace(type="simulated")

        module.og.Config = Config
        module.og.Model = model
        module.og.Tokenizer = Tokenizer
        module.og.GeneratorParams = GeneratorParams
        module.og.Generator = Generator

    def run(self, module, monkeypatch, *, aggregation="mean", reuse=False, warmup=0, generation_length=16):
        monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=self.clock, time=self.clock))
        args = SimpleNamespace(
            repetitions=len(self.requests) - warmup,
            verbose=False,
            input_folder="unused-simulated-model",
            execution_provider="cpu",
            chat_template="{input}",
            use_random_tokens=True,
            use_prompt_set=False,
            top_k=1,
            top_p=1.0,
            reuse_generator=reuse,
            warmup=warmup,
            print_model_output=False,
            aggregation=aggregation,
            print_memory_usage=False,
        )
        return module.run_benchmark(args, self.batch_size, 4, generation_length, 4 + generation_length)


@pytest.mark.parametrize("aggregation", ["mean", "median"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("reuse", [False, True])
def test_interleaved_refills_include_buffered_calls_and_continuous_loop_time(
    benchmark, monkeypatch, aggregation, batch_size, reuse
):
    # 16 emitted tokens: 42 ms in post-first calls + 15 * 0.2 ms between calls = 45 ms.
    # The 73 ms first token and all construction/prefill costs are outside decode.
    request = Request((73, *([1, 1, 1, 10] * 3), 1, 1, 1), loop_gap_ms=0.2)
    simulation = Simulation(benchmark, [request], batch_size)
    metrics = simulation.run(benchmark, monkeypatch, aggregation=aggregation, reuse=reuse)

    assert metrics[14:16] == pytest.approx([batch_size * 333.3333333333333, 3])
    assert simulation.generators[0].emitted == 16


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("aggregation", ["mean", "median"])
def test_ordinary_nonbuffered_decode(benchmark, monkeypatch, batch_size, aggregation):
    simulation = Simulation(benchmark, [Request((73, 4, 4, 4, 4))], batch_size)
    metrics = simulation.run(benchmark, monkeypatch, aggregation=aggregation, generation_length=5)

    assert metrics[14:16] == pytest.approx([batch_size * 250, 4])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_early_eos_uses_actual_post_first_count(benchmark, monkeypatch, batch_size):
    simulation = Simulation(benchmark, [Request((73, 2, 6, 1), loop_gap_ms=0.2)], batch_size)
    metrics = simulation.run(benchmark, monkeypatch, generation_length=16)

    assert simulation.generators[0].emitted == 4
    assert metrics[14:16] == pytest.approx([batch_size * 312.5, 3.2])


@pytest.mark.parametrize("aggregation", ["mean", "median"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("reuse", [False, True])
def test_requests_are_aggregated_after_measuring_each_request(benchmark, monkeypatch, aggregation, batch_size, reuse):
    # Warmup is deliberately much slower and must never enter measured statistics.
    requests = [
        Request((1000, 1000, 1000)),
        Request((73, 2, 4)),
        Request((73, 1, 2, 3, 10)),
        Request((73, 8)),
    ]
    simulation = Simulation(benchmark, requests, batch_size)
    metrics = simulation.run(benchmark, monkeypatch, aggregation=aggregation, reuse=reuse, warmup=1)

    # Request throughputs: 2/.006, 4/.016, 1/.008. Latencies: 3, 4, 8 ms.
    if aggregation == "mean":
        expected = [batch_size * (333.3333333333333 + 250 + 125) / 3, 5]
    else:
        expected = [batch_size * 250, 4]
    assert len(simulation.started) == 4
    assert len(simulation.generators) == (1 if reuse else 4)
    assert simulation.rewinds == (4 if reuse else 0)
    assert metrics[14:16] == pytest.approx(expected)


@pytest.mark.parametrize("generation_length", [1, 16], ids=["one-token-request", "early-eos-first-token"])
@pytest.mark.parametrize("aggregation", ["mean", "median"])
def test_one_emitted_token_has_explicitly_unavailable_decode_metrics(
    benchmark, monkeypatch, capsys, generation_length, aggregation
):
    simulation = Simulation(benchmark, [Request((73,))], 1)
    metrics = simulation.run(benchmark, monkeypatch, aggregation=aggregation, generation_length=generation_length)

    assert metrics[14:16] == [None, None]
    output = capsys.readouterr().out
    assert "Token Generation Latency (per token): N/A" in output
    assert "Token Generation Throughput (per token): N/A" in output


@pytest.mark.parametrize("aggregation", ["mean", "median"])
def test_unavailable_request_is_not_a_zero_in_aggregation(benchmark, monkeypatch, aggregation):
    simulation = Simulation(benchmark, [Request((73,)), Request((73, 4, 4))], 1)
    metrics = simulation.run(benchmark, monkeypatch, aggregation=aggregation)

    assert metrics[14:16] == pytest.approx([250, 4])


@pytest.mark.parametrize("aggregation", ["mean", "median"])
def test_zero_decode_interval_is_unavailable(benchmark, monkeypatch, aggregation):
    simulation = Simulation(benchmark, [Request((73, 0, 0))], 1)
    metrics = simulation.run(benchmark, monkeypatch, aggregation=aggregation)

    assert metrics[14:16] == [None, None]


@pytest.mark.parametrize("emitted", [1, 2])
def test_decode_availability_is_saved_without_nonfinite_json(benchmark, monkeypatch, tmp_path, emitted):
    simulation = Simulation(benchmark, [Request((73,) if emitted == 1 else (73, 4))], 1)
    metrics = simulation.run(benchmark, monkeypatch, generation_length=emitted)
    args = SimpleNamespace(model_name="simulated", precision="fp32", execution_provider="cpu", aggregation="mean")
    destination = tmp_path / "metrics.csv"
    benchmark.save_results(args, [metrics], str(destination))

    def reject_nonfinite(value):
        raise AssertionError(f"Nonfinite JSON value: {value}")

    record = json.loads(destination.with_suffix(".json").read_text(), parse_constant=reject_nonfinite)[0]
    saved = record["metrics"]
    if emitted == 1:
        assert saved["token_generation_metrics_status"] == "unavailable"
        assert "token_generation_throughput_tps" not in saved
        assert "token_generation_latency_ms" not in saved
    else:
        assert saved["token_generation_throughput_tps"] == pytest.approx(250)
        assert saved["token_generation_latency_ms"] == pytest.approx(4)
        assert "token_generation_metrics_status" not in saved

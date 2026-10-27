# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ENGINE_DIR = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("engine_benchmark_run", ENGINE_DIR / "run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def invoke(self, providers, extra=(), returncode=0, write_result=True, result_status="success"):
        observations = []
        active = 0
        peak_active = 0
        lock = threading.Lock()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            output = root / "out"
            config.write_text(
                json.dumps(
                    [
                        {
                            "scenario": "decode_baseline",
                            "model_path": "original",
                            **({"execution_provider": provider} if provider is not None else {}),
                        }
                        for provider in providers
                    ]
                )
            )
            argv = ["run.py", "--executable", "benchmark", "--config", str(config), "--out", str(output), *extra]

            def run(command, **kwargs):
                nonlocal active, peak_active
                scenario = json.loads(Path(command[2]).read_text())[0]
                observations.append((scenario, kwargs["env"]))
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                time.sleep(0.01)
                if write_result:
                    (Path(command[4]) / "decode_baseline_results_001.json").write_text(
                        json.dumps({"status": result_status})
                    )
                with lock:
                    active -= 1
                kwargs["stdout"].write("benchmark stdout\nbenchmark stderr\n")
                return subprocess.CompletedProcess(command, returncode)

            with patch.object(sys, "argv", argv), patch.object(runner.subprocess, "run", side_effect=run):
                status = runner.main()
            files = {path.name: path.read_text() for path in output.glob("*")}
        return status, observations, peak_active, files

    def test_webgpu_runs_serially_without_cuda_ids_and_preserves_results(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}):
            status, observations, peak, files = self.invoke(["webgpu", "webgpu"])
        self.assertEqual(status, 0)
        self.assertEqual(peak, 1)
        self.assertTrue(all("CUDA_VISIBLE_DEVICES" not in env for _, env in observations))
        self.assertIn("decode_baseline_results_001.json", files)
        self.assertIn("decode_baseline_results_002.json", files)
        self.assertIn("benchmark stderr", files["decode_baseline_001.log"])

    def test_cuda_requires_device_ids(self):
        with self.assertRaisesRegex(ValueError, "at least one GPU ID"):
            self.invoke(["cuda"])

    def test_mixed_providers_are_rejected_before_reserving_devices_or_launching_workers(self):
        for providers in (["cuda", "webgpu"], ["webgpu", "cuda"], [None, "webgpu"]):
            for extra in ([], ["--cuda_visible_devices", "2"], ["--cuda_visible_devices", "2,3"]):
                with (
                    self.subTest(providers=providers, extra=extra),
                    patch.object(runner, "Queue") as queue,
                    patch.object(runner, "ThreadPoolExecutor") as executor,
                ):
                    with self.assertRaisesRegex(ValueError, "must not mix execution providers"):
                        self.invoke(providers, extra)
                    queue.assert_not_called()
                    executor.assert_not_called()

    def test_webgpu_stays_serial_when_cuda_ids_are_supplied(self):
        status, _, peak, _ = self.invoke(["webgpu", "webgpu"], ["--cuda_visible_devices", "2,3"])
        self.assertEqual(status, 0)
        self.assertEqual(peak, 1)

    def test_cuda_keeps_parallel_device_isolation(self):
        status, observations, peak, _ = self.invoke(["cuda", "cuda"], ["--cuda_visible_devices", "2,3"])
        self.assertEqual(status, 0)
        self.assertEqual(peak, 2)
        self.assertEqual({env["CUDA_VISIBLE_DEVICES"] for _, env in observations}, {"2", "3"})

    def test_model_and_provider_overrides(self):
        status, observations, _, _ = self.invoke(
            ["webgpu"], ["--model_path", "model", "--execution_provider_library", "provider.dylib"]
        )
        self.assertEqual(status, 0)
        scenario = observations[0][0]
        self.assertEqual(scenario["model_path"], str(Path("model").resolve()))
        self.assertEqual(scenario["execution_provider_library"], str(Path("provider.dylib").resolve()))

    def test_child_failure_is_reported_and_logged(self):
        status, _, _, files = self.invoke(["webgpu"], returncode=1, result_status="failed")
        self.assertEqual(status, 1)
        self.assertIn("benchmark stderr", files["decode_baseline_001.log"])

    def test_missing_result_is_failure(self):
        self.assertEqual(self.invoke(["webgpu"], write_result=False)[0], 1)

    def test_failed_result_is_failure_even_when_child_exits_zero(self):
        self.assertEqual(self.invoke(["webgpu"], result_status="failed")[0], 1)

    def test_smoke_contains_only_minimum_decode_and_prefill_cases(self):
        scenarios = json.loads((ENGINE_DIR / "configs/webgpu-decode-prefill-smoke.json").read_text())
        self.assertEqual(
            [(s["scenario"], s["concurrency"], s["prompt_length_k"]) for s in scenarios],
            [("decode_baseline", 1, 4), ("long_prefill", 1, 32)],
        )
        self.assertTrue(all(s["execution_provider"] == "webgpu" for s in scenarios))
        self.assertTrue(all(s["warmup_runs"] == 1 and s["measured_runs"] == 3 for s in scenarios))

    def test_prefill_scaling_uses_exact_prefixes_with_identical_settings(self):
        scenarios = json.loads((ENGINE_DIR / "configs/webgpu-prefill-scaling.json").read_text())
        self.assertEqual([s["prompt_tokens"] for s in scenarios], [4096, 8192, 16384, 32768])
        settings = [{key: value for key, value in scenario.items() if key != "prompt_tokens"} for scenario in scenarios]
        self.assertTrue(all(s == settings[0] for s in settings))
        self.assertEqual(settings[0]["scenario"], "long_prefill")
        self.assertEqual(settings[0]["prompt_length_k"], 32)
        self.assertEqual(settings[0]["generation_tokens"], 1)
        self.assertFalse(settings[0]["model_config"]["engine"]["dynamic_batching"]["prefix_caching"])


if __name__ == "__main__":
    unittest.main()

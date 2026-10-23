# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from builders.qwen3_8_flash.builder import Qwen38FlashBuilder, package_config
from builders.qwen3_8_flash.calibration import build_runner, calibration_arrays, ple_hash, read_samples, write_request
from builders.qwen3_8_flash.graph import build_target_graph
from builders.qwen3_8_flash.target_export import WEIGHT_RECIPE, finalize_target
from test_qwen38_flash_graph import target_config


class CalibrationTests(unittest.TestCase):
    def test_single_command_runs_raw_export_calibration_draft_and_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, draft, sdk, output = [root / name for name in ("target", "draft", "sdk", "output")]
            target.mkdir()
            draft.mkdir()
            (sdk / "lib").mkdir(parents=True)
            core = sdk / "lib/libonnxruntime.so"
            core.touch()
            provider = sdk / "lib/provider.so"
            provider.touch()
            prompts = root / "samples.jsonl"
            prompts.write_text('{"input_ids":[1,2,3,4]}\n')
            config = target_config()
            config["text_config"].update(eos_token_id=248044, bos_token_id=248044)
            (target / "config.json").write_text(json.dumps(config))
            (target / "tokenizer.json").write_text("{}")
            (draft / "config.json").write_text(
                json.dumps({"num_hidden_layers": 5, "hidden_size": 2560, "num_key_value_heads": 8, "head_dim": 128})
            )
            stages = []

            def export(source, destination, **kwargs):
                stages.append("raw_target")
                self.assertEqual(source, target)
                destination.mkdir()
                (destination / "config.json").write_text(json.dumps(config))

            def calibrate(destination, source, samples, library, headers):
                stages.append("fresh_calibration")
                self.assertEqual(
                    (destination, source, samples, library, headers), (output, target, prompts, provider, sdk)
                )
                (destination / "kv_calibration.json").write_text("{}")

            def finalize(destination):
                stages.append("final_target")
                (destination / "model.onnx").touch()

            def run(command, *, env, check):
                self.assertTrue(check)
                self.assertEqual(env["ORT_LIB_PATH"], str(core))
                if command[2] == "builders.qwen3_8_flash.dflash_export":
                    stages.append("raw_drafter")
                    self.assertIn(str(draft), command)
                    self.assertIn(str(output / "model.onnx"), command)
                    directory = output / ".draft_export"
                    directory.mkdir()
                    (directory / "dflash2.onnx").touch()
                    (directory / "dflash2.onnx.data").touch()
                    (directory / "dflash2_config.json").write_text(
                        json.dumps(
                            {
                                "num_hidden_layers": 5,
                                "num_key_value_heads": 8,
                                "head_size": 128,
                                "block_size": 4,
                                "num_draft_tokens": 3,
                            }
                        )
                    )
                else:
                    stages.append("native_verification")
                    self.assertEqual(command[2], "builders.qwen3_8_flash.verification")
                    self.assertTrue((output / "genai_config.json").exists())

            with (
                patch("builders.qwen3_8_flash.builder.export_target", side_effect=export),
                patch("builders.qwen3_8_flash.builder.calibrate_target", side_effect=calibrate),
                patch("builders.qwen3_8_flash.builder.finalize_target", side_effect=finalize),
                patch("builders.qwen3_8_flash.builder.subprocess.run", side_effect=run),
                patch("builders.qwen3_8_flash.builder.onnx.checker.check_model"),
            ):
                state = Qwen38FlashBuilder.build(target, draft, output, prompts, provider, sdk)
            self.assertEqual(
                stages, ["raw_target", "fresh_calibration", "final_target", "raw_drafter", "native_verification"]
            )
            self.assertTrue(state["package_ready"])
            self.assertTrue(json.loads((output / "model_build.json").read_text())["runtime_smoke_passed"])

    def test_finalization_rejects_unrelated_or_invalid_profile_without_writing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            graph = b"fresh-calibration-graph"
            (output / "target.calibration.onnx").write_bytes(graph)
            (output / "config.json").write_text(json.dumps(target_config()))
            (output / "target_export.json").write_text(json.dumps({"weight_recipe": WEIGHT_RECIPE}))
            (output / "target_weights.json").write_text("{}")
            data = output / "model.onnx.data"
            data.write_bytes(b"unchanged")
            profile = {
                "method": "fresh-bf16-per-channel-amax",
                "graph_sha256": "different",
                "weight_recipe": WEIGHT_RECIPE,
            }
            path = output / "kv_calibration.json"
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "fresh calibration"):
                finalize_target(output)
            profile["graph_sha256"] = hashlib.sha256(graph).hexdigest()
            profile["scales"] = {}
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "exactly the 24"):
                finalize_target(output)
            profile["scales"] = {
                f"{layer}.{kind}": [0.01] * 512 for layer in range(3, 48, 4) for kind in ("key", "value")
            }
            for invalid in ([1.0] * 511, [-1.0] * 512, [float("nan")] * 512):
                profile["scales"]["3.key"] = invalid
                path.write_text(json.dumps(profile))
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    finalize_target(output)
            self.assertEqual(data.read_bytes(), b"unchanged")
            self.assertFalse((output / "model.onnx").exists())

    def test_hash_matches_scalar_cuda_contract_and_eos_reset(self):
        tokens = [248044, 1000, 2000, 248319, 248044, 3000]
        mult = [23703573157769, 20109073645365, 8052911324071]
        vocab = np.array([20000003 + 2 * head for head in range(16)], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(vocab)[:-1]))
        result = ple_hash(tokens, mult, vocab, offsets, 248044)
        for row, token in enumerate(tokens):
            previous = tokens[row - 1] if row else 248044
            older = tokens[row - 2] if row >= 2 else 248044
            if previous == 248044:
                older = 248044
            for head in range(16):
                mixed = (token * mult[0]) ^ (previous * mult[1])
                if head >= 8:
                    mixed ^= older * mult[2]
                mixed &= 2**64 - 1
                if mixed >= 2**63:
                    mixed -= 2**64
                self.assertEqual(result[row, head], mixed % int(vocab[head]) + offsets[head])

    def test_hash_rejects_invalid_metadata_or_tokens(self):
        for tokens in ([], [[1]], [-1]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                ple_hash(tokens, [1, 2, 3], np.ones(16), np.zeros(16), 248044)
        with self.assertRaises(ValueError):
            ple_hash([1], [1, 2], np.ones(16), np.zeros(16), 248044)

    def test_samples_require_bounded_valid_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            good = [1, 2, 3, 4, 5, 6, 7, 8]
            path.write_text(json.dumps({"input_ids": good}) + "\n")
            self.assertEqual(read_samples(path, directory, 8), [good])
            for record in (
                {"input_ids": [1, 2, 3]},
                {"input_ids": [*good, 9]},
                {"input_ids": [1, 2, 3, True]},
                {"input_ids": [1, 2, 3, 248320]},
                {"input_ids": None},
                {"other": "text"},
                [],
            ):
                path.write_text(json.dumps(record) + "\n")
                with self.subTest(record=record), self.assertRaises(ValueError):
                    read_samples(path, directory, 8)
            path.write_text("\n")
            with self.assertRaisesRegex(ValueError, "no samples"):
                read_samples(path, directory, 8)

    def test_calibration_arrays_and_native_request_protocol(self):
        model, _ = build_target_graph(target_config(), calibration=True, calibration_capacity=32)
        ple = SimpleNamespace(eos=248044, embeddings=lambda ids: torch.zeros((1, len(ids), 2560), dtype=torch.bfloat16))
        ids = [1, 2, 3, 4]
        arrays = calibration_arrays(model, ids, ple, 32)
        self.assertEqual(arrays["prefill_cache.past_key_values.3.index_key.scoring"].shape, (1, 8, 128))
        self.assertEqual(arrays["past_key_values.3.key"].shape, (1, 2, 32, 256))
        self.assertEqual(arrays["past_key_values.3.key"].dtype, np.uint16)
        self.assertEqual(arrays["ple_embeddings.1"].shape, (1, 4, 2560))
        self.assertEqual(arrays["past_key_values.1.ple_context"].tolist(), [[248044, 248044]])
        self.assertEqual(arrays["prefill_numerics"].tolist(), [1])
        self.assertEqual(arrays["logits_indices"].tolist(), [3])
        dsa = next(node for node in model.graph.node if node.op_type == "DynamicSparseAttention")
        control = next(value for value in model.graph.initializer if value.name == dsa.input[20])
        self.assertEqual(np.frombuffer(control.raw_data, dtype=np.int32).tolist(), [0])
        self.assertFalse(any(name.startswith("prefill_cache.") and name.endswith(".key") for name in arrays))
        with tempfile.TemporaryDirectory() as directory:
            manifest, outputs = write_request(
                Path(directory) / "sample with spaces",
                model,
                arrays,
                ["calibration.3.key.amax", "validation.aggregate"],
            )
            lines = manifest.read_text().splitlines()
            self.assertEqual(lines[0], "qwen38_calibration_request 1")
            self.assertEqual(int(lines[1]), len(arrays))
            self.assertEqual(len(outputs), 2)
            self.assertIn('"input_ids" 7 2 1 4 ', lines[2])
            self.assertEqual((manifest.parent / "input.0.bin").stat().st_size, 32)

    def test_missing_sdk_fails_before_compiling(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            build_runner(directory, Path(directory) / "build")

    def test_package_contract(self):
        target = target_config()
        target["text_config"].update(eos_token_id=248044, bos_token_id=248044)
        draft = {
            "num_hidden_layers": 5,
            "hidden_size": 2560,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "dflash_config": {"target_layer_ids": [5, 15, 25, 35, 45]},
        }
        section = {
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "head_size": 128,
            "block_size": 4,
            "num_draft_tokens": 3,
        }
        config = package_config(target, draft, section)
        self.assertEqual(config["model"]["context_length"], 262144)
        self.assertEqual(config["model"]["qwen38_graph"]["kv_capacity"], 262148)
        self.assertEqual(config["model"]["dflash2"]["num_draft_tokens"], 3)
        self.assertEqual(config["model"]["dflash2"]["block_size"], 4)
        self.assertFalse(config["search"]["past_present_share_buffer"])
        self.assertNotIn("session_options", section)
        draft["dflash_config"]["target_layer_ids"] = [5, 15]
        with self.assertRaisesRegex(ValueError, "capture layers"):
            package_config(target, draft)


if __name__ == "__main__":
    unittest.main()

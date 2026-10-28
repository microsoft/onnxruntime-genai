# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models import builder


class Qwen38FlashBuilderDispatchTests(unittest.TestCase):
    def make_hf_details(self, checkpoint):
        return {
            "extra_kwargs": {},
            "hf_name": str(checkpoint),
            "hf_config": SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"]),
            "input_path": str(checkpoint),
        }

    def build_options(self):
        return {
            "draft_checkpoint": Path("draft"),
            "calibration_prompts": Path("prompts.jsonl"),
            "provider": Path("libonnxruntime_providers_cuda.so"),
            "ort_home": Path("ort"),
            "workers": 8,
            "threads": 1,
            "calibration_capacity": 4096,
            "resume": False,
        }

    def test_specialized_mode_dispatches_through_existing_builder_entrypoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "target"
            checkpoint.mkdir()
            with patch.object(builder.Qwen38FlashBuilder, "build") as build:
                builder.create_model(
                    str(checkpoint),
                    str(checkpoint),
                    str(root / "output"),
                    "bf16",
                    "cuda",
                    str(root / "cache"),
                    hf_details=self.make_hf_details(checkpoint),
                    _qwen38_flash_build=self.build_options(),
                )
            args, kwargs = build.call_args
            self.assertEqual(args[:3], (str(checkpoint), Path("draft"), str(root / "output")))
            self.assertEqual(kwargs["resume"], False)

    def test_specialized_mode_requires_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "target"
            checkpoint.mkdir()
            with self.assertRaisesRegex(ValueError, "requires --execution_provider cuda"):
                builder.create_model(
                    str(checkpoint),
                    str(checkpoint),
                    str(root / "output"),
                    "bf16",
                    "cpu",
                    str(root / "cache"),
                    hf_details=self.make_hf_details(checkpoint),
                    _qwen38_flash_build=self.build_options(),
                )


if __name__ == "__main__":
    unittest.main()

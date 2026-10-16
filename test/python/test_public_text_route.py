# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Public token-only Nemotron text contracts; no NumPy BF16 construction.

Run this file with a built onnxruntime_genai binding. Set
GENAI_TEST_PROVIDER=cuda to execute the same tiny contracts on CUDA.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
import onnxruntime_genai as og


class PublicTextRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).resolve().parents[1] / "models/cache-dispatch/phase-multimodal-bf16"
        cls.config = og.Config(str(cls.path))
        cls.provider = os.environ.get("GENAI_TEST_PROVIDER", "cpu")
        if cls.provider not in ("cpu", "cuda"):
            raise ValueError("GENAI_TEST_PROVIDER must be cpu or cuda")
        if cls.provider == "cuda":
            cls.config.clear_providers()
            cls.config.append_provider("cuda")
        cls.model = og.Model(cls.config)
        if cls.model.device_type.upper() != cls.provider.upper():
            raise AssertionError(f"Requested {cls.provider}, got {cls.model.device_type}")

    def generator(self, **options):
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=6, do_sample=False, **options)
        return og.Generator(self.model, params)

    @staticmethod
    def inputs(ids, **extras):
        inputs = og.NamedTensors()
        inputs["input_ids"] = np.asarray([ids], dtype=np.int32)
        for name, value in extras.items():
            inputs[name] = value
        return inputs

    def phase(self, generator, expected):
        phase = generator.get_output("observed_phase")
        self.assertEqual(phase.dtype, np.dtype("bool"))
        self.assertEqual(phase.shape, ())
        self.assertEqual(bool(phase), expected)

    def terminal(self, generator):
        with self.assertRaisesRegex(RuntimeError, "create a new Generator"):
            generator.append_tokens(np.asarray([2], dtype=np.int32))

    def test_public_tokenizer_append_and_set_inputs_generate_identically(self):
        tokenizer = og.Tokenizer(self.model)
        tokens = tokenizer.encode("ab")
        np.testing.assert_array_equal(tokens, np.asarray([2, 3], dtype=np.int32))
        sequences = []
        for entry in ("append", "set_inputs"):
            generator = self.generator()
            if entry == "append":
                generator.append_tokens(tokens)
            else:
                generator.set_inputs(self.inputs(tokens.tolist()))
            self.phase(generator, False)
            for step in range(4):
                generator.generate_next_token()
                if step < 3:
                    generator.get_logits()
                    self.phase(generator, True)
                self.assertEqual(generator.get_next_tokens().dtype, np.dtype("int32"))
            self.assertTrue(generator.is_done())
            sequences.append(generator.get_sequence(0).copy())
            self.assertIsInstance(tokenizer.decode(sequences[-1]), str)
        np.testing.assert_array_equal(*sequences)

    def test_fresh_single_token_prefill_is_false(self):
        generator = self.generator()
        generator.append_tokens(np.asarray([2], dtype=np.int32))
        self.phase(generator, False)
        generator.append_tokens(np.asarray([3], dtype=np.int32))
        self.phase(generator, True)

    def test_optional_all_ones_mask_stays_prompt_only(self):
        generator = self.generator()
        generator.set_inputs(self.inputs([2, 3], attention_mask=np.ones((1, 2), dtype=np.int64)))
        self.phase(generator, False)
        generator.append_tokens(np.asarray([4], dtype=np.int32))
        self.phase(generator, True)

    def test_placeholder_and_padding_fail_initial_and_cached_and_poison(self):
        for token, message in ((18, "media placeholders"), (0, "must be unpadded")):
            for cached in (False, True):
                for entry in ("append", "set_inputs"):
                    with self.subTest(token=token, cached=cached, entry=entry):
                        generator = self.generator()
                        if cached:
                            generator.append_tokens(np.asarray([2], dtype=np.int32))
                            self.phase(generator, False)
                        with self.assertRaisesRegex(RuntimeError, message):
                            if entry == "append":
                                generator.append_tokens(np.asarray([token], dtype=np.int32))
                            else:
                                generator.set_inputs(self.inputs([token]))
                        if cached:
                            self.phase(generator, False)
                        self.terminal(generator)
        fresh = self.generator()
        fresh.append_tokens(np.asarray([2], dtype=np.int32))
        self.phase(fresh, False)

    def test_raw_media_and_unsupported_extras_reject_and_poison(self):
        for name, value in (
            ("pixel_values", np.zeros((1, 3, 2, 2), dtype=np.float32)),
            ("is_video", np.asarray(False)),
            ("retention_mask", np.asarray([True])),
            ("kept_indices", np.asarray([0], dtype=np.int64)),
            ("num_image_tokens", np.asarray([0], dtype=np.int64)),
            ("position_ids", np.asarray([[0]], dtype=np.int64)),
            ("arbitrary", np.asarray(1, dtype=np.int32)),
        ):
            with self.subTest(name=name):
                generator = self.generator()
                with self.assertRaisesRegex(RuntimeError, "does not accept media or extra inputs"):
                    generator.set_inputs(self.inputs([2], **{name: value}))
                self.terminal(generator)

    def test_bad_masks_and_ids_reject(self):
        for mask in (
            np.zeros((1, 2), dtype=np.int64),
            np.ones((1, 1), dtype=np.int64),
            np.ones((1, 2), dtype=np.float32),
        ):
            generator = self.generator()
            with self.assertRaises(RuntimeError):
                generator.set_inputs(self.inputs([2, 3], attention_mask=mask))
            self.terminal(generator)
        for ids in (np.asarray([2], dtype=np.int32), np.asarray([[2]], dtype=np.int64)):
            generator = self.generator()
            inputs = og.NamedTensors()
            inputs["input_ids"] = ids
            with self.assertRaisesRegex(RuntimeError, "CPU int32 input_ids"):
                generator.set_inputs(inputs)
            self.terminal(generator)

    def test_cached_extra_inputs_are_not_silently_ignored(self):
        for name, value in (
            ("pixel_values", np.zeros((1, 3, 2, 2), dtype=np.float32)),
            ("image_features", np.zeros((0, 2), dtype=np.float32)),
            ("attention_mask", np.ones((1, 1), dtype=np.int64)),
            ("has_previous_state", np.asarray(False)),
            ("arbitrary", np.asarray(1, dtype=np.int32)),
        ):
            with self.subTest(name=name):
                generator = self.generator()
                generator.append_tokens(np.asarray([2], dtype=np.int32))
                self.phase(generator, False)
                with self.assertRaisesRegex(RuntimeError, "cached SetInputs accepts only input_ids"):
                    generator.set_inputs(self.inputs([3], **{name: value}))
                self.phase(generator, False)
                self.terminal(generator)

    def test_prepared_features_still_enforce_shape_and_type(self):
        for features in (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float16),
            np.zeros((1, 1, 2), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        ):
            generator = self.generator()
            with self.assertRaisesRegex(RuntimeError, "shape|element type"):
                generator.set_inputs(self.inputs([2], image_features=features))
            self.terminal(generator)

    def test_lifecycle_rejections_and_duplicate_phase_remain(self):
        for options in ({"chunk_size": 1}, {"num_beams": 2}, {"batch_size": 2}):
            with self.assertRaises(RuntimeError):
                self.generator(**options)
        generator = self.generator()
        generator.append_tokens(np.asarray([2], dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, "rewind"):
            generator.rewind_to(0)
        with self.assertRaisesRegex(RuntimeError, "cached decoding requires S==1"):
            generator.append_tokens(np.asarray([3, 4], dtype=np.int32))
        self.terminal(generator)
        duplicate = self.generator()
        with self.assertRaisesRegex(RuntimeError, "runtime-owned"):
            duplicate.set_inputs(self.inputs([2], has_previous_state=np.asarray(False)))
        self.terminal(duplicate)

    def test_existing_float32_numpy_tensor_type_unchanged(self):
        tensor = og.Tensor(np.asarray([1.25, -2], dtype=np.float32))
        np.testing.assert_array_equal(tensor.as_numpy(), [1.25, -2])

    def test_actual_embedding_failure_before_and_after_cached_success_is_terminal(self):
        config = og.Config(str(self.path.with_name("phase-multimodal-bf16-embedding-failure")))
        if self.provider == "cuda":
            config.clear_providers()
            config.append_provider("cuda")
        model = og.Model(config)
        for cached in (False, True):
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=8, do_sample=False)
            generator = og.Generator(model, params)
            if cached:
                generator.append_tokens(np.asarray([2], dtype=np.int32))
                self.phase(generator, False)
                generator.append_tokens(np.asarray([3], dtype=np.int32))
                self.phase(generator, True)
            else:
                self.assertFalse(bool(generator.get_input("has_previous_state")))
            with self.assertRaisesRegex(RuntimeError, "embedding_failure_reshape"):
                generator.append_tokens(np.asarray([13], dtype=np.int32))
            if cached:
                self.phase(generator, True)  # Prior successful decoder output, not a new commit.
            else:
                self.assertFalse(bool(generator.get_input("has_previous_state")))
            self.terminal(generator)
        params = og.GeneratorParams(model)
        fresh = og.Generator(model, params)
        fresh.append_tokens(np.asarray([2], dtype=np.int32))
        self.phase(fresh, False)


if __name__ == "__main__":
    unittest.main()

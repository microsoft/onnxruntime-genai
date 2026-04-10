# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from ext_test_case import ExtTestCase


class TestFixGenaiConfig(ExtTestCase):
    """Tests for the fix_genai_config utility function."""

    def _make_config_with_nulls(self):
        """Return a genai_config dict whose search section is all-null,
        as can happen when transformers >= 5 is used."""
        return {
            "model": {
                "bos_token_id": 1,
                "context_length": 2048,
                "decoder": {
                    "filename": "model.onnx",
                    "head_size": 64,
                    "hidden_size": 512,
                    "inputs": {"input_ids": "input_ids"},
                    "num_attention_heads": 8,
                    "num_hidden_layers": 1,
                    "num_key_value_heads": 4,
                    "outputs": {"logits": "logits"},
                    "session_options": {"log_id": "onnxruntime-genai", "provider_options": []},
                },
                "eos_token_id": 2,
                "pad_token_id": 2,
                "type": "llama",
                "vocab_size": 32000,
            },
            "search": {
                "diversity_penalty": None,
                "do_sample": None,
                "early_stopping": None,
                "length_penalty": None,
                "max_length": 2048,
                "min_length": None,
                "no_repeat_ngram_size": None,
                "num_beams": None,
                "num_return_sequences": None,
                "past_present_share_buffer": True,
                "repetition_penalty": None,
                "temperature": None,
                "top_k": None,
                "top_p": None,
            },
        }

    def test_fix_genai_config_replaces_null_search_values(self):
        """fix_genai_config replaces every null in the search section with
        the onnxruntime-genai default value."""
        from models.genai_config_utils import fix_genai_config

        config = self._make_config_with_nulls()
        result = fix_genai_config(config)

        search = result["search"]
        self.assertEqual(search["diversity_penalty"], 0.0)
        self.assertEqual(search["do_sample"], False)
        self.assertEqual(search["early_stopping"], True)
        self.assertEqual(search["length_penalty"], 1.0)
        self.assertEqual(search["min_length"], 0)
        self.assertEqual(search["no_repeat_ngram_size"], 0)
        self.assertEqual(search["num_beams"], 1)
        self.assertEqual(search["num_return_sequences"], 1)
        self.assertEqual(search["repetition_penalty"], 1.0)
        self.assertEqual(search["temperature"], 1.0)
        self.assertEqual(search["top_k"], 50)
        self.assertEqual(search["top_p"], 1.0)

    def test_fix_genai_config_preserves_non_null_search_values(self):
        """fix_genai_config must not overwrite already-set search values."""
        from models.genai_config_utils import fix_genai_config

        config = self._make_config_with_nulls()
        config["search"]["temperature"] = 0.7
        config["search"]["top_k"] = 40
        config["search"]["do_sample"] = True

        result = fix_genai_config(config)

        self.assertAlmostEqual(result["search"]["temperature"], 0.7)
        self.assertEqual(result["search"]["top_k"], 40)
        self.assertEqual(result["search"]["do_sample"], True)
        # Other null values are still filled in.
        self.assertEqual(result["search"]["top_p"], 1.0)
        self.assertEqual(result["search"]["num_beams"], 1)

    def test_fix_genai_config_preserves_extra_keys(self):
        """Keys that are not in the defaults dict must not be touched."""
        from models.genai_config_utils import fix_genai_config

        config = self._make_config_with_nulls()
        # max_length and past_present_share_buffer are not in the defaults
        result = fix_genai_config(config)

        self.assertEqual(result["search"]["max_length"], 2048)
        self.assertEqual(result["search"]["past_present_share_buffer"], True)

    def test_fix_genai_config_returns_same_dict(self):
        """fix_genai_config modifies the dict in-place and returns it."""
        from models.genai_config_utils import fix_genai_config

        config = self._make_config_with_nulls()
        result = fix_genai_config(config)
        self.assertIs(result, config)

    def test_fix_genai_config_no_search_section(self):
        """fix_genai_config is a no-op when there is no search section."""
        from models.genai_config_utils import fix_genai_config

        config = {"model": {"type": "llama"}}
        result = fix_genai_config(config)
        # No search key is added.
        self.assertNotIn("search", result)

    def test_fix_genai_config_already_valid(self):
        """fix_genai_config leaves a fully-populated config unchanged."""
        from models.genai_config_utils import fix_genai_config

        config = {
            "search": {
                "diversity_penalty": 0.0,
                "do_sample": False,
                "early_stopping": True,
                "length_penalty": 1.0,
                "max_length": 512,
                "min_length": 0,
                "no_repeat_ngram_size": 0,
                "num_beams": 1,
                "num_return_sequences": 1,
                "past_present_share_buffer": False,
                "repetition_penalty": 1.0,
                "temperature": 1.0,
                "top_k": 50,
                "top_p": 1.0,
            }
        }
        import copy

        original = copy.deepcopy(config)
        fix_genai_config(config)
        self.assertEqual(config["search"], original["search"])

    def test_fix_genai_config_importable(self):
        """fix_genai_config is accessible from models.genai_config_utils."""
        from models.genai_config_utils import fix_genai_config

        self.assertTrue(callable(fix_genai_config))


if __name__ == "__main__":
    unittest.main(verbosity=2)

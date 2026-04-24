# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from model_builder_test_case import ModelBuilderTestCase, hide_stdout


class TestCheckExtraOptions(ModelBuilderTestCase):
    """Tests for the check_extra_options utility function."""

    # All keys that check_extra_options treats as booleans.
    _ALL_BOOL_KEYS = [
        "int4_is_symmetric",
        "exclude_embeds",
        "exclude_lm_head",
        "include_hidden_states",
        "enable_cuda_graph",
        "enable_webgpu_graph",
        "use_8bits_moe",
        "use_qdq",
        "use_webgpu_fp32",
        "use_cuda_bf16",
        "shared_embeddings",
        "hf_remote",
        "disable_qkv_fusion",
        "prune_lm_head",
    ]

    def _check(self, kv_pairs, execution_provider="cpu"):
        from models.builder import check_extra_options

        check_extra_options(kv_pairs, execution_provider)

    # ------------------------------------------------------------------
    # Boolean conversion
    # ------------------------------------------------------------------

    def test_bool_false_variants_set_to_false(self):
        """All falsy string representations are converted to False."""
        for false_str in ("false", "False", "0"):
            for key in self._ALL_BOOL_KEYS:
                kv = {key: false_str}
                self._check(kv)
                self.assertIs(kv[key], False, f"Expected False for key={key!r}, value={false_str!r}")

    def test_bool_true_variants_set_to_true(self):
        """All truthy string representations are converted to True.

        'enable_webgpu_graph' is excluded here because the EP guard that
        follows the conversion resets it to False when EP != 'webgpu'.
        It is covered separately in test_enable_webgpu_graph_*.
        """
        keys_without_webgpu_guard = [k for k in self._ALL_BOOL_KEYS if k != "enable_webgpu_graph"]
        for true_str in ("true", "True", "1"):
            for key in keys_without_webgpu_guard:
                kv = {key: true_str}
                self._check(kv)
                self.assertIs(kv[key], True, f"Expected True for key={key!r}, value={true_str!r}")

    def test_invalid_bool_value_raises(self):
        """An unrecognised value for a bool key must raise ValueError."""
        from models.builder import check_extra_options

        with self.assertRaises(ValueError):
            check_extra_options({"exclude_embeds": "yes"}, "cpu")

    def test_unrelated_keys_not_touched(self):
        """Keys that are not in the bool list are left as-is."""
        kv = {"some_unknown_key": "hello"}
        self._check(kv)
        self.assertEqual(kv["some_unknown_key"], "hello")

    # ------------------------------------------------------------------
    # int4_op_types_to_quantize
    # ------------------------------------------------------------------

    def test_int4_op_types_to_quantize_single(self):
        """A single op type is wrapped in a tuple."""
        kv = {"int4_op_types_to_quantize": "MatMul"}
        self._check(kv)
        self.assertEqual(kv["int4_op_types_to_quantize"], ("MatMul",))

    def test_int4_op_types_to_quantize_multiple(self):
        """Multiple op types separated by '/' are split into a tuple."""
        kv = {"int4_op_types_to_quantize": "MatMul/Gather"}
        self._check(kv)
        self.assertEqual(kv["int4_op_types_to_quantize"], ("MatMul", "Gather"))

    # ------------------------------------------------------------------
    # int4_nodes_to_exclude
    # ------------------------------------------------------------------

    def test_int4_nodes_to_exclude_single(self):
        """A single node name is wrapped in a list."""
        kv = {"int4_nodes_to_exclude": "node_a"}
        self._check(kv)
        self.assertEqual(kv["int4_nodes_to_exclude"], ["node_a"])

    def test_int4_nodes_to_exclude_multiple(self):
        """Multiple node names separated by ',' are split into a list."""
        kv = {"int4_nodes_to_exclude": "node_a,node_b,node_c"}
        self._check(kv)
        self.assertEqual(kv["int4_nodes_to_exclude"], ["node_a", "node_b", "node_c"])

    # ------------------------------------------------------------------
    # Mutual-exclusion: exclude_lm_head + include_hidden_states
    # ------------------------------------------------------------------

    def test_exclude_lm_head_and_include_hidden_states_raises(self):
        """Using both 'exclude_lm_head' and 'include_hidden_states' raises ValueError."""
        from models.builder import check_extra_options

        kv = {"exclude_lm_head": "true", "include_hidden_states": "true"}
        with self.assertRaises(ValueError):
            check_extra_options(kv, "cpu")

    def test_exclude_lm_head_alone_is_valid(self):
        """'exclude_lm_head' alone must not raise."""
        kv = {"exclude_lm_head": "true"}
        self._check(kv)
        self.assertIs(kv["exclude_lm_head"], True)

    def test_include_hidden_states_alone_is_valid(self):
        """'include_hidden_states' alone must not raise."""
        kv = {"include_hidden_states": "true"}
        self._check(kv)
        self.assertIs(kv["include_hidden_states"], True)

    # ------------------------------------------------------------------
    # enable_webgpu_graph EP guard
    # ------------------------------------------------------------------

    def test_enable_webgpu_graph_with_webgpu_ep_kept(self):
        """enable_webgpu_graph=true stays True when EP is 'webgpu'."""
        kv = {"enable_webgpu_graph": "true"}
        self._check(kv, execution_provider="webgpu")
        self.assertIs(kv["enable_webgpu_graph"], True)

    @hide_stdout()
    def test_enable_webgpu_graph_with_non_webgpu_ep_disabled(self):
        """enable_webgpu_graph=true is set to False for non-webgpu EPs."""
        for ep in ("cpu", "cuda", "dml"):
            kv = {"enable_webgpu_graph": "true"}
            self._check(kv, execution_provider=ep)
            self.assertIs(kv["enable_webgpu_graph"], False, f"Expected False for ep={ep!r}")

    def test_enable_webgpu_graph_false_with_non_webgpu_ep_unchanged(self):
        """enable_webgpu_graph=false is never forced to True."""
        kv = {"enable_webgpu_graph": "false"}
        self._check(kv, execution_provider="cpu")
        self.assertIs(kv["enable_webgpu_graph"], False)

    # ------------------------------------------------------------------
    # Empty / no-op cases
    # ------------------------------------------------------------------

    def test_empty_dict_is_valid(self):
        """An empty kv_pairs dict must not raise."""
        kv = {}
        self._check(kv)
        self.assertEqual(kv, {})


class TestParseExtraOptions(ModelBuilderTestCase):
    """Tests for the parse_extra_options helper."""

    @hide_stdout()
    def test_none_items_returns_empty_dict(self):
        """parse_extra_options(None, ...) returns an empty dict."""
        from models.builder import parse_extra_options

        result = parse_extra_options(None, "cpu")
        self.assertEqual(result, {})

    @hide_stdout()
    def test_empty_list_returns_empty_dict(self):
        """parse_extra_options([], ...) returns an empty dict."""
        from models.builder import parse_extra_options

        result = parse_extra_options([], "cpu")
        self.assertEqual(result, {})

    @hide_stdout()
    def test_single_key_value_pair(self):
        """A single 'key=value' string is parsed correctly."""
        from models.builder import parse_extra_options

        result = parse_extra_options(["exclude_embeds=true"], "cpu")
        self.assertIs(result["exclude_embeds"], True)

    @hide_stdout()
    def test_multiple_key_value_pairs(self):
        """Multiple 'key=value' strings are all parsed."""
        from models.builder import parse_extra_options

        result = parse_extra_options(["exclude_embeds=false", "int4_nodes_to_exclude=node_a,node_b"], "cpu")
        self.assertIs(result["exclude_embeds"], False)
        self.assertEqual(result["int4_nodes_to_exclude"], ["node_a", "node_b"])

    @hide_stdout()
    def test_whitespace_around_key_and_value_stripped(self):
        """Leading/trailing whitespace around keys and values is stripped."""
        from models.builder import parse_extra_options

        result = parse_extra_options([" exclude_embeds = true "], "cpu")
        self.assertIs(result["exclude_embeds"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)

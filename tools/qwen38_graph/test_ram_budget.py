# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import unittest

from run_with_ram_budget import plan_environment


class RamBudgetTest(unittest.TestCase):
    def test_64_gib_plan_and_unlimited(self):
        original = {"CUSTOM": "keep"}
        env, mib = plan_environment(64, 8, 48, original)
        self.assertEqual(mib, 56 * 1024)
        self.assertEqual(env["ORT_QMOE_HOST_CACHE_LAYERS"], "48")
        self.assertEqual(original, {"CUSTOM": "keep"})
        env, mib = plan_environment(0, 8, 48, env)
        self.assertEqual(mib, 0)
        self.assertEqual(env["ORT_QMOE_HOST_CACHE_MIB"], "0")
        self.assertNotIn("ORT_QMOE_HOST_CACHE_LAYERS", env)
        self.assertEqual(env["CUSTOM"], "keep")

    def test_invalid_and_incompatible_budgets(self):
        for ram, reserve, layers in (
            (8, 8, 48),
            (1, 0.99, 48),
            (-1, 0, 48),
            (float("nan"), 0, 48),
            (float("inf"), 0, 48),
            (64, 8, 0),
        ):
            with self.subTest(ram=ram, reserve=reserve, layers=layers), self.assertRaises(ValueError):
                plan_environment(ram, reserve, layers, {})
        for flag in (
            "ORT_QMOE_HOST_COMPLEMENT_MODE",
            "ORT_QMOE_SSD_CACHE_SLOTS",
            "ORT_QMOE_SHARED_PIN_EXPERTS",
            "ORT_QMOE_SHARED_STREAM_ROWS",
            "ORTGENAI_PIN_CPU_POOL",
        ):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                plan_environment(64, 8, 48, {flag: "1"})


if __name__ == "__main__":
    unittest.main()

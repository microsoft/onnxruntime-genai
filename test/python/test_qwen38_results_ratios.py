# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest

QWEN38_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "python" / "qwen38"
_spec = importlib.util.spec_from_file_location("check_results_ratios", QWEN38_DIR / "check_results_ratios.py")
checker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checker)

RESULTS = checker.RESULTS.read_text(encoding="utf-8")
DASH = checker.DASH


def mismatches(text):
    return [item for item in checker.check_results(text).items if not item.ok]


def test_quoted_ratios_follow_the_recorded_rows():
    assert mismatches(RESULTS) == []
    assert checker.main([]) == 0


@pytest.mark.parametrize(
    "baseline,session1,final,expected",
    [
        ("2.65", "1.97", ("2.38", "2.38"), ("1.11x", "0.83x")),
        ("8.07", "3.21", ("3.06", "3.06"), ("2.64x", "1.05x")),
        ("32.02", "9.48", ("9.57", "10.78"), (f"2.97x{DASH}3.35x", f"0.88x{DASH}0.99x")),
        ("140.47", "42.16", ("42.94", "43.98"), (f"3.19x{DASH}3.27x", f"0.96x{DASH}0.98x")),
    ],
)
def test_ttft_factors_divide_baseline_and_session1_by_the_final_time(baseline, session1, final, expected):
    assert checker.ttft_factors(Decimal(baseline), Decimal(session1), tuple(map(Decimal, final))) == expected


def test_rounding_is_half_up_at_the_documented_precision():
    assert checker.rounded(Decimal("2.665"), 2) == Decimal("2.67")
    assert checker.gain(Decimal(17) / Decimal(16)) == Decimal("6.3")


@pytest.mark.parametrize(
    "old,new",
    [
        (f"`1.11x`{DASH}`3.35x`", f"`2.64x`{DASH}`3.20x`"),  # the range that did not follow from the rows
        ("**1.45x**", "**1.55x**"),
        (f"42.94{DASH}43.98 s", f"42.94{DASH}40.98 s"),
        ("`+128.4%` (", "`+130.0%` ("),
        ("by **88.6%**", "by **89.6%**"),
        ("116,123 MiB (113.40 GiB)", "116,123 MiB (113.41 GiB)"),
    ],
)
def test_stale_or_hand_edited_values_are_reported(old, new):
    assert old in RESULTS
    assert mismatches(RESULTS.replace(old, new))


def test_unreadable_rows_are_reported_instead_of_raised():
    assert mismatches(RESULTS.replace("| 128 | 2.65 s |", "| 128 | n/a |"))

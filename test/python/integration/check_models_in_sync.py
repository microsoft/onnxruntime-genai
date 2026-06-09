#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Verify the integration pipeline's model lists stay in sync with models.py.

The pipeline file ``.pipelines/integration-tests.yml`` declares ``pr_models``
and ``all_models`` parameter defaults so each ADO job can fan out per model.
Those lists must match the ``pr`` and ``all_`` suites in ``models.py``;
otherwise PRs and main merges silently test a different set of models from
what the catalog claims.

The pipeline passes its own lists in as arguments, so this script doesn't
need to know where the YAML lives or how to parse it:

    python check_models_in_sync.py \\
        --pr   qwen2.5-0.5b-instruct,qwen3-0.6b,... \\
        --all  qwen2.5-0.5b-instruct,qwen3-0.6b,...

Exits non-zero with a clear diff on mismatch.
"""

from __future__ import annotations

import argparse
import sys

import models


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _diff(expected: list[str], actual: list[str]) -> list[str]:
    exp_set, act_set = set(expected), set(actual)
    missing = [m for m in expected if m not in act_set]
    extra = [m for m in actual if m not in exp_set]
    lines: list[str] = []
    if missing:
        lines.append(f"  missing from pipeline yaml: {missing}")
    if extra:
        lines.append(f"  extra in pipeline yaml:    {extra}")
    if not missing and not extra and expected != actual:
        lines.append(f"  order differs:\n    models.py: {expected}\n    yaml:      {actual}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pr",
        required=True,
        help="Comma-separated pr_models list from the pipeline yaml.",
    )
    parser.add_argument(
        "--all",
        dest="all_models",
        required=True,
        help="Comma-separated all_models list from the pipeline yaml.",
    )
    args = parser.parse_args(argv)

    suites = (
        ("pr", "pr_models", list(models.pr), _split(args.pr)),
        ("all", "all_models", list(models.all_), _split(args.all_models)),
    )

    problems: list[str] = []
    for suite_name, yaml_key, expected, actual in suites:
        diff = _diff(expected, actual)
        if diff:
            problems.append(
                f"Suite '{suite_name}' is out of sync: "
                f"models.py '{suite_name}' vs pipeline '{yaml_key}':\n"
                + "\n".join(diff)
            )

    if problems:
        print("ERROR: integration pipeline model lists are out of sync.", file=sys.stderr)
        for p in problems:
            print(p, file=sys.stderr)
        print(
            "\nFix: edit .pipelines/integration-tests.yml so 'pr_models' and "
            "'all_models' defaults match the 'pr' and 'all_' lists in "
            "test/python/integration/models.py.",
            file=sys.stderr,
        )
        return 1

    print("OK: pipeline model lists match models.py (pr and all suites).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

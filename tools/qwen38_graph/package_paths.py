# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Path-safety helpers shared by the Qwen3.8 package builders."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def reject_output_overlap(out: Path, inputs: Mapping[str, Path | None]) -> Path:
    """Return the resolved `out` after refusing any overlap with an input or reuse source.

    The builders delete and re-link files under the output folder, so an output that is equal to, nested
    inside or around an input would delete or rewrite the artifacts that are about to be read. Both sides
    are canonicalised (symlinks resolved) first, and the check runs before the caller creates or removes
    anything. Inputs that were not given (`None`) are skipped; files are checked like folders.
    """
    resolved_out = Path(out).resolve()
    for label, given in inputs.items():
        if given is None:
            continue
        source = Path(given).resolve()
        if resolved_out == source or resolved_out.is_relative_to(source) or source.is_relative_to(resolved_out):
            raise SystemExit(
                f"--out ({resolved_out}) must not be equal to, nested inside or contain {label} ({source})"
            )
    return resolved_out

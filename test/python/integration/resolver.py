# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Resolve a logical model id + device to an on-disk model directory.

The integration suite reads models from a directory that mirrors the
``foundrylocalmodels/models/`` blob layout:

    <root>/<logical_id>/onnx/<device_dir>/v<N>/genai_config.json

CI populates ``<root>`` by azcopy-syncing the prefixes returned by
``test/python/integration/suite_paths.py``. Local devs can either point
``--model-root`` (or ``ORTGENAI_MODEL_ROOT``) at a similarly shaped local
folder, or do their own one-off azcopy.

The resolver picks the highest available ``vN`` directory automatically.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from . import models

_VERSION_DIR = re.compile(r"^v(\d+)$")


def _newest_version_dir(base: Path) -> Path | None:
    if not base.is_dir():
        return None
    versions = []
    for child in base.iterdir():
        m = _VERSION_DIR.match(child.name)
        if m and child.is_dir():
            versions.append((int(m.group(1)), child))
    if not versions:
        return None
    return max(versions, key=lambda v: v[0])[1]


def get_path_for(
    logical_id: str,
    device: str,
    *,
    model_root: str | None = None,
) -> Path:
    """Resolve ``(logical_id, device)`` to an ORT GenAI model directory.

    Skips the test if the model doesn't declare support for that device.
    Fails the test if the directory is missing under ``model_root`` - that
    indicates a stale azcopy filter or a missing upload, both worth
    surfacing loudly.
    """
    if not models.supports(logical_id, device):
        pytest.skip(f"Model '{logical_id}' does not support device '{device}'.")

    root = model_root or os.environ.get("ORTGENAI_MODEL_ROOT")
    if not root:
        pytest.skip("No model source configured. Set ORTGENAI_MODEL_ROOT or pass --model-root.")

    base = Path(root) / models.storage_subpath(logical_id, device)
    chosen = _newest_version_dir(base)
    if chosen is None:
        pytest.fail(f"Model '{logical_id}' (device={device}) has no v<N> directory under {base}.")
    if not (chosen / "genai_config.json").exists():
        pytest.fail(f"Model '{logical_id}' (device={device}) has no genai_config.json at {chosen}.")
    return chosen

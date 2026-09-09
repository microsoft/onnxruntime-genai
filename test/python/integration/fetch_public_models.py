# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Fetch immutable public integration artifacts into the resolver's v<N> layout."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

if __package__:
    from . import models
else:
    import models


def verify_file(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"Pinned artifact SHA-256 mismatch: {path}; expected {expected}, got {digest.hexdigest()}")


def verify_artifact(directory: Path, logical_id: str, device: str) -> None:
    for filename, expected in models.PUBLIC_IDENTITY[logical_id][device].items():
        verify_file(directory / filename, expected)


def fetch(logical_id: str, device: str, root: Path) -> Path:
    artifact = models.PUBLIC_ARTIFACTS[logical_id]
    if device not in artifact["subdirs"]:
        raise ValueError(f"No verified public artifact for {logical_id}/{device}")
    destination = root / models.storage_subpath(logical_id, device) / f"v{models.pinned_version(logical_id)}"
    destination.mkdir(parents=True, exist_ok=True)
    for filename, expected in models.PUBLIC_IDENTITY[logical_id][device].items():
        target = destination / filename
        if target.exists():
            verify_file(target, expected)
            continue
        source = Path(
            hf_hub_download(
                repo_id=artifact["repo_id"],
                revision=artifact["revision"],
                subfolder=artifact["subdirs"][device],
                filename=filename,
                cache_dir=str(root / ".huggingface"),
                token=False,
            )
        )
        verify_file(source, expected)
        # ORT rejects external data with multiple hard links. Keep an ordinary
        # copy rather than linking the Hub cache into the normalized layout.
        shutil.copyfile(source, target)
        print(f"Verified {target}", flush=True)
    print(f"Ready: {destination} (revision {artifact['revision']})", flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(models.PUBLIC_ARTIFACTS))
    parser.add_argument("--device", required=True, choices=list(models.DEVICE_DIRNAMES))
    parser.add_argument("--model-root", required=True, type=Path)
    args = parser.parse_args()
    fetch(args.model, args.device, args.model_root)


if __name__ == "__main__":
    main()

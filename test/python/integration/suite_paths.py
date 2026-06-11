#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Print the storage subpath for a single (model, device) pair.

Used by the integration pipeline's per-model fetch step. Prints nothing
(empty string) if the model doesn't support the device.

Usage:
    python test/python/integration/suite_paths.py --model qwen3-0.6b --device cuda
"""

from __future__ import annotations

import argparse

import models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(models.MODELS))
    parser.add_argument("--device", required=True, choices=list(models.DEVICE_DIRNAMES))
    args = parser.parse_args()

    if not models.supports(args.model, args.device):
        return
    print(models.storage_subpath(args.model, args.device))


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate the external-data variant used by the memory asset API tests."""

from pathlib import Path

import onnx


def main():
    source = Path(__file__).parent
    destination = source / "external-data"
    destination.mkdir(exist_ok=True)
    weights = destination / "weights.data"
    if weights.exists():
        weights.unlink()
    model = onnx.load(source / "past.onnx")
    onnx.save_model(
        model,
        destination / "past.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weights.name,
        size_threshold=0,
        convert_attribute=False,
    )


if __name__ == "__main__":
    main()

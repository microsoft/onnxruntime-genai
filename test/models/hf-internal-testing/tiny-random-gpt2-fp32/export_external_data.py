# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate the external-data variant used by the memory asset API tests."""

import argparse
from pathlib import Path

import onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--filename", default="past.onnx")
    parser.add_argument("--weights-filename", default="weights.data")
    parser.add_argument(
        "--float-initializers-only",
        action="store_true",
        help="Keep integer shape/axis constants inline for ORT shape inference.",
    )
    args = parser.parse_args()
    source = args.model_dir
    destination = source / "external-data"
    destination.mkdir(exist_ok=True)
    weights = destination / args.weights_filename
    if weights.exists():
        weights.unlink()
    model = onnx.load(source / args.filename)
    if args.float_initializers_only:
        for tensor in model.graph.initializer:
            if tensor.data_type == onnx.TensorProto.FLOAT:
                onnx.external_data_helper.set_external_data(tensor, weights.name)
    else:
        onnx.external_data_helper.convert_model_to_external_data(
            model,
            all_tensors_to_one_file=True,
            location=weights.name,
            size_threshold=0,
            convert_attribute=False,
        )
    onnx.save_model(model, destination / args.filename)


if __name__ == "__main__":
    main()

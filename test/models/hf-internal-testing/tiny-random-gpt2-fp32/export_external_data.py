# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate the external-data variant used by the memory asset API tests."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--filename", default="past.onnx")
    parser.add_argument("--weights-filename", default="weights.data")
    parser.add_argument(
        "--with-adapter",
        action="store_true",
        help="Generate an adapter-enabled graph and deterministic logit-bias adapter.",
    )
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
    output_filename = args.filename
    if args.with_adapter:
        # Keep ONNX Runtime optional for the existing graph-only exporter.
        import onnxruntime as ort  # noqa: PLC0415

        config = json.loads((source / "genai_config.json").read_text())
        logits = config["model"]["decoder"]["outputs"]["logits"]
        vocab_size = config["model"]["vocab_size"]
        for node in model.graph.node:
            for index, name in enumerate(node.output):
                if name == logits:
                    node.output[index] = "base_logits"
        for tensor in model.graph.initializer:
            if tensor.name == logits:
                tensor.name = "base_logits"
        model.graph.input.append(
            onnx.helper.make_tensor_value_info("adapter_bias", onnx.TensorProto.FLOAT, [vocab_size])
        )
        model.graph.initializer.append(
            onnx.numpy_helper.from_array(np.zeros(vocab_size, dtype=np.float32), "adapter_bias")
        )
        model.graph.node.append(onnx.helper.make_node("Add", ["base_logits", "adapter_bias"], [logits]))
        bias = np.zeros(vocab_size, dtype=np.float32)
        bias[7] = 1000
        adapter = ort.AdapterFormat()
        adapter.set_adapter_version(1)
        adapter.set_model_version(1)
        adapter.set_parameters({"adapter_bias": ort.OrtValue.ortvalue_from_numpy(bias)})
        adapter.export_adapter(str(destination / "logits-bias.onnx_adapter"))
        output_filename = f"{Path(args.filename).stem}-with-adapter.onnx"
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
    onnx.save_model(model, destination / output_filename)


if __name__ == "__main__":
    main()

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os

import onnxruntime_genai as og


def run_model(model_path: str | bytes | os.PathLike, device: og.DeviceType):
    model = og.Model(model_path, device)

    tokenizer = og.Tokenizer(model)
    prompts = [
        "def is_prime(n):",
        "def compute_gcd(x, y):",
        "def binary_search(arr, x):",
    ]

    sequences = tokenizer.encode_batch(prompts)
    params = og.GeneratorParams(model)
    params.set_search_options({"max_length": 200})
    params.input_ids = sequences

    output_sequences = model.generate(params)
    output = tokenizer.decode_batch(output_sequences)
    assert output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        help="Path to the model directory",
        required=True,
    )
    args = parser.parse_args()

    device = "cuda" if og.is_cuda_available() else "cpu"
    run_model(
        args.model_path, og.DeviceType.CPU if device == "cpu" else og.DeviceType.CUDA
    )

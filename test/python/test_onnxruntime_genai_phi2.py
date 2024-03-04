# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
from _test_utils import run_subprocess
import onnxruntime_genai as og
import tempfile


def download_model(download_path: str | bytes | os.PathLike, device: str):
    # python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -p int4 -e cpu -o download_path
    command = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "-m",
        "microsoft/phi-2",
        "-p",
        "int4",
        "-e",
        device,
        "-o",
        download_path,
    ]
    run_subprocess(command).check_returncode()


def run_model(model_path: str | bytes | os.PathLike, device: og.DeviceType):
    model = og.Model(model_path, device)

    tokenizer = model.create_tokenizer()
    prompts = [
        "def is_prime(n):",
        "def compute_gcd(x, y):",
        "def binary_search(arr, x):",
    ]

    sequences = tokenizer.encode_batch(prompts)
    params = og.GeneratorParams(model)
    params.set_search_options({"max_length": 200})
    params.set_input_sequences(sequences)

    output_sequences = model.generate(params)
    output = tokenizer.decode_batch(output_sequences)
    assert output


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temp_dir:
        device = "cpu"  # FIXME: "cuda" if og.is_cuda_available() else "cpu"
        download_model(temp_dir, device)
        run_model(
            temp_dir, og.DeviceType.CPU if device == "cpu" else og.DeviceType.CUDA
        )

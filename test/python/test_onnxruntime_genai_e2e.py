# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import tempfile

import onnxruntime_genai as og
from _test_utils import run_subprocess


def download_model(
    download_path: str | bytes | os.PathLike, device: str, model_identifier: str, precision: str
):
    # python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -p int4 -e cpu -o download_path
    # Or with cuda graph enabled:
    # python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -p int4 -e cuda --extra_options enable_cuda_graph=1 -o download_path
    command = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "-m",
        model_identifier,
        "-p",
        precision,
        "-e",
        device,
        "-o",
        download_path,
    ]
    if device == "cuda":
        command.append("--extra_options")
        command.append("enable_cuda_graph=1")
    run_subprocess(command).check_returncode()


def run_model(model_path: str | bytes | os.PathLike):
    model = og.Model(model_path)

    tokenizer = og.Tokenizer(model)
    prompts = [
        "def is_prime(n):",
        "def compute_gcd(x, y):",
        "def binary_search(arr, x):",
    ]

    sequences = tokenizer.encode_batch(prompts)
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=200)
    params.try_graph_capture_with_max_batch_size(16)
    params.input_ids = sequences

    output_sequences = model.generate(params)
    output = tokenizer.decode_batch(output_sequences)
    assert output


if __name__ == "__main__":
    for model_name in ["microsoft/phi-2"]:
        for precision in ["int4", "fp32"]:
            with tempfile.TemporaryDirectory() as temp_dir:
                device = "cuda" if og.is_cuda_available() else "cpu"
                download_model(temp_dir, device, model_name, precision)
                run_model(temp_dir)

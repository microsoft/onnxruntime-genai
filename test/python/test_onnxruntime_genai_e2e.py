# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import json
import os
# import sys
# import tempfile

import onnxruntime_genai as og
# from _test_utils import download_models


# def download_model(
#     download_path: str | bytes | os.PathLike, device: str, model_path: str, precision: str
# ):
#     # python -m onnxruntime_genai.models.builder -i input_path -p int4 -e cpu -o download_path
#     # Or with cuda graph enabled:
#     # python -m onnxruntime_genai.models.builder -i input_path -p int4 -e cuda -o download_path --extra_options enable_cuda_graph=1
#     command = [
#         sys.executable,
#         "-m",
#         "onnxruntime_genai.models.builder",
#         "-i",
#         model_path,
#         "-o",
#         download_path,
#         "-p",
#         precision,
#         "-e",
#         device,
#     ]
#     models_not_compatible_with_cuda_graph = {"microsoft/Phi-3-mini-128k-instruct"}
#     if device == "cuda" and precision != "fp32" and model_path not in models_not_compatible_with_cuda_graph:
#         command.append("--extra_options")
#         command.append("enable_cuda_graph=1")
#     run_subprocess(command).check_returncode()


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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        required=True,
        help="List of model paths to run. Pass as `json.dumps(model_paths)` to this argument.",
    )

    args = parser.parse_args()
    args.models = json.loads(args.models)
    return args


if __name__ == "__main__":
    args = get_args()
    for model_path in args.models:
        run_model(model_path)

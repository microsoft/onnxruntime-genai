# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import argparse
import json
import os
import logging

import onnxruntime_genai as og
from _test_utils import get_ci_path

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG
)
log = logging.getLogger("onnxruntime-genai-tests")


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
    params.set_search_options(batch_size=3, max_length=200)

    generator = og.Generator(model, params)
    generator.append_tokens(sequences)
    while not generator.is_done():
        generator.generate_next_token()
    
    for i in range(3):
        assert generator.get_sequence(i) is not None


# TODO: enable once ORT extensions fixes the audio features calculation
def run_whisper(model_path: str | os.PathLike):
    log.debug("Running Whisper Python E2E Test")

    ci_data_path = get_ci_path()
    if not os.path.exists(ci_data_path):
        return

    num_beams = 1
    (audio_path, expected_transcription) = (
        os.path.join(os.path.abspath(__file__), "..", "test_models", "audios", "1272-141231-0002.mp3"),
        "The cut on his chest still dripping blood. The ache of his overstrain dyes. Even the soaring arena around him with thousands of spectators, retrievalidates not worth thinking about.",
    )
    command = [
        sys.executable,
        os.path.join(os.path.abspath(__file__), "..", "..", "examples", "python", "whisper.py"),
        "-m",
        os.path.join(ci_data_path, "onnx", "whisper-tiny-fp16" if og.is_cuda_available() else "whisper-tiny-fp32"),
        "-e",
        "cuda" if og.is_cuda_available() else "cpu",
        "-b",
        str(num_beams),
        "-a",
        audio_path,
        "-o",
        expected_transcription,
        "-ci",
    ]
    run_subprocess(command, cwd=cwd, log=log).check_returncode()


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
        try:
            log.info(f"Running {model_path}")
            run_model(model_path)
        except Exception as e:
            log.error(e)
            log.error(f"Failed to run {model_path}", exc_info=True)

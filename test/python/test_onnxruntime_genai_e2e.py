# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import logging
import os
import sys

import onnxruntime_genai as og
from _test_utils import download_model, get_ci_data_path, run_subprocess

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
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


def run_whisper():
    log.debug("Running Whisper Python E2E Test")

    cwd = os.path.dirname(os.path.abspath(__file__))
    ci_data_path = get_ci_data_path()
    if not os.path.exists(ci_data_path):
        return

    num_beams = 5
    (audio_path, expected_transcription) = (
        os.path.join(cwd, "..", "test_models", "audios", "1272-141231-0002.mp3"),
        "The cut on his chest is still dripping blood. The ache of his overstrained eyes. Even the soaring arena around him with thousands of spectators, retrievalidies not worth thinking about.",
    )

    for precision, execution_provider in [("fp16", "cuda"), ("fp32", "cuda"), ("fp32", "cpu")]:
        # Generate model via model builder
        built_model = os.path.join(cwd, "..", "test_models", f"whisper-tiny-{precision}-{execution_provider}")
        download_model(model_name="openai/whisper-tiny", input_path="", output_path=built_model, precision=precision, device=execution_provider, one_layer=False)

        # Get prebuilt model from CI
        ci_model = os.path.join(ci_data_path, "onnx", f"whisper-tiny-{precision}-{execution_provider}")
        for model in [built_model, ci_model]:
            # Conditions for skipping test
            if execution_provider == "cuda" and not og.is_cuda_available():
                continue
            if not os.path.exists(model):
                continue

            command = [
                sys.executable,
                os.path.join(cwd, "..", "..", "examples", "python", "whisper.py"),
                "-m",
                model,
                "-e",
                execution_provider,
                "-b",
                str(num_beams),
                "-a",
                audio_path,
                "-o",
                expected_transcription,
                "--non_interactive",
            ]
            run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_tool_calling():
    log.debug("Running tool calling Python E2E Tests")

    cwd = os.path.dirname(os.path.abspath(__file__))
    tool_call_models = [("qwen-2.5-0.5b", "<tool_call>", "</tool_call>")]

    # Runtime settings
    max_length = 256
    user_prompt = "What is the weather in Redmond, WA?"
    response_format = "lark_grammar"

    for (model_name, tool_call_start, tool_call_end) in tool_call_models:
        for (precision, execution_provider) in [("int4", "cpu")]: # TODO: add ("int4", "cuda"), ("int4", "dml") in CIs later
            model_path = os.path.join(cwd, "..", "test_models", model_name, precision, execution_provider)
            if not os.path.exists(model_path): continue

            # Run special_tokens.py to mark tool call token ids as special
            command = [
                sys.executable,
                os.path.join(cwd, "special_tokens.py"),
                "-p",
                os.path.join(model_path, "tokenizer.json"),
                "-s",
                tool_call_start,
                "-e",
                tool_call_end,
            ]
            run_subprocess(command, cwd=cwd, log=log).check_returncode()

            # Run model-qa.py for inference
            command = [
                sys.executable,
                os.path.join(cwd, "..", "..", "examples", "python", "model-qa.py"),
                "-m",
                model_path,
                "-e",
                execution_provider,
                "--max_length",
                str(max_length),
                "--response_format",
                response_format,
                "--tools_file",
                os.path.join(cwd, "..", "test_models", "tool-definitions", "weather.json"),
                "--tool_call_start",
                tool_call_start,
                "--tool_call_end",
                tool_call_end,
                "--user_prompt",
                user_prompt,
                "--tool_output",
                "--non_interactive",
                "--verbose",
            ]
            run_subprocess(command, cwd=cwd, log=log).check_returncode()

            # Run model_qa.cpp for inference
            command = [
                os.path.join(cwd, "..", "..", "examples", "c", "build", f"{'Release' if sys.platform.startswith('win') else ''}", f"model_qa{'.exe' if sys.platform.startswith('win') else ''}"),
                "-m",
                model_path,
                "-e",
                execution_provider,
                "--max_length",
                str(max_length),
                "--response_format",
                response_format,
                "--tools_file",
                os.path.join(cwd, "..", "test_models", "tool-definitions", "weather.json"),
                "--tool_call_start",
                tool_call_start,
                "--tool_call_end",
                tool_call_end,
                "--user_prompt",
                user_prompt,
                "--tool_output",
                "--non_interactive",
                "--verbose",
            ]
            run_subprocess(command, cwd=cwd, log=log).check_returncode()


def run_nemotron_speech():
    """Run Nemotron Speech Streaming ASR E2E test by invoking the nemotron_speech.py example."""
    log.debug("Running Nemotron Speech Python E2E Test")

    # Look for nemotron speech model in test_models directory
    cwd = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(cwd, "..", "test_models", "nemotron-speech-streaming")
    model_path = "/datadisks/disk1/jiafa/accuracy/onnxruntime-genai/tools/nemotron_export/onnx_models_cpu_int4_1"
    if not os.path.exists(model_path):
        log.info(f"Nemotron speech model not found at {model_path}, skipping E2E test.")
        return

    # Look for a test audio file
    audio_path = os.path.join(cwd, "..", "test_models", "audios", "1272-141231-0002.mp3")
    if not os.path.exists(audio_path):
        log.info(f"Test audio file not found at {audio_path}, skipping E2E test.")
        return

    command = [
        sys.executable,
        os.path.join(cwd, "..", "..", "examples", "python", "nemotron_speech.py"),
        "--model_path",
        model_path,
        "--audio_file",
        audio_path,
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
    '''
    args = get_args()
    for model_path in args.models:
        try:
            log.info(f"Running {model_path}")
            run_model(model_path)
        except Exception as e:
            log.error(e)
            log.error(f"Failed to run {model_path}", exc_info=True)

    # Run Whisper E2E tests
    run_whisper()
    '''

    # Run Nemotron Speech E2E tests
    run_nemotron_speech()

    # Run tool calling E2E tests
    # run_tool_calling()

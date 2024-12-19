# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Union


def is_windows():
    return sys.platform.startswith("win")


def run_subprocess(
    args: List[str],
    cwd: Optional[Union[str, bytes, os.PathLike]] = None,
    capture: bool = False,
    dll_path: Optional[Union[str, bytes, os.PathLike]] = None,
    shell: bool = False,
    env: Dict[str, str] = {},
    log: Optional[logging.Logger] = None,
):
    if log:
        log.info(f"Running subprocess in '{cwd or os.getcwd()}'\n{args}")
    user_env = os.environ.copy()
    user_env.update(env)
    if dll_path:
        if is_windows():
            user_env["PATH"] = dll_path + os.pathsep + user_env["PATH"]
        else:
            if "LD_LIBRARY_PATH" in user_env:
                user_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                user_env["LD_LIBRARY_PATH"] = dll_path

    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (None, None)
    completed_process = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=stdout,
        stderr=stderr,
        env=user_env,
        shell=shell,
    )

    if log:
        log.debug(
            "Subprocess completed. Return code=" + str(completed_process.returncode)
        )
    return completed_process


def get_model_paths():
    hf_paths = {
        "phi-2": "microsoft/phi-2",
        # "phi-3-mini": "microsoft/Phi-3-mini-128k-instruct",
    }

    ci_data_path = os.path.join("/", "data", "ortgenai", "pytorch")
    if not os.path.exists(ci_data_path):
        return {}, hf_paths

    # Note: If a model has over 4B parameters, please add a quantized version
    # to `ci_paths` instead of `hf_paths` to reduce file size and testing time.
    ci_paths = {
        "llama-2": os.path.join(ci_data_path, "Llama-2-7B-Chat-GPTQ"),
        "llama-3": os.path.join(ci_data_path, "Meta-Llama-3-8B-AWQ"),
        "mistral-v0.2": os.path.join(ci_data_path, "Mistral-7B-Instruct-v0.2-GPTQ"),
        "phi-2": os.path.join(ci_data_path, "phi2"),
        "gemma-2b": os.path.join(ci_data_path, "gemma-1.1-2b-it"),
        "gemma-7b": os.path.join(ci_data_path, "gemma-7b-it-awq"),
        "phi-3-mini": os.path.join(ci_data_path, "phi3-mini-128k-instruct"),
        "gemma-2-2b": os.path.join(ci_data_path, "gemma-2-2b-it"),
        "llama-3.2": os.path.join(ci_data_path, "llama-3.2b-1b-instruct"),
        "qwen-2.5": os.path.join(ci_data_path, "qwen2.5-0.5b-instruct"),
        "nemotron-mini": os.path.join(ci_data_path, "nemotron-mini-4b"),
    }

    return ci_paths, hf_paths


def download_model(model_name, input_path, output_path, precision, device, one_layer=True):
    command = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
    ]

    if model_name is not None:
        # If model_name is provided:
        # python -m onnxruntime_genai.models.builder -m <model_name> -o <output_path> -p <precision> -e <device>
        command += ["-m", model_name]
    elif input_path != "":
        # If input_path is provided:
        # python -m onnxruntime_genai.models.builder -i <input_path> -o <output_path> -p <precision> -e <device>
        command += ["-i", input_path]
    else:
        raise Exception("Either `model_name` or `input_path` can be provided for PyTorch models, not both.")

    command += [
        "-o",
        output_path,
        "-p",
        precision,
        "-e",
        device,
    ]

    extra_options = ["--extra_options", "include_hidden_states=true"]
    if device == "cpu" and precision == "int4":
        extra_options += ["int4_accuracy_level=4"]
    if one_layer:
        extra_options += ["num_hidden_layers=1"]
    if len(extra_options) > 1:
        command += extra_options

    run_subprocess(command).check_returncode()


def download_models(download_path, precision, device):
    ci_paths, hf_paths = get_model_paths()
    output_paths = []
    
    # python -m onnxruntime_genai.models.builder -i <input_path> -o <output_path> -p <precision> -e <device>
    for model_name, input_path in ci_paths.items():
        output_path = os.path.join(download_path, model_name, precision, device)
        if not os.path.exists(output_path):
            download_model(None, input_path, output_path, precision, device)
            output_paths.append(output_path)

    # python -m onnxruntime_genai.models.builder -m <model_name> -o <output_path> -p <precision> -e <device>
    for model_name, hf_name in hf_paths.items():
        output_path = os.path.join(download_path, model_name, precision, device)
        if not os.path.exists(output_path):
            download_model(hf_name, "", output_path, precision, device)
            output_paths.append(output_path)

    return output_paths

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
    model_paths = {
        # "llama-2": "meta-llama/Llama-2-7b-hf",
        # "mistral-v0.1": "mistralai/Mistral-7B-v0.1",
        # "phi-2": "microsoft/phi-2",
        # "gemma-2b": "google/gemma-2b",
        # "gemma-7b": "google/gemma-7b",
        "phi-3-mini": "microsoft/Phi-3-mini-128k-instruct",
    }
    return model_paths


def download_model(input_path, output_path, precision, device):
    # python -m onnxruntime_genai.models.builder -i <input_path> -o <output_path> -p <precision> -e <device>
    command = [
        sys.executable,
        "-m",
        "onnxruntime_genai.models.builder",
        "-i",
        input_path,
        "-o",
        output_path,
        "-p",
        precision,
        "-e",
        device,
    ]
    if device == "cpu" and precision == "int4":
        command += ["--extra_options", "int4_accuracy_level=4"]

    run_subprocess(command).check_returncode()


def download_models(download_path, precision, device):
    # python -m onnxruntime_genai.models.builder -i <input_path> -o <output_path> -p <precision> -e <device>
    model_paths = get_model_paths()
    output_paths = []
    
    for model_name, input_path in model_paths.items():
        output_path = os.path.join(download_path, model_name, precision, device)
        if not os.path.exists(output_path):
            download_model(input_path, output_path, precision, device)
            output_paths.append(output_path)

    return output_paths

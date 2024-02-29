# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Union

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG
)
log = logging.getLogger("onnxruntime-genai-tests")


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


def run_onnxruntime_genai_api_tests(
    cwd: Union[str, bytes, os.PathLike],
    log: logging.Logger,
    test_models: Union[str, bytes, os.PathLike],
):
    log.debug("Running: ONNX Runtime GenAI API Tests")

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-sv",
        "test_onnxruntime_genai_api.py",
        "--test_models",
        test_models,
    ]

    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", help="Path to the current working directory")
    parser.add_argument(
        "--test_models", help="Path to the test_models directory", required=True
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    log.info("Running onnxruntime-genai tests pipeline")

    run_onnxruntime_genai_api_tests(
        os.path.abspath(args.cwd), log, os.path.abspath(args.test_models)
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())

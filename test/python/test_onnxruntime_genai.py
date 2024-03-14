# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import logging
import os
import pathlib
import sys
import sysconfig
from typing import Union

import onnxruntime_genai as og
from _test_utils import download_models, run_subprocess

logging.basicConfig(
    format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG
)
log = logging.getLogger("onnxruntime-genai-tests")


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


def run_onnxruntime_genai_e2e_tests(
    cwd: Union[str, bytes, os.PathLike],
    log: logging.Logger,
):
    log.debug("Running: ONNX Runtime GenAI E2E Tests")

    command = [sys.executable, "test_onnxruntime_genai_e2e.py"]
    run_subprocess(command, cwd=cwd, log=log).check_returncode()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cwd",
        help="Path to the current working directory",
        default=pathlib.Path(__file__).parent.resolve().absolute(),
    )
    parser.add_argument(
        "--test_models",
        help="Path to the test_models directory",
        default=pathlib.Path(__file__).parent.parent.resolve().absolute()
        / "test_models",
    )
    parser.add_argument(
        "--e2e",
        help="Whether to run e2e tests. If not specified e2e tests will not run.",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    log.info("Running onnxruntime-genai tests pipeline")

    if not args.e2e:
        if not (
            sysconfig.get_platform().endswith("arm64") or sys.version_info.minor < 8
        ):
            download_models(os.path.abspath(args.test_models), "cpu")
            if og.is_cuda_available():
                download_models(
                    os.path.abspath(args.test_models),
                    "cuda",
                )

        run_onnxruntime_genai_api_tests(
            os.path.abspath(args.cwd), log, os.path.abspath(args.test_models)
        )

    else:
        run_onnxruntime_genai_e2e_tests(os.path.abspath(args.cwd), log)

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import shutil
import subprocess
import sys


def is_windows():
    """Check if the current platform is Windows."""
    return sys.platform.startswith("win")


def is_linux():
    """Check if the current platform is Linux."""
    return sys.platform.startswith("linux")


def platform():
    """Get the current platform."""
    return sys.platform


def resolve_executable_path(command_or_path: str):
    """Resolve the full path to an executable given a command or path.

    Args:
        command_or_path (str): The command or path to resolve.

    Returns:
        str: The full path to the executable.
    """
    if command_or_path and command_or_path.strip():
        executable_path = shutil.which(command_or_path)
        if executable_path is None:
            raise RuntimeError(f"Failed to resolve executable path for '{command_or_path}'.")
        return os.path.abspath(executable_path)
    else:
        return None


def run_subprocess(
    args: list[str],
    cwd: str | bytes | os.PathLike | None = None,
    capture: bool = False,
    shell: bool = False,
    env: dict[str, str] = {},
):
    """Run a subprocess and return the result.

    Args:
        args: The command to run and its arguments.
        cwd: The working directory. Defaults to None.
        capture: Whether to capture the output. Defaults to False.
        shell: Whether to use the shell. Defaults to False.
        env: The environment variables. Defaults to {}.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess.
    """
    user_env = os.environ.copy()
    user_env.update(env)
    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (None, None)
    return subprocess.run(args, cwd=cwd, check=True, stdout=stdout, stderr=stderr, env=user_env, shell=shell)


def update_submodules():
    """Update the git submodules."""
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"]).check_returncode()


def build(
    skip_wheel: bool = False,
    cuda_home: str | bytes | os.PathLike | None = None,
    cudnn_home: str | bytes | os.PathLike | None = None,
    cmake_generator: str | None = None,
):
    """Generates the CMake build tree and builds the project.

    Args:
        skip_wheel: Whether to skip building the Python wheel. Defaults to False.
    """
    command = [resolve_executable_path("cmake")]
    if cmake_generator:
        command += ["-G", cmake_generator]
    build_wheel = "OFF" if skip_wheel else "ON"
    if is_windows():
        if not cmake_generator:
            command += ["-G", "Visual Studio 17 2022"]
        command += ["-S", ".", "-B", "build", f"-DBUILD_WHEEL={build_wheel}"]
        run_subprocess(command).check_returncode()
        make_command = ["cmake", "--build", ".", "--config", "Release"]
        run_subprocess(make_command, cwd="build").check_returncode()
    elif is_linux():
        cuda_arch = 80
        cuda_compiler = None
        env = {}
        if cuda_home or os.environ.get("CUDA_HOME"):
            env["CUDA_HOME"] = cuda_home if cuda_home else os.environ.get("CUDA_HOME")
            cuda_compiler = f"{env['CUDA_HOME']}/bin/nvcc"
        if cudnn_home or os.environ.get("CUDNN_HOME"):
            env["CUDNN_HOME"] = cudnn_home if cudnn_home else os.environ.get("CUDNN_HOME")

        command += [
            "-S",
            ".",
            "-B",
            "build",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            f"-DCMAKE_CUDA_COMPILER={cuda_compiler}",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
            "-DUSE_CXX17=ON",
            "-DUSE_CUDA=ON",
            f"-DBUILD_WHEEL={build_wheel}",
        ]
        run_subprocess(command, env=env).check_returncode()
        make_command = ["make"]
        run_subprocess(make_command, cwd="build", env=env).check_returncode()
    else:
        raise OSError(f"Unsupported platform {platform()}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI Build Driver.",
    )
    parser.add_argument(
        "--cmake_generator",
        choices=[
            "MinGW Makefiles",
            "Ninja",
            "NMake Makefiles",
            "Unix Makefiles",
            "Visual Studio 17 2022",
            "Xcode",
        ],
        default=None,
        help="Specify the generator that CMake invokes.",
    )
    parser.add_argument(
        "--cuda_home",
        help="Path to CUDA home."
        "Read from CUDA_HOME environment variable if --use_cuda is true and "
        "--cuda_home is not specified.",
    )
    parser.add_argument(
        "--cudnn_home",
        help="Path to CUDNN home. "
        "Read from CUDNN_HOME environment variable if --use_cuda is true and "
        "--cudnn_home is not specified.",
    )
    parser.add_argument("--skip_wheel", action="store_true", help="Skip building the Python wheel.")
    args = parser.parse_args()

    update_submodules()
    build(
        skip_wheel=args.skip_wheel,
        cuda_home=args.cuda_home,
        cudnn_home=args.cudnn_home,
        cmake_generator=args.cmake_generator,
    )

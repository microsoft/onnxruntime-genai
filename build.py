# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import warnings


class BaseError(Exception):
    """Base class for errors originating from build.py."""


class BuildError(BaseError):
    """Error from running build steps."""

    def __init__(self, *messages):
        super().__init__("\n".join(messages))


def is_windows():
    """Check if the current platform is Windows."""
    return sys.platform.startswith("win")


def is_linux():
    """Check if the current platform is Linux."""
    return sys.platform.startswith("linux")

def is_mac():
    """Check if the current platform is MacOS"""
    return sys.platform.startswith("darwin")


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


def validate_cuda_home(cuda_home: str | bytes | os.PathLike | None):
    """Validate the CUDA home paths."""
    validated_cuda_home = cuda_home if cuda_home else os.getenv("CUDA_HOME")
    if not validated_cuda_home and is_windows():
        validated_cuda_home = os.getenv("CUDA_PATH")

    cuda_home_valid = os.path.exists(validated_cuda_home) if validated_cuda_home else False

    if not cuda_home_valid:
        raise RuntimeError(
            "cuda_home paths must be specified and valid.",
            "cuda_home='{}' valid={}.".format(
                cuda_home, cuda_home_valid
            ),
        )

    return validated_cuda_home


def build(
    skip_wheel: bool = False,
    use_cuda: bool | None = None,
    use_dml: bool | None = None,
    cuda_home: str | bytes | os.PathLike | None = None,
    cmake_generator: str | None = None,
    ort_home: str | bytes | os.PathLike | None = None,
    skip_csharp: bool = False,
    build_dir: str | bytes | os.PathLike | None = None,
    parallel: bool = False,
    config: str = "RelWithDebInfo",
    ios: bool | None = None,
    ios_sysroot: str | None = None,
    osx_arch: str | None = None,
    apple_deployment_target: str | None = None,
    ios_toolchain_file: str | None = None,
):
    """Generates the CMake build tree and builds the project.

    Args:
        skip_wheel: Whether to skip building the Python wheel. Defaults to False.
    """
    if not is_windows() and not is_linux() and not is_mac():
        raise OSError(f"Unsupported platform {platform()}.")

    if cuda_home and not use_cuda:
        use_cuda = True

    cuda_home = validate_cuda_home(cuda_home) if use_cuda else None

    command = [resolve_executable_path("cmake")]

    if cmake_generator:
        command += ["-G", cmake_generator]

    if is_windows():
        if not cmake_generator:
            command += ["-G", "Visual Studio 17 2022", "-A", "x64"]
        if cuda_home:
            toolset = "host=x64" + ",cuda=" + cuda_home
            command += ["-T", toolset]
    command += [f"-DCMAKE_BUILD_TYPE={config}"]

    build_wheel = "OFF" if skip_wheel else "ON"
    build_dir = os.path.abspath(build_dir) if build_dir else os.path.join(os.getcwd(), "build")

    command += [
        "-S",
        ".",
        "-B",
        build_dir,
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DUSE_CXX17=ON",
        "-DUSE_CUDA=ON" if cuda_home else "-DUSE_CUDA=OFF",
        "-DUSE_DML=ON" if use_dml else "-DUSE_DML=OFF",
        f"-DBUILD_WHEEL={build_wheel}",
    ]

    if ort_home:
        ort_home = os.path.abspath(ort_home)
        if not os.path.isdir(ort_home):
            raise RuntimeError(f"ORT_HOME '{ort_home}' does not exist.")
        command += [f"-DORT_HOME={ort_home}"]

    cuda_compiler = None
    env = {}
    if cuda_home:
        cuda_arch = 80
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = os.path.join(env["CUDA_HOME"], "bin") + os.pathsep + os.environ["PATH"]
        cuda_compiler = os.path.join(env["CUDA_HOME"], "bin", "nvcc")
        command += [f"-DCMAKE_CUDA_COMPILER={cuda_compiler}", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"]
        
    # ios build
    if ios:
        if is_mac():
            needed_args = [
                ios_sysroot,
                osx_arch,
                apple_deployment_target,
            ]
            arg_names = [
                "--ios_sysroot          " +
                "<the location or name of the macOS platform SDK>",
                "--osx_arch             " +
                "<the Target specific architectures for iOS>",
                "--apple_deploy_target  " +
                "<the minimum version of the target platform>",
            ]
            if not all(_ is not None for _ in needed_args):
                raise BuildError(
                    "iOS build on MacOS canceled due to missing arguments: " +
                    ', '.join(
                        val for val, cond in zip(arg_names, needed_args)
                        if not cond))
            command += [
                "-DCMAKE_SYSTEM_NAME=iOS",
                "-DENABLE_TESTS=OFF",
                "-DCMAKE_OSX_SYSROOT=" + ios_sysroot,
                "-DCMAKE_OSX_ARCHITECTURES=" + osx_arch,
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + apple_deployment_target,
                "-DENABLE_PYTHON=OFF",
                "-DCMAKE_TOOLCHAIN_FILE=" + (
                    ios_toolchain_file if ios_toolchain_file
                    else "cmake/genai_ios.toolchain.cmake")
            ]
        # else:
            # cross compile for ios on Linux

    run_subprocess(command, env=env).check_returncode()
    make_command = ["cmake", "--build", ".", "--config", config]
    if parallel:
        make_command.append("--parallel")
    run_subprocess(make_command, cwd=build_dir, env=env).check_returncode()

    if not skip_csharp:
        if not is_windows():
            warnings.warn("C# API is only supported on Windows.", UserWarning)
            return

        dotnet = resolve_executable_path("dotnet")
        configuration = f"/p:Configuration={config}"
        platorm = "/p:Platform=Any CPU"
        # Tests folder does not have a sln file. We use the csproj file to build and test.
        # The csproj file requires the platform to be AnyCPU (not "Any CPU")
        platform_for_tests_csproj = "/p:Platform=AnyCPU"
        native_lib_path = f"/p:NativeBuildOutputDir={os.path.join(build_dir, config)}"

        # Build the library
        properties = [configuration, platorm, native_lib_path]
        csharp_build_command = [
            dotnet,
            "build",
            ".",
        ]
        run_subprocess(
            csharp_build_command + properties, cwd=os.path.join("src", "csharp")
        ).check_returncode()

        # Build the tests and run them
        tests_properties = [configuration, platform_for_tests_csproj, native_lib_path]
        if ort_home:
            tests_properties += [f"/p:OrtHome={ort_home}"]
        run_subprocess(
            csharp_build_command + tests_properties, cwd=os.path.join("test", "csharp")
        ).check_returncode()
        run_subprocess(
            [dotnet, "test"] + tests_properties, cwd=os.path.join("test", "csharp")
        ).check_returncode()


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
        "Read from CUDA_HOME or CUDA_PATH environment variable if not specified. Not read if --use_cuda is not specified.",
    )
    parser.add_argument("--skip_wheel", action="store_true", help="Skip building the Python wheel.")
    parser.add_argument("--ort_home", default=None, help="Root directory of onnxruntime.")
    parser.add_argument("--skip_csharp", action="store_true", help="Skip building the C# API.")
    parser.add_argument("--build_dir", default=None, help="Path to output directory.")
    parser.add_argument("--use_cuda", action="store_true", help="Whether to use CUDA. Default is to not use cuda.")
    parser.add_argument("--use_dml", action="store_true", help="Whether to use DML. Default is to not use DML.")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel build.")
    parser.add_argument(
        "--config",
        default="RelWithDebInfo",
        type=str,
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build.",
    )

    # iOS build options
    parser.add_argument("--ios", action="store_true", help="build for ios")
    parser.add_argument(
        "--ios_sysroot",
        default="",
        help="Specify the location name of the macOS platform SDK to be used",
    )
    parser.add_argument(
        "--ios_toolchain_file",
        default="",
        help="Path to ios toolchain file, "
        "or cmake/genai_ios.toolchain.cmake will be used",
    )
    parser.add_argument(
        "--osx_arch",
        type=str,
        help="Specify the Target specific architectures for iOS "
        "This is only supported on MacOS",
    )
    parser.add_argument(
        "--apple_deployment_target",
        type=str,
        help="Specify the minimum version of the target platform "
        "This is only supported on MacOS",
    )
    args = parser.parse_args()

    update_submodules()
    build(
        skip_wheel=args.skip_wheel,
        use_cuda=args.use_cuda,
        use_dml=args.use_dml,
        cuda_home=args.cuda_home,
        cmake_generator=args.cmake_generator,
        ort_home=args.ort_home,
        skip_csharp=args.skip_csharp,
        build_dir=args.build_dir,
        parallel=args.parallel,
        config=args.config,
        ios=args.ios,
        ios_sysroot=args.ios_sysroot,
        ios_toolchain_file=args.ios_toolchain_file,
        osx_arch=args.osx_arch,
        apple_deployment_target=args.apple_deployment_target,
    )

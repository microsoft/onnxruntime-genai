# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import shlex
import shutil
import sys
import textwrap

from pathlib import Path

REPO_ROOT = Path(__file__).parent
sys.path.append(str(REPO_ROOT / "tools" / "python"))
import util  # ./tools/python/util noqa: E402


log = util.get_logger("build.py")


def _path_from_env_var(env_var: str):
    env_var_value = os.environ.get(env_var)
    return Path(env_var_value) if env_var_value is not None else None


def _parse_args():
    class Parser(argparse.ArgumentParser):
        # override argument file line parsing behavior - allow multiple arguments per line and handle quotes
        def convert_arg_line_to_args(self, arg_line):
            return shlex.split(arg_line)

    class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = Parser(
        description="ONNX Runtime GenAI Build Driver.",
        epilog=textwrap.dedent("""
            There are 3 phases which can be individually selected.

            The update (--update) phase will run CMake to generate makefiles.
            The build (--build) phase will build all projects.
            The test (--test) phase will run all unit tests.

            Default behavior is --update --build --test for native architecture builds.
            Default behavior is --update --build for cross-compiled builds.

            If phases are explicitly specified only those phases will be run.
            E.g., run with --build to rebuild without running the update or test phases.
            """),
        # files containing arguments can be specified on the command line with "@<filename>" and the arguments within
        # will be included at that point
        fromfile_prefix_chars="@",
        formatter_class=HelpFormatter,
    )

    parser.add_argument(
        "--build_dir",
        type=Path,
        # We set the default programmatically as it needs to take into account whether we're cross-compiling
        help="Path to the build directory. Defaults to 'build/<target platform>'. "
             "The build configuration will be a subdirectory of the build directory. e.g. build/Linux/Debug",
    )

    parser.add_argument(
        "--config",
        default="RelWithDebInfo",
        type=str,
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration to build.")

    # Build phases.
    parser.add_argument("--update", action="store_true", help="Update makefiles.")
    parser.add_argument("--build", action="store_true", help="Build.")
    parser.add_argument("--test", action="store_true", help="Run tests.")
    parser.add_argument(
        "--clean", action="store_true", help="Run 'cmake --build --target clean' for the selected config/s."
    )

    parser.add_argument("--skip_tests", action="store_true", help="Skip all tests. Overrides --test.")
    parser.add_argument("--skip_wheel", action="store_true", help="Skip building the Python wheel.")

    # Default to not building the language bindings
    parser.add_argument("--build_csharp", action="store_true", help="Build the C# API.")
    parser.add_argument("--build_java", action="store_true", help="Build Java bindings.")

    parser.add_argument("--parallel", action="store_true", help="Enable parallel build.")

    # CI's sometimes explicitly set the path to the CMake and CTest executables.
    parser.add_argument("--cmake_path", default="cmake", type=Path, help="Path to the CMake program.")
    parser.add_argument("--ctest_path", default="ctest", type=Path, help="Path to the CTest program.")

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
        default=("Visual Studio 17 2022" if util.is_windows() else "Unix Makefiles"),
        help="Specify the generator that CMake invokes.",
    )
    parser.add_argument(
        "--cmake_extra_defines",
        nargs="+",
        action="append",
        help="Extra definitions to pass to CMake during build system "
        "generation. These are just CMake -D options without the leading -D.",
    )

    parser.add_argument("--ort_home", default=None, type=Path, help="Root directory of onnxruntime.")

    parser.add_argument("--use_cuda", action="store_true", help="Whether to use CUDA. Default is to not use cuda.")
    parser.add_argument(
        "--cuda_home",
        type=Path,
        help="Path to CUDA home. Read from CUDA_HOME or CUDA_PATH environment variable if not specified."
             "Used when --use_cuda is specified.",
    )

    parser.add_argument("--use_rocm", action="store_true", help="Whether to use ROCm. Default is to not use rocm.")

    parser.add_argument("--use_webgpu", action="store_true", help="Whether to use WebGpu. Default is to not use WebGpu.")

    parser.add_argument("--use_dml", action="store_true", help="Whether to use DML. Default is to not use DML.")

    parser.add_argument("--use_guidance", action="store_true", help="Whether to add guidance support. Default is False.")
    
    # The following options are mutually exclusive (cross compiling options such as android, ios, etc.)
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument("--android", action="store_true", help="Build for Android")
    platform_group.add_argument("--ios", action="store_true", help="Build for iOS")
    platform_group.add_argument(
        "--macos",
        choices=["MacOSX", "Catalyst"],
        help="Specify the target platform for macOS build. Only specify this argument when --build_apple_framework is present.",
    )

    # Android options
    parser.add_argument(
        "--android_abi",
        default="arm64-v8a",
        choices=["armeabi-v7a", "arm64-v8a", "x86", "x86_64"],
        help="Specify the target Android Application Binary Interface (ABI)",
    )
    parser.add_argument("--android_api", type=int, default=27,
                        help="Android API Level. Default is 27 (Android 8.1, released in 2017).")
    parser.add_argument(
        "--android_home", type=Path, default=_path_from_env_var("ANDROID_HOME"), help="Path to the Android SDK."
    )
    parser.add_argument(
        "--android_ndk_path",
        type=Path,
        default=_path_from_env_var("ANDROID_NDK_HOME"),
        help="Path to the Android NDK. Typically `<Android SDK>/ndk/<ndk_version>`.",
    )
    parser.add_argument("--android_run_emulator", action="store_true",
                        help="Create/start an Android emulator to run the test application. "
                             "Requires --android, --build_java and --android_abi=x86_64.")

    # iOS build options
    parser.add_argument(
        "--apple_sysroot",
        default="",
        help="Specify the location name of the macOS platform SDK to be used",
    )
    parser.add_argument(
        "--osx_arch",
        type=str,
        help="Specify the Target specific architectures for iOS "
        "This is only supported on MacOS host",
    )
    parser.add_argument(
        "--apple_deploy_target",
        type=str,
        help="Specify the minimum version of the target platform "
        "This is only supported on MacOS host",
    )

    parser.add_argument(
        "--build_apple_framework", action="store_true", help="Build a macOS/iOS framework for the ONNXRuntime."
    )

    parser.add_argument(
        "--arm64",
        action="store_true",
        help="[cross-compiling] Create ARM64 makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed",
    )

    parser.add_argument(
        "--arm64ec",
        action="store_true",
        help="[cross-compiling] Create ARM64EC makefiles. Requires --update and no existing cache "
        "CMake setup. Delete CMakeCache.txt if needed",
    )

    return parser.parse_args()


def _resolve_executable_path(command_or_path: Path, resolution_failure_allowed: bool = False):
    """
    Returns the absolute path of an executable.
    If `resolution_failure_allowed` is True, returns None if the executable path cannot be found.
    """
    executable_path = shutil.which(str(command_or_path))
    if executable_path is None:
        if resolution_failure_allowed:
            return None
        else:
            raise ValueError(f"Failed to resolve executable path for '{command_or_path}'.")

    return Path(executable_path)


def _validate_build_dir(args: argparse.Namespace):
    if not args.build_dir:
        target_sys = platform.system()

        # override if we're cross-compiling
        # TODO: Add ios and arm64 support
        if args.android:
            target_sys = "Android"
        elif platform.system() == "Darwin":
            # also tweak build directory name for mac builds
            target_sys = "macOS"

        args.build_dir = Path("build") / target_sys

    # set to a config specific build dir. it should exist unless we're creating the cmake setup
    is_strict = not args.update
    args.build_dir = args.build_dir.resolve(strict=is_strict) / args.config


def _validate_cuda_args(args: argparse.Namespace):
    if args.cuda_home:
        # default use_cuda to True if cuda_home is specified
        args.use_cuda = True

    if args.use_cuda:
        cuda_home = args.cuda_home if args.cuda_home else _path_from_env_var("CUDA_HOME")
        if not cuda_home and util.is_windows():
            cuda_home = _path_from_env_var("CUDA_PATH")

        cuda_home_valid = cuda_home.exists() if cuda_home else False

        if not cuda_home_valid:
            raise RuntimeError(
                f"cuda_home paths must be specified and valid. cuda_home='{cuda_home}' valid={cuda_home_valid}."
            )

        args.cuda_home = cuda_home.resolve(strict=True)


def _validate_android_args(args: argparse.Namespace):
    if args.android:
        if not args.android_home:
            raise ValueError("--android_home is required to build for Android")

        if not args.android_ndk_path:
            raise ValueError("--android_ndk_path is required to build for Android")

        args.android_home = args.android_home.resolve(strict=True)
        args.android_ndk_path = args.android_ndk_path.resolve(strict=True)

        if not args.android_home.is_dir() or not args.android_ndk_path.is_dir():
            raise ValueError("Android home and NDK paths must be directories.")

        # auto-adjust the cmake generator for cross-compiling Android
        original_cmake_generator = args.cmake_generator
        if original_cmake_generator not in ["Ninja", "Unix Makefiles"]:
            if _resolve_executable_path("ninja", resolution_failure_allowed=True) is not None:
                args.cmake_generator = "Ninja"
            elif _resolve_executable_path("make", resolution_failure_allowed=True) is not None:
                args.cmake_generator = "Unix Makefiles"
            else:
                raise ValueError(
                    "Unable to find appropriate CMake generator for cross-compiling for Android. "
                    "Valid generators are 'Ninja' or 'Unix Makefiles'."
                )

        if args.cmake_generator != original_cmake_generator:
            log.info(f"Setting CMake generator to '{args.cmake_generator}' for cross-compiling for Android.")

        # no C# on Android so automatically skip
        args.build_csharp = False


def _validate_ios_args(args: argparse.Namespace):
    if args.ios:
        if not util.is_mac():
            raise ValueError("A Mac host is required to build for iOS")

        needed_args = [
            args.apple_sysroot,
            args.osx_arch,
            args.apple_deploy_target,
        ]
        arg_names = [
            "--apple_sysroot          <the location or name of the macOS platform SDK>",
            "--osx_arch              <the Target specific architectures for iOS>",
            "--apple_deploy_target   <the minimum version of the target platform>",
        ]
        have_required_args = all(_ is not None for _ in needed_args)
        if not have_required_args:
            raise ValueError(
                "iOS build on MacOS canceled due to missing arguments: "
                + ", ".join(
                val for val, cond in zip(arg_names, needed_args) if not cond
                )
            )


def _validate_cmake_args(args: argparse.Namespace):
    args.cmake_extra_defines = [i for j in args.cmake_extra_defines for i in j] if args.cmake_extra_defines else []
    args.cmake_extra_defines = [f"-D{define}" for define in args.cmake_extra_defines]


def _validate_args(args: argparse.Namespace):
    # default to all 3 stages
    if not args.update and not args.build and not args.test:
        args.update = True
        args.build = True
        args.test = True

    # validate args. this updates values in args where applicable (e.g. fully resolve paths).
    args.cmake_path = _resolve_executable_path(args.cmake_path)
    args.ctest_path = _resolve_executable_path(args.ctest_path)

    _validate_build_dir(args)
    _validate_cuda_args(args)
    _validate_android_args(args)
    _validate_ios_args(args)
    _validate_cmake_args(args)

    if args.ort_home:
        if not args.ort_home.exists() or not args.ort_home.is_dir():
            raise ValueError(f"{args.ort_home} does not exist or is not a directory.")

        args.ort_home = args.ort_home.resolve(strict=True)


def _create_env(args: argparse.Namespace):
    env = os.environ.copy()

    if args.use_cuda:
        env["CUDA_HOME"] = str(args.cuda_home)
        env["PATH"] = str(args.cuda_home / "bin") + os.pathsep + os.environ["PATH"]

    if args.android:
        env["ANDROID_HOME"] = str(args.android_home)
        env["ANDROID_NDK_HOME"] = str(args.android_ndk_path)

    return env


def _get_csharp_properties(args: argparse.Namespace, ort_lib_dir: Path):
    # Tests folder does not have a sln file. We use the csproj file to build and test.
    # The csproj file requires the platform to be AnyCPU (not "Any CPU")
    configuration = f"/p:Configuration={args.config}"
    platform = "/p:Platform=Any CPU"
    # need an extra config on windows as the actual build output is in the original build dir / config / config
    native_lib_path = f"/p:NativeBuildOutputDir={str(args.build_dir / args.config) if util.is_windows() else str(args.build_dir)}"
    ort_lib_path = f"/p:OrtLibDir={ort_lib_dir}"

    props = [configuration, platform, native_lib_path, ort_lib_path]

    return props


def _run_android_tests(args: argparse.Namespace):
    # only run the tests on the emulator for x86_64 currently.
    # TODO: may also be possible to run on a Mac with an arm64 chip
    if args.android_abi != "x86_64":
        log.info("Skipping Android tests as they are only supported on x86_64 currently.")
        return

    if not args.build_java:
        # currently we only have an Android test app that we run on the emulator to test the Java bindings.
        log.warning("Android testing requires --build_java to be set.")
        return

    sdk_tool_paths = util.android.get_sdk_tool_paths(args.android_home)
    adb = sdk_tool_paths.adb
    with contextlib.ExitStack() as context_stack:
        # use API 27 or higher so the emulator is Android 8.1 (2017) or later
        android_api = max(args.android_api, 27)

        if args.android_run_emulator:
            avd_name = "ort_genai_android"
            system_image = f"system-images;android-{android_api};default;{args.android_abi}"

            util.android.create_virtual_device(sdk_tool_paths, system_image, avd_name)
            emulator_proc = context_stack.enter_context(
                util.android.start_emulator(
                    sdk_tool_paths=sdk_tool_paths,
                    avd_name=avd_name,
                    extra_args=["-partition-size", "2047", "-wipe-data"],
                )
            )
            context_stack.callback(util.android.stop_emulator, emulator_proc)

        # use the gradle wrapper under <repo root>/java to run the test app on the emulator.
        # the test app loads and runs a test model using the GenAI Java bindings
        gradle_executable = str(REPO_ROOT / "src" / "java" / ("gradlew.bat" if util.is_windows() else "gradlew"))
        android_test_path = args.build_dir / "src" / "java" / "androidtest"
        import subprocess
        exception = None
        try:
            util.run([gradle_executable, "--no-daemon",
                      f"-DminSdkVer={android_api}",
                      "clean",
                      "connectedDebugAndroidTest"],
                     cwd=android_test_path,
                     capture_stdout=True,
                     capture_stderr=True,)
        except subprocess.CalledProcessError as e:
            exception = e
            print(e)
            print(f"Output:\n{e.output.decode('utf-8')}")
            print(f"stderr:\n{e.stderr.decode('utf-8')}")

        # Print test log output so we can easily check that the test ran as expected
        util.run([adb, "logcat", "-s", "-d", "GenAI:V ORTGenAIAndroidTest:V TestRunner:V"])

        if exception:
            # uncomment if you need more logcat output in a CI
            # util.run([adb, "logcat", "-d", "*:E"])
            raise exception


def update(args: argparse.Namespace, env: dict[str, str]):
    """
    Update the cmake build files.
    """

    # build the cmake command to create/update the build files
    command = [str(args.cmake_path)]

    command += ["-G", args.cmake_generator]

    if util.is_windows():
        if args.cmake_generator == "Ninja":
            if args.use_cuda:
                command += ["-DCUDA_TOOLKIT_ROOT_DIR=" + str(args.cuda_home)]

        elif args.cmake_generator.startswith("Visual Studio"):
            toolset_options = []

            is_x64_host = platform.machine() == "AMD64"
            if is_x64_host:
                pass

            if args.use_cuda:
                toolset_options += ["cuda=" + str(args.cuda_home)]

            if toolset_options:
                command += ["-T", ",".join(toolset_options)]

    command += [f"-DCMAKE_BUILD_TYPE={args.config}"]

    build_wheel = "OFF" if args.skip_wheel else "ON"

    command += [
        "-S",
        str(REPO_ROOT),
        "-B",
        str(args.build_dir),
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DUSE_CUDA={'ON' if args.use_cuda else 'OFF'}",
        f"-DUSE_ROCM={'ON' if args.use_rocm else 'OFF'}",
        f"-DUSE_WEBGPU={'ON' if args.use_webgpu else 'OFF'}",
        f"-DUSE_DML={'ON' if args.use_dml else 'OFF'}",
        f"-DENABLE_JAVA={'ON' if args.build_java else 'OFF'}",
        f"-DBUILD_WHEEL={build_wheel}",
        f"-DUSE_GUIDANCE={'ON' if args.use_guidance else 'OFF'}",
    ]

    if args.ort_home:
        command += [f"-DORT_HOME={args.ort_home}"]

    if args.use_cuda:
        cuda_compiler = str(args.cuda_home / "bin" / "nvcc")
        command += [f"-DCMAKE_CUDA_COMPILER={cuda_compiler}"]

    if args.android:
        command += [
            "-DCMAKE_TOOLCHAIN_FILE="
            + str((args.android_ndk_path / "build" / "cmake" / "android.toolchain.cmake").resolve(strict=True)),
            f"-DANDROID_PLATFORM=android-{args.android_api}",
            f"-DANDROID_ABI={args.android_abi}",
            f"-DANDROID_MIN_SDK={args.android_api}",
            "-DENABLE_PYTHON=OFF",
            "-DENABLE_TESTS=OFF",
        ]

    if args.ios or args.macos:
        platform_name = "macabi" if args.macos == "Catalyst" else args.apple_sysroot
        command += [
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + args.apple_deploy_target,
            f"-DBUILD_APPLE_FRAMEWORK={'ON' if args.build_apple_framework else 'OFF'}",
            "-DPLATFORM_NAME=" + platform_name,
        ]

    if args.macos:
        command += [
            f"-DCMAKE_OSX_SYSROOT={args.apple_sysroot}",
            f"-DCMAKE_OSX_ARCHITECTURES={args.osx_arch}",
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={args.apple_deploy_target}",
        ]

    if args.ios:
        def _get_opencv_toolchain_file():
            if args.apple_sysroot == "iphoneos":
                return (
                    REPO_ROOT / "cmake" / "external" / "opencv" / "platforms" / "iOS" / "cmake" /
                        "Toolchains" / "Toolchain-iPhoneOS_Xcode.cmake"
                )
            else:
                return (
                    REPO_ROOT / "cmake" / "external" / "opencv" / "platforms" / "iOS" / "cmake" /
                        "Toolchains" / "Toolchain-iPhoneSimulator_Xcode.cmake"
                )

        command += [
            "-DCMAKE_SYSTEM_NAME=iOS",
            f"-DIOS_ARCH={args.osx_arch}",
            f"-DIPHONEOS_DEPLOYMENT_TARGET={args.apple_deploy_target}",
            # The following arguments are specific to the OpenCV toolchain file
            f"-DCMAKE_TOOLCHAIN_FILE={_get_opencv_toolchain_file()}",
            "-DENABLE_PYTHON=OFF",
            "-DENABLE_TESTS=OFF",
            "-DENABLE_MODEL_BENCHMARK=OFF",
        ]
        if args.use_guidance:
            command += ["-DRust_CARGO_TARGET=aarch64-apple-ios-sim"]

    if args.macos == "Catalyst":
        if args.cmake_generator == "Xcode":
            raise Exception("Xcode CMake generator ('--cmake_generator Xcode') doesn't support Mac Catalyst build.")

        macabi_target = f"{args.osx_arch}-apple-ios{args.apple_deploy_target}-macabi"
        command += [
            "-DCMAKE_CXX_COMPILER_TARGET=" + macabi_target,
            "-DCMAKE_C_COMPILER_TARGET=" + macabi_target,
            "-DCMAKE_CC_COMPILER_TARGET=" + macabi_target,
            f"-DCMAKE_CXX_FLAGS=--target={macabi_target}",
            f"-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG --target={macabi_target}",
            f"-DCMAKE_C_FLAGS=--target={macabi_target}",
            f"-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG --target={macabi_target}",
            f"-DCMAKE_CC_FLAGS=--target={macabi_target}",
            f"-DCMAKE_CC_FLAGS_RELEASE=-O3 -DNDEBUG --target={macabi_target}",
            "-DMAC_CATALYST=1",
            "-DENABLE_PYTHON=OFF",
            "-DENABLE_TESTS=OFF",
            "-DENABLE_MODEL_BENCHMARK=OFF",
        ]

    if args.arm64:
        command += ["-A", "ARM64"]
    elif args.arm64ec:
        command += ["-A", "ARM64EC"]

    if args.arm64 or args.arm64ec:
        if args.test:
            log.warning(
                "Cannot test on host build machine for cross-compiled "
                "ARM64 builds. Will skip test running after build."
            )
            args.test = False

    if args.cmake_extra_defines != []:
        command += args.cmake_extra_defines

    util.run(command, env=env)


def build(args: argparse.Namespace, env: dict[str, str]):
    """
    Build the targets.
    """

    make_command = [str(args.cmake_path), "--build", str(args.build_dir), "--config", args.config]

    if args.parallel:
        make_command.append("--parallel")

    util.run(make_command, env=env)

    lib_dir = args.build_dir
    if not args.ort_home:
        _ = util.download_dependencies(args.use_cuda, args.use_rocm, args.use_dml, lib_dir)
    else:
        lib_dir = args.ort_home / "lib"

    if args.build_csharp:
        dotnet = str(_resolve_executable_path("dotnet"))

        # Build the library
        csharp_build_command = [dotnet, "build", ".",]
        csharp_build_command += _get_csharp_properties(args, ort_lib_dir=lib_dir)
        util.run(csharp_build_command, cwd=REPO_ROOT / "src" / "csharp")
        util.run(csharp_build_command, cwd=REPO_ROOT / "test" / "csharp")


def test(args: argparse.Namespace, env: dict[str, str]):
    """
    Run the tests.
    """
    lib_dir = args.build_dir / "test"
    if util.is_windows():
        # On Windows, the unit test executable is found inside a directory named after the configuration
        # (e.g. Debug, Release, etc.) within the test directory.
        # Whereas on as on platforms, the executable is directly under the test directory.
        lib_dir = lib_dir / args.config
    if not args.ort_home:
        _ = util.download_dependencies(args.use_cuda, args.use_rocm, args.use_dml, lib_dir)
    else:
        lib_dir = args.ort_home / "lib"

    ctest_cmd = [str(args.ctest_path), "--build-config", args.config, "--verbose", "--timeout", "10800"]
    util.run(ctest_cmd, cwd=str(args.build_dir / "test"))

    if args.build_csharp:
        dotnet = str(_resolve_executable_path("dotnet"))
        csharp_test_command = [dotnet, "test"]
        csharp_test_command += _get_csharp_properties(args, ort_lib_dir=lib_dir)
        util.run(csharp_test_command, env=env, cwd=str(REPO_ROOT / "test" / "csharp"))

    if args.build_java:
        ctest_cmd = [str(args.ctest_path), "--build-config", args.config, "--verbose", "--timeout", "10800"]
        util.run(ctest_cmd, cwd=str(args.build_dir / "src" / "java"))

    if args.android:
        _run_android_tests(args)


def clean(args: argparse.Namespace, env: dict[str, str]):
    """
    Clean the build output.
    """
    log.info("Cleaning targets")
    cmd_args = [str(args.cmake), "--build", str(args.build_dir), "--config", args.config, "--target", "clean"]
    util.run(cmd_args, env=env)


if __name__ == "__main__":
    if not (util.is_windows() or util.is_linux() or util.is_mac()):
        raise OSError(f"Unsupported platform {sys.platform}.")

    arguments = _parse_args()

    _validate_args(arguments)
    environment = _create_env(arguments)

    if arguments.update:
        update(arguments, environment)

    if arguments.build:
        build(arguments, environment)

    if arguments.test and not arguments.skip_tests:
        test(arguments, environment)

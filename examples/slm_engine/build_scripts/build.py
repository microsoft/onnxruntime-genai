#!/usr/bin/env python3
import os
import argparse
import platform
import subprocess
import pathlib
from build_deps import get_machine_type

BLUE = "\033[34m"
RED = "\033[31m"
CLEAR = "\033[0m"


def cmake_options_android(ndk_dir):
    if not os.path.exists(ndk_dir):
        raise Exception(f"{RED}NDK Directory doesn't exist: {ndk_dir}{CLEAR}")
    else:
        cmake_option = [
            f"-DCMAKE_TOOLCHAIN_FILE={ndk_dir}/build/cmake/android.toolchain.cmake",
            "-DANDROID_PLATFORM=android-33",
            "-DANDROID_ABI=arm64-v8a",
        ]
        return cmake_option


def main():
    parser = argparse.ArgumentParser(description="Build script for this repo")

    # Adding arguments
    parser.add_argument("--android", action="store_true", help="Build for Android")
    parser.add_argument("--android_ndk_path", type=str, help="Path to ANDROID NDK")
    parser.add_argument(
        "--build_type",
        type=str,
        default="Release",
        help="{Release|RelWithDebInfo|Debug}",
    )

    # Parsing arguments
    args = parser.parse_args()

    # Determine the toplevel directory
    path = pathlib.Path(__file__).parent.resolve()
    TOPLEVEL_DIR = path.parent.absolute()

    # We need to get the name of the toplevel/src directory
    TOPLEVEL_DIR = f"{TOPLEVEL_DIR}/src"

    artifacts_dir = os.path.abspath(f"deps/artifacts/")
    cmake_options = [
        "cmake",
        "-G",
        "Ninja",
        TOPLEVEL_DIR,
        f"-DARTIFACTS_DIR={artifacts_dir}",
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
    ]

    # We keep the build directory prefix as same as that's returned by the
    # platform.system() call which maps 1:1 with the Linux uname -s command.
    # When cross-compiling for Android, we use Android as the prefix.

    # Use list comprehension to get the platform specific build directory prefix
    dir_prefix = platform.system() if not args.android else "Android"
    build_dir = f"builds/{dir_prefix}-{get_machine_type(args)}"
    if args.android:
        # Check to make sure that the other two options are also defined
        if args.android_ndk_path is None:
            print(f"Need to define android_ndk_path for Android builds")
            return
        cmake_options.extend(cmake_options_android(args.android_ndk_path))

    # Launch build
    print(f"BUILD Dir:", build_dir)
    os.makedirs(build_dir, exist_ok=True)

    print(f"{BLUE}CMAKE Options: {cmake_options}{CLEAR}")

    print(f"Building ...")
    os.chdir(build_dir)
    result = subprocess.call(cmake_options)
    if result != 0:
        raise Exception(f"{RED}CMake error!{CLEAR}")

    result = subprocess.call(
        [
            "cmake",
            "--build",
            ".",
            "--parallel",
            "--config",
            args.build_type,
        ]
    )
    if result != 0:
        raise Exception(f"{RED}Build error!{CLEAR}")

    # Now run the installation
    print(f"Installing...")
    result = subprocess.call(["cmake", "--install", "."])
    if result != 0:
        raise Exception(f"{RED}Installation error!{CLEAR}")

    print(f"{BLUE}Build complete{CLEAR}")
    print(f"{BLUE}Artifacts are available here: {artifacts_dir}{CLEAR}")


if __name__ == "__main__":
    main()

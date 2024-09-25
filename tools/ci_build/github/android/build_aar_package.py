#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import os
import subprocess
import sys

from pathlib import Path

REPO_ROOT = Path(__file__).parents[4]
BUILD_PY = REPO_ROOT / "build.py"
JAVA_ROOT = REPO_ROOT / "src" / "java"
sys.path.append(str(REPO_ROOT / "tools" / "python"))
import util  # REPO_ROOT/tools/python/util noqa: E402

# Default to only build 64-bit ABIs. TBD if we need to support any 32-bit ABIs.
# DEFAULT_BUILD_ABIS = ["armeabi-v7a", "arm64-v8a", "x86", "x86_64"]
DEFAULT_BUILD_ABIS = ["arm64-v8a", "x86_64"]

# Match onnxruntime and use NDK API 21 by default
# It is possible to build from source for Android API levels below 21, but it is not guaranteed.
# As level 21 is Android 5.0 from 2014, it's highly unlikely that we will need to support anything lower.
DEFAULT_ANDROID_MIN_SDK_VER = 21

# Android API 33+ is required for new apps and app updates from August 31, 2023
# Android API 34+ will be required from August 31, 2024
# See https://apilevels.com/
DEFAULT_ANDROID_TARGET_SDK_VER = 33


def _path_from_env_var(env_var: str):
    env_var_value = os.environ.get(env_var)
    return Path(env_var_value) if env_var_value is not None else None


def _parse_build_settings(args):
    setting_file = args.build_settings_file.resolve()

    if not setting_file.is_file():
        raise FileNotFoundError(f"Build config file {setting_file} is not a file.")

    with open(setting_file) as f:
        build_settings_data = json.load(f)

    build_settings = {}

    if "build_abis" in build_settings_data:
        build_settings["build_abis"] = build_settings_data["build_abis"]
    else:
        build_settings["build_abis"] = DEFAULT_BUILD_ABIS

    build_params = []
    if "build_params" in build_settings_data:
        build_params += build_settings_data["build_params"]
    else:
        raise ValueError("build_params is required in the build config file")

    if "android_min_sdk_version" in build_settings_data:
        build_settings["android_min_sdk_version"] = build_settings_data["android_min_sdk_version"]
    else:
        build_settings["android_min_sdk_version"] = DEFAULT_ANDROID_MIN_SDK_VER

    build_params += ["--android_api=" + str(build_settings["android_min_sdk_version"])]

    if "android_target_sdk_version" in build_settings_data:
        build_settings["android_target_sdk_version"] = build_settings_data["android_target_sdk_version"]
    else:
        build_settings["android_target_sdk_version"] = DEFAULT_ANDROID_TARGET_SDK_VER

    if build_settings["android_min_sdk_version"] > build_settings["android_target_sdk_version"]:
        raise ValueError(
            f"android_min_sdk_version {build_settings['android_min_sdk_version']} cannot be larger than "
            f"android_target_sdk_version {build_settings['android_target_sdk_version']}"
        )

    build_settings["build_params"] = build_params

    return build_settings


def _build_aar(args):
    build_settings = _parse_build_settings(args)
    build_dir = Path(args.build_dir).resolve()

    # Setup temp environment for building
    temp_env = os.environ.copy()
    temp_env["ANDROID_HOME"] = str(args.android_home.resolve(strict=True))
    temp_env["ANDROID_NDK_HOME"] = str(args.android_ndk_path.resolve(strict=True))

    # Temp dirs to hold building results
    intermediates_dir = build_dir / "intermediates"
    build_config = args.config
    aar_dir = intermediates_dir / "aar" / build_config
    jnilibs_dir = intermediates_dir / "jnilibs" / build_config
    base_build_command = [sys.executable, str(BUILD_PY), f"--config={build_config}"]
    if args.ort_home:
        base_build_command += [f"--ort_home={str(args.ort_home)}"]
    base_build_command += build_settings["build_params"]

    header_files_path = None

    # Build binary for each ABI, one by one
    for abi in build_settings["build_abis"]:
        abi_build_dir = intermediates_dir / abi
        abi_build_command = [*base_build_command, "--android_abi=" + abi, "--build_dir=" + str(abi_build_dir)]

        subprocess.run(abi_build_command, env=temp_env, shell=False, check=True, cwd=REPO_ROOT)

        # create symbolic links for libonnxruntime-genai.so and libonnxruntime-genai-jni.so
        # to jnilibs/[abi] for later compiling the aar package
        abi_jnilibs_dir = jnilibs_dir / abi
        abi_jnilibs_dir.mkdir(parents=True, exist_ok=True)
        for lib_name in ["libonnxruntime-genai.so", "libonnxruntime-genai-jni.so"]:
            src_lib_name = abi_build_dir / build_config / "src" / "java" / "android" / abi / lib_name
            target_lib_name = abi_jnilibs_dir / lib_name
            # If the symbolic already exists, delete it first
            target_lib_name.unlink(missing_ok=True)
            target_lib_name.symlink_to(src_lib_name)

        # we only need to define the header files path once
        if not header_files_path:
            header_files_path = abi_build_dir / build_config / "src" / "java" / "android" / "headers"

    # The directory to publish final AAR
    aar_publish_dir = os.path.join(build_dir, "aar_out", build_config)
    os.makedirs(aar_publish_dir, exist_ok=True)

    gradle_path = JAVA_ROOT / ("gradlew" if not util.is_windows() else "gradlew.bat")

    # get the common gradle command args
    gradle_command = [
        gradle_path,
        "--no-daemon",
        "-b=build-android.gradle",
        "-c=settings-android.gradle",
        f"-DjniLibsDir={jnilibs_dir}",
        f"-DbuildDir={aar_dir}",
        f"-DheadersDir={header_files_path}",
        f"-DpublishDir={aar_publish_dir}",
        f"-DminSdkVer={build_settings['android_min_sdk_version']}",
        f"-DtargetSdkVer={build_settings['android_target_sdk_version']}",
    ]

    # clean, build, and publish to a local directory
    subprocess.run([*gradle_command, "clean"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)
    subprocess.run([*gradle_command, "build"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)
    subprocess.run([*gradle_command, "publish"], env=temp_env, shell=False, check=True, cwd=JAVA_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Create Android Archive (AAR) package for one or more Android ABI(s)
        and building properties specified in the given build config file.
        See tools/ci_build/github/android/default_aar_build_settings.json for details.
        The output of the final AAR package can be found under [build_dir]/aar_out
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--android_home", type=Path, default=_path_from_env_var("ANDROID_HOME"),
        help="Path to the Android SDK."
    )

    parser.add_argument(
        "--android_ndk_path", type=Path, default=_path_from_env_var("ANDROID_NDK_HOME"),
        help="Path to the Android NDK."
    )

    parser.add_argument(
        "--build_dir", type=Path, default=(REPO_ROOT / "build" / "android_aar"),
        help="Provide the root directory for build output",
    )

    parser.add_argument(
        "--config", type=str, default="Release",
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration to build.",
    )

    parser.add_argument("--ort_home", type=Path, default=None,
                        help="Path to an unzipped onnxruntime AAR.")

    parser.add_argument(
        "build_settings_file", type=Path, help="Provide the file contains settings for building AAR"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Android SDK home and NDK path are required to come from explicit args or env vars.
    # If they're not set here, neither contained a value.
    if not args.android_home:
        raise ValueError("android_home is required")
    if not args.android_ndk_path:
        raise ValueError("android_ndk_path is required")

    _build_aar(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import glob
import os
import argparse
import platform
import shutil
import subprocess


def cmake_options_android(ndk_dir):
    if not os.path.exists(ndk_dir):
        raise Exception(f"NDK Directory doesn't exist: {ndk_dir}")
    cmake_option = [
        f"-DCMAKE_TOOLCHAIN_FILE={ndk_dir}/build/cmake/android.toolchain.cmake",
        "-DANDROID_PLATFORM=android-33",
        "-DANDROID_ABI=arm64-v8a",
    ]
    return cmake_option


def get_platform_dirname(args):
    # Get the name of the OS
    import platform

    platform_name = platform.system()
    if platform_name == "Darwin":
        platform_name = "MacOS"

    if args.android:
        platform_name = "Android"

    return platform_name


def get_machine_type(args):
    machine_type = platform.machine()
    if args.android:
        machine_type = "aarch64"

    return machine_type


def copy_files_without_hidden(src, dest):
    """
    Recursively copies files from the source directory to the destination directory,
    excluding hidden files and directories.

    Args:
      src: Path to the source directory.
      dest: Path to the destination directory.
    """
    try:
        os.makedirs(
            dest, exist_ok=True
        )  # Create destination directory if it doesn't exist

        for root, dirs, files in os.walk(src):
            for file in files:
                if not file.startswith("."):  # Exclude hidden files
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest, os.path.relpath(src_file, src))
                    os.makedirs(
                        os.path.dirname(dest_file), exist_ok=True
                    )  # Create necessary directories
                    shutil.copy2(src_file, dest_file)

    except OSError as e:
        print(f"Error: {e}")
        raise e


def copy_files_keeping_symlinks(src_files, dest):
    if not type(src_files) == list:
        raise Exception("src_files must be a list")

    for file in src_files:
        # print(f"\033[34;1mFile: {file}\033[0m")
        # Preserve symlinks
        if os.path.islink(file):
            # Get the name of the link without the rest of the path
            linkname = f"{dest}/{os.path.basename(file)}"
            linkto = os.path.basename(os.readlink(file))
            if not os.path.exists(linkname):
                os.symlink(linkto, linkname)
                # print(f"\033[35;1mCreating symlink: {linkname} -> {linkto}\033[0m")
                # print(f"\033[35;1mDest Dir: {dest}\033[0m")
        else:
            shutil.copy2(file, dest)


def build_ort(args):
    """
    Build the ONNX Runtime library and ORT-GenAI library
    """

    # Navigate to the deps directory
    os.chdir("deps")

    # Make the src directory if needed
    os.makedirs("src", exist_ok=True)

    if not os.path.exists("src/onnxruntime"):
        # Clone the ORT Repo
        print("Cloning ONNX Runtime")
        os.chdir("src")
        if (
            subprocess.call(
                ["git", "clone", "https://github.com/microsoft/onnxruntime.git"]
            )
            != 0
        ):
            raise Exception("Failed to clone ONNX Runtime")

        # Now get the dependencies
        os.chdir("onnxruntime")

        # Checkout the correct version
        version = "v1.20.1"
        if subprocess.call(["git", "checkout", version]) != 0:
            raise Exception("Failed to checkout ONNX Runtime version")

        if (
            subprocess.call(["git", "submodule", "update", "--init", "--recursive"])
            != 0
        ):
            raise Exception("Failed to  update ONNX Runtime submodules")

        # Return to the original directory
        os.chdir("../..")

    # Prepare the command arguments
    cmd_args = [
        "--build_shared_lib",
        "--skip_tests",
        "--parallel",
        "--config",
        args.build_type,
    ]
    if args.android:
        cmd_args.extend(
            [
                "--android",
                "--android_sdk_path",
                args.android_sdk_path,
                "--android_ndk_path",
                args.android_ndk_path,
                "--android_abi",
                "arm64-v8a",
                "--android_api",
                args.api_level,
            ]
        )
        if args.qnn_sdk_path:
            cmd_args.extend(["--use_qnn", "--qnn_home", args.qnn_sdk_path])

    # now build the ORT library
    print("Building ONNX Runtime")
    os.chdir("src/onnxruntime")

    build_script = "build.bat" if platform.system() == "Windows" else "./build.sh"
    print(f"Running {build_script} with args: {cmd_args}")
    result = subprocess.call([build_script] + cmd_args)
    if result != 0:
        raise Exception("Failed to build ONNX Runtime")

    # Now add the symbolic links
    # First save the current directory
    current_dir = os.getcwd()

    # Get the absolute path tot he build directory
    build_dir_name = f"build/{get_platform_dirname(args)}/{args.build_type}"
    build_dir_name = os.path.abspath(build_dir_name)
    ort_home = os.path.abspath(f"{build_dir_name}/install")

    os.chdir(build_dir_name)

    # Run install
    print("Running install")
    result = subprocess.call(
        [
            "cmake",
            "--install",
            ".",
            "--prefix",
            ort_home,
        ]
    )

    if result != 0:
        raise Exception("Failed to install ONNX Runtime")

    # Now create the symbolic links only if Android Build
    if args.android:
        os.chdir(ort_home)
        # Create the symbolic links only in doesn't exist
        if not os.path.exists("headers"):
            os.symlink("include/onnxruntime", "headers")

        # Make the jni directory
        os.makedirs("jni", exist_ok=True)
        os.chdir("jni")
        if not os.path.exists("arm64-v8a"):
            os.symlink("../lib", "arm64-v8a")

    # Back to the original directory
    os.chdir(current_dir)
    os.chdir("../../../")
    print(f"Current Directory: {os.getcwd()}")

    # Save the current directory
    current_dir = os.getcwd()
    print(f"Current Directory: {current_dir}")

    # Go to the toplevel directory. To determine the top level directory, we need to
    # find the directory of this python file and then go from there
    top_level_dir = f"../../../"
    os.chdir(top_level_dir)

    if subprocess.call(["git", "submodule", "update", "--init", "--recursive"]) != 0:
        raise Exception("Failed to update submodules")

    # Now build the ORT-GenAI library
    print("Building ORT-GenAI")
    # Prepare the command arguments
    cmd_args = [
        "--skip_wheel",
        "--skip_tests",
        "--parallel",
        "--config",
        args.build_type,
        "--cmake_extra_defines",
        "ENABLE_PYTHON=OFF",
    ]
    if args.android:
        cmd_args.extend(
            [
                "--android",
                "--android_home",
                args.android_sdk_path,
                "--android_ndk_path",
                args.android_ndk_path,
                "--android_abi",
                "arm64-v8a",
                "--android_api",
                args.api_level,
                "--ort_home",
                ort_home,
            ]
        )

    print(f"Running build.py with args: {cmd_args}")
    result = subprocess.call(["python", "build.py"] + cmd_args)
    if result != 0:
        raise Exception("Failed to build ORT-GenAI")

    # Now install the ORT-GenAI library
    build_dir_name = f"build/{get_platform_dirname(args)}/{args.build_type}"
    build_dir_name = os.path.abspath(build_dir_name)

    os.chdir(build_dir_name)

    # Run install
    print("Running install")
    result = subprocess.call(
        [
            "cmake",
            "--install",
            ".",
            "--prefix",
            f"{build_dir_name}/install",
        ]
    )

    if result != 0:
        raise Exception("Failed to install ONNX Runtime")

    # Now copy the ORT Libs to the ORT-GenAI directory installation location
    dest_dir = f"{build_dir_name}/install/lib"
    copy_files_keeping_symlinks(glob.glob(f"{ort_home}/lib/*onnxruntime*"), dest_dir)

    # For Windows build, the .dll files are stored in the bin directory.
    # For Linux/Mac this is a no-op
    copy_files_keeping_symlinks(glob.glob(f"{ort_home}/bin/*onnxruntime*"), dest_dir)

    # The "current_dir" is the "build_scripts" directory. Need to
    os.chdir(current_dir)
    print(f"Current Directory: {os.getcwd()}")

    # Now copy the artifacts to the artifacts directory
    artifacts_dir = os.path.abspath(
        f"deps/artifacts/{get_platform_dirname(args)}-{get_machine_type(args)}"
    )
    print(f"\033[35;1mCopying artifacts to {artifacts_dir}\033[0m")

    os.makedirs(f"{artifacts_dir}/include", exist_ok=True)
    os.makedirs(f"{artifacts_dir}/lib", exist_ok=True)

    copy_files_keeping_symlinks(
        glob.glob(f"{build_dir_name}/install/lib/*"), f"{artifacts_dir}/lib"
    )
    copy_files_keeping_symlinks(
        glob.glob(f"{build_dir_name}/install/bin/*"), f"{artifacts_dir}/lib"
    )

    copy_files_keeping_symlinks(
        glob.glob(f"{build_dir_name}/install/include/*"),
        f"{artifacts_dir}/include",
    )


def build_header_only(args):
    """
    Build the header-only libraries
    """
    # List of header-only libraries
    header_only_libs = [
        {
            "name": "json",
            "url": "https://github.com/nlohmann/json.git",
            "version": "v3.11.3",
            "common_dest": False,
            "directory": "include",
        },
        {
            "name": "argparse",
            "url": "https://github.com/p-ranav/argparse.git",
            "version": "v3.2",
            "common_dest": False,
            "directory": "include",
        },
        {
            "name": "cpp-httplib",
            "url": "https://github.com/yhirose/cpp-httplib.git",
            "version": "v0.18.5",
            "common_dest": True,
            "files": ["httplib.h"],
        },
    ]

    # Copy the headers to the artifacts directory
    dest_root_dir = os.path.abspath(f"deps/artifacts/common/include")
    print(f"\033[35;1mCopying headers to {dest_root_dir}\033[0m")

    os.chdir("deps")
    print(f"Current Directory: {os.getcwd()}")

    for lib in header_only_libs:
        print(f"Building {lib['name']}")
        # Clone the repo
        if not os.path.exists(f"src/{lib['name']}"):
            # Clone the ORT Repo
            print(f"Cloning {lib['name']}")
            os.chdir("src")
            result = subprocess.call(["git", "clone", lib["url"]])
            if result != 0:
                print(f"Failed to clone {lib['name']}")
                return
            os.chdir("..")

        # Go to src
        os.chdir("src")

        # Checkout the specific version
        os.chdir(lib["name"])
        result = subprocess.call(["git", "fetch", "--tags", "origin"])
        if result != 0:
            print(f"Failed to get tags for {lib['name']}")
            return

        result = subprocess.call(["git", "checkout", lib["version"]])
        if result != 0:
            print(f"Failed to checkout version: {lib['version']} {lib['name']}")
            return

        if not os.path.exists(dest_root_dir):
            os.makedirs(dest_root_dir, exist_ok=True)

        # If the files key is defined, then copy the files
        if "files" in lib:
            print(f"Current Directory: {os.getcwd()}")
            print(f"Copying files: {lib['files']}")
            print(f"Destination Directory: {dest_root_dir}")
            for file in lib["files"]:
                shutil.copy2(file, dest_root_dir)
        elif "directory" in lib:
            os.chdir("..")
            print(f"Current Directory: {os.getcwd()}")
            print(f"Copying files: {lib['name']}/{lib['directory']}")
            print(f"Destination Directory: {dest_root_dir}")
            copy_files_without_hidden(
                f"{lib['name']}/{lib['directory']}", dest_root_dir
            )
        else:
            # Copy the entire directory
            os.chdir("..")
            print(f"Current Directory: {os.getcwd()}")
            print(f"Destination Directory: {dest_root_dir}")
            copy_files_without_hidden(lib["name"], dest_root_dir)

        # Return to the original directory
        os.chdir("..")
        print(f"Current Directory: {os.getcwd()}")


def main():
    parser = argparse.ArgumentParser(
        description="Build script for dependency libraries"
    )

    # Adding arguments
    parser.add_argument("--android", action="store_true", help="Build for Android")
    parser.add_argument("--android_sdk_path", type=str, help="Path to ANDROID SDK")
    parser.add_argument("--android_ndk_path", type=str, help="Path to ANDROID NDK")
    parser.add_argument(
        "--api_level", type=str, help="Android API Level", default="27"
    )  # e.g., 29
    parser.add_argument(
        "--qnn_sdk_path",
        type=str,
        help="Path to Qualcomm QNN SDK (AI Engine Direct)",
    )
    parser.add_argument(
        "--build_type",
        type=str,
        default="Release",
        help="{Release|RelWithDebInfo|Debug}",
    )

    parser.add_argument(
        "--skip_ort_build",
        action="store_true",
        help="If set, skip building ONNX Runtime",
    )

    # Parsing arguments
    args = parser.parse_args()

    # Change directory to where this Python file is located to avoid any issues
    # related to running this script from another directory
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Create the deps directory if it doesn't exist
    os.makedirs("deps", exist_ok=True)

    if not args.skip_ort_build:
        build_ort(args)

    build_header_only(args)

    # Return to the original directory
    os.chdir("..")
    print(f"Current Directory: {os.getcwd()}")


if __name__ == "__main__":
    main()

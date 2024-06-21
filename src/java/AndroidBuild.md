# Android Build Setup

## Install Android SDK

Install the Android SDK and NDK. The ONNX Runtime instructions can be followed. 

https://onnxruntime.ai/docs/build/android.html#prerequisites

Use the latest release Android NDK available.


## Get the ONNX Runtime Android package

Download the ONNX Runtime Android package from Maven.

https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android

Unzip the AAR file to a directory. This directory will be specified as the `--ort_home` parameter.

## Build GenAI for Android with Java bindings.

Build from the root directory of the repository using build.bat or build.sh (which call build.py).

- Specify the Android SDK directory in the ANDROID_HOME environment variable, or `--android_home` parameter.
- Specify the Android NDK directory in the ANDROID_NDK_HOME environment variable, or `--android_ndk_path` parameter. 
  - This will be a subdirectory of the Android SDK's `ndk` directory.
- Specify the minimum Android API to build for in `--android_api`.
- Specify the ABI to build for in `--android_abi`. 
  - If testing using the emulator this should match the host machine architecture. 
  - If testing on an Android device this should match the device architecture, most likely `arm64-v8a`
- Specify `--ort_home` to be the path to the unzipped ONNX Runtime AAR. This directory should contain folders called `jni` and `headers`.
- On Windows, the `--cmake_generator` must be `Ninja`. 
  - This can be installed from [here](https://github.com/ninja-build/ninja/releases) or with  `pip install ninja`.

If `--android_run_emulator` is specified the build script will create and run the emulator. A unit test app will be deployed to the emulator and run.

Run the build script with `--help` for further details on all these options.

### Example build commands

#### Windows

.\build.bat --parallel  --config=Release --cmake_generator=Ninja --build_java --android --android_home=D:\Android --android_ndk_path=D:\Android\ndk\26.2.11394342 --android_api=27 --android_abi=x86_64 --ort_home=<path to unzipped onnxruntime-android.aar> --android_run_emulator

#### Linux

./build.sh --parallel  --config=Release --build_java --android --android_home=/home/me/Android --android_ndk_path=/home/me/Android/ndk/26.2.11394342 --android_api=27 --android_abi=x86_64 --ort_home=<path to unzipped onnxruntime-android.aar> --android_run_emulator

## Build an AAR with the GenAI Java bindings

There is also a script to build an AAR with x86_64 and arm64-v8a libraries. See /tools/ci_build/github/android/build_aar_package.py.

This AAR can be used in a test Android app. 
See src\java\src\test\android\app\build.gradle for example of how to manually specify onnxruntime-android and a custom built onnxruntime-genai AAR as dependencies in build.gradle. 
The dependencies can also be added to an app using Android Studio.


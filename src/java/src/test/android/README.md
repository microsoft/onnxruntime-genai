# Android Test Application for ONNX Runtime GenAI

This directory contains a simple android application for testing the ONNX Runtime GenaI AAR package.

### Test Android Application Overview

This android application is mainly aimed for testing:

- Model used: test/test_models/hf-internal-testing/tiny-random-gpt2-fp32
- Main test file: An android instrumentation test under `app\src\androidtest\java\ai.onnxruntime.genai.example.javavalidator\SimpleTest.kt`
- The main dependency of this application is `onnxruntime-genai` aar package under `app\libs`.
- The onnxruntime dependency is provided by the latest released onnxruntime-android package.
- The MainActivity of this application is set to be empty.

### Requirements

- JDK version 11 or later is required.
- The [Gradle](https://gradle.org/) build system is required for building the APKs used to run [android instrumentation tests](https://source.android.com/compatibility/tests/development/instrumentation). Version 7.5 or newer is required.
  The Gradle wrapper at `java/gradlew[.bat]` may be used.

### Building

Build for Android with the additional  `--build_java` and `--android_run_emulator` options.

e.g.
`./build --android --android_home D:\Android --android_ndk_path D:\Android\ndk\26.3.11579264\ --android_abi x86_64 --ort_home 'path to unzipped onnxruntime-android.aar from https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android/<version>' --build_java --android_run_emulator`

Please note that you must set the `--android_abi` value to match the local system architecture, as the Android instrumentation test is run on an Android emulator on the local system.

See ../../AndroidBuild.md for more information on building for Android.

#### Build Output

The build will generate two apks which is required to run the test application in `$YOUR_BUILD_DIR/src/java/androidtest/app/build/outputs/apk`:

* `androidTest/debug/app-debug-androidTest.apk`
* `debug/app-debug.apk`

After running the build script, the two apks will be installed on `ort_genai_android` emulator and it will automatically run the test application in an adb shell.

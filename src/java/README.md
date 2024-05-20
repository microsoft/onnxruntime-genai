# ONNX Runtime GenAI Java API

This directory contains the Java language binding for the ONNX Runtime GenAI.
Java Native Interface (JNI) is used to allow for seamless calls to ONNX Runtime GenAI from Java.

## Usage

This document pertains to developing, building, running, and testing the API itself in your local environment.
For general purpose usage of the publicly distributed API, please see the [general Java API documentation](https://www.onnxruntime.ai/docs/reference/api/java-api.html).

### Building

Build with the `--build_java` option.

Windows: `REPO_ROOT/build --build_java`
*nix: `REPO_ROOT/build.sh --build_java`

#### Requirements

Java 11 or later is required to build the library. The compiled jar file will run on Java 8 or later.

The [Gradle](https://gradle.org/) build system is used here to manage the Java project's dependency management, compilation, testing, and assembly.
In particular, the Gradle [wrapper](https://docs.gradle.org/current/userguide/gradle_wrapper.html) at `java/gradlew[.bat]` is used, locking the Gradle version to the one specified in the `java/gradle/wrapper/gradle-wrapper.properties` configuration.
Using the Gradle wrapper removes the need to have the right version of Gradle installed on the system.

#### Build Output

The build will generate output in `$REPO_ROOT/build/$OS/$CONFIGURATION/src/java`:

* `build/docs/javadoc/` - HTML javadoc
* `build/reports/` - detailed test results and other reports
* `build/libs/onnxruntime-genai-VERSION.jar` - JAR with compiled classes
* `native-jni` - platform-specific JNI shared library
* `native-lib` - platform-specific onnxruntime-genai and onnxruntime shared libraries.

#### Build System Overview

The main CMake build system delegates building and testing to Gradle.
This allows the CMake system to ensure all of the C/C++ compilation is achieved prior to the Java build.
The Java build depends on C/C++ onnxruntime-genai shared library and a C JNI shared library (source located in the `src/main/native` directory).
The JNI shared library is the glue that allows for Java to call functions in onnxruntime-genai shared library.
Given the fact that CMake injects native dependencies during CMake builds, some gradle tasks (primarily, `build`, `test`, and `check`) may fail.

When running the build script, CMake will compile the `onnxruntime-genai` target and the JNI glue `onnxruntime-genai-jni` target and expose the resulting libraries in a place where Gradle can ingest them.
Upon successful compilation of those targets, a special Gradle task to build will be executed. The results will be placed in the output directory stated above.

### Advanced Loading

The default behavior is to load the shared libraries using classpath resources.
If your use case requires custom loading of the shared libraries, please consult the javadoc in the [package-info.java](src/main/java/ai/onnxruntime-genai/package-info.java) or [OnnxRuntimeGenAI.java](src/main/java/ai/onnxruntime-genai/GenAI.java) files.

## Development

### Code Formatting

[Spotless](https://github.com/diffplug/spotless/tree/master/plugin-gradle) is used to keep the code properly formatted.
Gradle's `spotlessCheck` task will show any misformatted code.
Gradle's `spotlessApply` task will try to fix the formatting.
Misformatted code will raise failures when checks are ran during test run.

###  JNI Headers

When adding or updating native methods in the Java files, the auto-generated JNI headers in `build/headers/ai_onnxruntime-genai*.h` can be used to determine the JNI function signature.

These header files can be manually generated using Gradle's `compileJava` task which will compile the Java and update the header files accordingly.

Cut-and-paste the function declaration from the auto-generated .h file to add the implementation in the `./src/main/native/ai_onnxruntime-genai_*.cpp` file.

### Dependencies

The Java API does not have any runtime or compile dependencies.

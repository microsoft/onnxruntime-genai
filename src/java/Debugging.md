# Debugging Notes for Windows

## Extremely hacky steps to debug using VS Code.

There must be a better way, but I'm not a java expert and don't know how to wire up the CMake + Gradle build with 
VS Code or something like IntelliJ.

For VS Code the following worked well enough to debug the java tests in a very manual way.

Create a config for java tests in settings.json. Adjust the paths based on your setup. 
My repo root is D:\src\github\ort.genai, and I was testing a Debug build on Windows. 

```yaml
    "java.test.config": {
        "testKind": "junit",
        "workingDirectory": "D:\\src\\github\\ort.genai\\build\\Windows\\Debug\\src\\java",
        "classPaths": [ 
            "D:\\src\\github\\ort.genai\\src\\java\\build\\classes\\java\\main",
            "D:\\src\\github\\ort.genai\\src\\java\\build\\classes\\java\\test",
            "D:\\src\\github\\ort.genai\\src\\java\\build\\resources\\test"
        ],
        "sourcePaths": [
            "D:\\src\\github\\ort.genai\\src\\java\\src\\main\\java",
            "D:\\src\\github\\ort.genai\\src\\java\\src\\test\\java"
        ],
        "vmArgs": [ "-Djava.library.path=D:\\src\\github\\ort.genai\\build\\Windows\\Debug\\src\\java\\native-lib\\ai\\onnxruntime_genai\\native\\win-x64" ],
    },
```

In theory vmArgs would allow updating the path so the native libraries can be found, but it doesn't seem to work.

The best way around this I found was to update the src/java/CMakeLists.txt to copy the native libs to native-jni
instead of native-lib. Easiest way to do that is to set JAVA_PACKAGE_LIB_DIR to be the same as JAVA_PACKAGE_JNI_DIR.

CMakeLists.txt has this: 
```cmake
if (WIN32)
  # Uncomment the below line if you need to debug unit tests on Windows so that all the native dlls end up in the
  # one directory. See Debugging.md and test/java/ai/onnxruntime_genai/TestUtils.java.
  # set(JAVA_PACKAGE_LIB_DIR ${JAVA_PACKAGE_JNI_DIR})
```
That gets all the dlls in the one directory, and you can add this to a static init in the test code for GenAI Java
bindings to look there.

See setLocalNativeLibraryPath in src/java/test/java/ai/onnxruntime_genai/TestUtils.java to adjust the build output path.
Add a static member to the test class you want to debug that calls this to ensure the path is set before the GenAI
bindings are loaded
```java
  private static final boolean customPathRegistered = TestUtils.setLocalNativeLibraryPath();
```

Then you can run the tests from VS Code.

You may also want to set this in the VS Code settings: 
```
    "java.debug.settings.onBuildFailureProceed": true,
```

I didn't try to setup VS Code to be able to build the tests using cmake or gradle, so the build VS Code attempts 
before run/debug of a test always fails. Instead I built the binding/test code from the command line, and then debugged
the tests from VS Code.

You can do a top level build (`./build --build_java --config Debug --build --test ...` from the repo root), 
or manually run gradlew from the src/java directory.

e.g. the command line looks something like this. Adjust build output path as needed.
```
> D:\src\github\ort.genai\src\java>D:/src/github/ort.genai/src/java/gradlew --info test -DcmakeBuildDir="D:/src/github/ort.genai/build/Windows/Debug/src/java" -Dorg.gradle.daemon=false
```

NOTE: If using the top-level build, the unit test code gets built in the 'test' phase - that's just how the gradle build is setup.

-----

# Debugging native code

It's challenging to debug the native code that is called from the Java as they're run via gradle but executed in java.exe processes.
I don't know how to create the necessary environment to run java.exe directly with a test. 

One approach that works to be able to at least see the native code stack and locals is to force a heap violation. You can't continue, but you can at least attach to the process to see what's going on.

e.g. Add something this to the ORT or GenAI code in question to force a heap violation.
```c++
  std::string str = "break";
  delete str.c_str();
```

Build the native library and ensure it's in the native-jni path mentioned above when you run the tests.
That should result in an error window with Debug/Retry/Ignore buttons.
When that error window appears, you first need to attach to the Java process that is running the tests. 
Process Explorer (from sysinternals) is a good way to find the process as you can use it's 'target' feature to select the error window.
Attach to that process in Visual Studio, ideally using the solution for the native library you built with the heap violation.

Once you have attached, hit the 'Retry' button in the error window. That _should_ drop you into VS with full debug info and call stack in the genai code. Note that you may need to change to the current thread to see the call stack as it may be in the Java thread that is displaying the dialog box.

If someone knows the magic incantation to run java.exe directly with a unit test (with class paths etc. set up correctly) that would be a much better way to debug the native code.

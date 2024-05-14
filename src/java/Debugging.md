Extremely hacky steps to debug using VS Code.

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
            "D:\\src\\github\\ort.genai\\src\\java\build\resources\test"
        ],
        "sourcePaths": [
            "D:\\src\\github\\ort.genai\\src\\java\\src\\main\\java",
            "D:\\src\\github\\ort.genai\\src\\java\\src\\test\\java"
        ],
        "vmArgs": [ "-Djava.library.path=D:\\src\\github\\ort.genai\\build\\Windows\\Debug\\src\\java\\native-lib\\ai\\onnxruntime_genai\\native\\win-x64" ],
    },
```

In theory vmArgs would allow updating the path so the native libraries can be found, but it doesn't seem to work.

I ended up copying the native libs into teh src/java/native-jni/ai/onnxruntime_genai/native/win-x64 folder and 
manually added code to the unit test file to set the system property name that the GenAI.java code can use as a 
path to attempt loading from. Whilst this works, you have to delete/copy the native libs each time you build otherwise
you get an error about having duplicate versions of the native files. Maybe a better alternative is to hack the 
CMakeLists.txt to copy the native libs to native-jni instead of native-lib. I assume that directory structure is
required for packaging though, so it would be a temporary change.

```java
    System.setProperty(
        "onnxruntime_genai.native.path",
        "D:\\src\\github\\ort.genai\\build\\Windows\\Debug\\src\\java\\native-jni\\ai\\onnxruntime_genai\\native\\win-x64");
```

You may also want to set this: 
```
    "java.debug.settings.onBuildFailureProceed": true,
```

I didn't try to setup VS Code to be able to build the tests using cmake or gradle, so the build VS Code attempts 
before run/debug of a test always fails. 
I built the binding/test code from the command line, and then debugged the tests from VS Code.

You can do a top level build (`./build --build_java --config Debug --build --test ...` from the repo root), 
or manually run gradlew from the src/java directory.

e.g. 
> D:\src\github\ort.genai\src\java>D:/src/github/ort.genai/src/java/gradlew --info test -DcmakeBuildDir="D:/src/github/ort.genai/build/Windows/Debug/src/java" -Dorg.gradle.daemon=false

NOTE: If using the top-level build, the unit test code gets built in the 'test' phase - that's just how the gradle build is setup.


---
If the JNI code causes an unhandled exception it is possible to do something meaningful. 

First you need to attach to the Java process that is running the tests. Process Explorer (from sysinternals) is a good way to find the process as you can use it's 'target' feature to select the error message dialog box.

Once you have attached, hit the 'Retry' button. That _should_ drop you into VS with full debug info and call stack in the genai code.



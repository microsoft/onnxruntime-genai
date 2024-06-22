# Debugging Notes for Windows

## To debug using VS Code.

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
        "vmArgs": [ "-Djava.library.path=D:\\src\\github\\ort.genai\\build\\Windows\\Debug\\src\\java\\native-lib\\ai\\onnxruntime-genai\\native\\win-x64" ],
    },
```

You may also want to set this in the VS Code settings: 
```yaml
    "java.debug.settings.onBuildFailureProceed": true,
```

I didn't try to setup VS Code to be able to build the tests using cmake or gradle, so the build VS Code attempts 
before run/debug of a test always fails. Instead I built the binding/test code from the command line, and then debugged
the tests from VS Code.

You can do a top level build (`./build --build_java --config Debug --build --test ...` from the repo root), 
or manually run gradlew from the src/java directory.

e.g. the gradlew command to build looks something like this. Adjust build output paths as needed.
> D:\src\github\ort.genai\src\java>./gradlew cmakeBuild '-DcmakeBuildDir=D:/src/github/ort.genai/build/Windows/Debug/src/java' '-DnativeLibDir=D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime\genai\native\win-x64' '-Dorg.gradle.daemon=false' 

e.g. the gradlew command to test looks something like this. Adjust build output path as needed.
> D:\src\github\ort.genai\src\java>D:/src/github/ort.genai/src/java/gradlew --info cmakeCheck -DcmakeBuildDir="D:\src\github\ort.genai\build\Windows\Debug\src\java" -DnativeLibDir="D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime-genai\native\win-x64" -Dorg.gradle.daemon=false

Either of these commands can have ':spotlessApply' appended to them to automatically format the code as per the coding standards.

NOTE: If using the top-level build, the unit test code gets built in the 'test' phase - that's just how the gradle build is setup.

## To debug using IntelliJ

The test Debug/Run config for the test needs the cmakeBuildDir and nativeLibDir values to be added.

Easiest way to create a test config is to right-click on the test or test directory in the Project window and run it.
The run will fail, but the bulk of the configuration will be created for you.

Now open the test configuration, and in the 'Run' command which will start with something like
  `:test --tests "ai.onnxruntime.genai..."` 
add the values for cmakeBuildDir and nativeLibDir. 

e.g.
`"-DcmakeBuildDir=D:\src\github\ort.genai\build\Windows\Debug\src\java -DnativeLibDir=D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime-genai\native\win-x64`

# Debugging native code

Download a junit-platform-console-standalone jar file from https://central.sonatype.com/artifact/org.junit.platform/junit-platform-console-standalone/versions

With that the magic incantation to run the tests from the command line (on Windows at least) is...

All tests:

> D:\Java\jdk-11.0.17\bin\java.exe "-Djava.library.path=D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime-genai\native\win-x64" -jar junit-platform-console-standalone-1.10.2.jar -cp D:\src\github\ort.genai\src\java\build\classes\java\test -cp D:\src\github\ort.genai\src\java\build\resources\test  -cp D:\src\github\ort.genai\src\java\build\classes\java\main --scan-classpath

Specific test class uses `-c` and the full class name. e.g.

> D:\Java\jdk-11.0.17\bin\java.exe "-Djava.library.path=D:\src\github\ort.genai\build\Windows\Debug\src\java\native-lib\ai\onnxruntime-genai\native\win-x64" -jar junit-platform-console-standalone-1.10.2.jar -cp D:\src\github\ort.genai\src\java\build\classes\java\test -cp D:\src\github\ort.genai\src\java\build\resources\test  -cp D:\src\github\ort.genai\src\java\build\classes\java\main -c ai.onnxruntime.genai.GenerationTest

Adjust the paths for your setup. Run from the java build output directory: e.g. D:\src\github\ort.genai\build\Windows\Debug\src\java 

That command can also be run from Visual Studio using the solution file for the native library you need to debug (onnxruntime-genai or onnxruntime) by setting the debug command, arguments and working directory in the project properties. 

e.g. to debug the `onnxruntime` project (which builds the onnxruntime shared library) in onnxruntime.sln, in the project properties for the `onnxruntime` project, under Debugging, set Command/Command Arguments/Working Directory to the above values. 
You can then right-click on the `onnxruntime` project -> Debug -> Start new instance. That should run java.exe and let you break on any exceptions with full symbols for the native code.

To also be able to set breakpoints, make sure a local debug build of the library is in the nativeLibDir so that java.exe is loading that.

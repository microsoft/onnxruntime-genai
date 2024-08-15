# Updating Java Bindings

## Overview

The Java bindings expose the GenAI C API to Java. The starting point for determining new things to add is /src/ort_genai_c.h.
Guidance for how to add them is to look at how the python and C# bindings have been updated.
Keep things consistent across the language bindings in terms of classes and naming where possible.
It's probably easiest to compare to the C# classes. Pay attention to the C# class API and not so much about how they're implemented as the implementation may differ significantly across the languages in terms of types, memory management and error handling.

Note: 
- Directory names beginning with '/' are relative to the root of the repository.
- Directory names that don't begin with '/' are relative to the /src/java directory.

## Development Environment

VS Code, the free version of IntelliJ, Eclipse or whatever your Java IDE of choice is can be used. 
Load the /src/java directory as a project. 

Build/debug instructions are in Debugging.md for VS Code and IntelliJ. 

Note: It can be useful to copy ort_genai_c.h to the src/java/src/main/native directory temporarily. 
That at least lets VS Code provide intellisense for the C API when implementing JNI code as it can't determine the 
include path to find the header from the CMakeLists.txt file.

## Adding a new class
### Java
The Java class should be in a .java file named after the class in src/main/java/ai/onnxruntime/genai. 
Use existing classes as a guide.

### JNI C++ code
The Java class will have a matching .cpp file in src/main/native. 
The file name format is based on the Java class. 
e.g. ai_onnxruntime_genai_Generator.cc is the name for the JNI implementation of the Generator class in the ai.onnxruntime.genai package.

This is the format the Java build uses by default. Use the existing files as an example.

## Adding a new function
### Java
Add a new method to the relevant Java class and a new `native` function to call the GenAI C API function. 
- If the most sensible function name clashes with the Java class's method name, append 'Native' to the native function name
so there's no potential confusion about what the code is doing caused by having a class method and a native function with exactly the same names.
  - e.g. In the Generator class, 'generateNextToken' is the most sensible name at the Java and native level, so the Java function is called 'generateNextToken' and the native function is called 'generateNextTokenNative'.

Any addresses (e.g. for a newly created object) are passed between Java and JNI as a `long` and cast to the correct type.

The GenAI API will typically have a 'create' and 'destroy' function for the object being wrapped by the class.

Keep the code consistent with the existing classes.

The result of the 'create' is stored in a private member
- `private long nativeHandle = 0;`
The constructor should set this.
Other methods will pass this as the first argument to the native function.
- This is equivalent to using it as the `this` pointer of a C++ class.

Classes should be `AutoCloseable` and have a `close` method that calls the destroy function.

All classes should have a static init to ensure the native libraries are fully loaded upfront. This avoids cryptic errors.

Cut-and-paste this if adding a new class:
```java
  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }
```

### JNI C++ code
First compile the Java code with the `native` functions defined. No other implementation is needed at that point.
A header file for each class with the JNI function signatures will be generated in /src/java/build/headers.

To create/update the .cpp file that implements the JNI function (which will be located in src/main/native), 
cut-and-paste the relevant parts of the header into the .cpp file.

If creating the .cpp file, add a `#include` for the corresponding generated header file.
While not strictly necessary, doing so allows us to not have to explicitly specify the correct language linkage
(`extern "C"`) for the JNI functions as the linkage is inherited from the earlier declarations in the header.

Update the first 2 parameters of each new function to be meaningful by adding the parameter names
    Generated:  `JNIEnv*, jobject`
    Meaningful: `JNIEnv* env, jobject thiz`

The 'thiz' is the Java class instance that the function is being called on. 
It's rarely required in the JNI code as we're not trying to manipulate the Java object from the JNI level.

In the JNI native code function implementations the first step is to reinterpret_cast to the GenAI object type. 
Please keep things const correct (i.e. reinterpret_cast to a const pointer wherever possible).

There are helpers to check the OgaResult (`ThrowIfError`) and to throw exceptions (`ThrowException`). 
Exception handling is slightly counter-intuitive as you need to manually return from the JNI function. 
Calling `ThrowIfError` or `ThrowException` will create the exception that the Java level will receive, 
but the rest of the JNI function will execute before that happens.

That translates to 'if ThrowIfError or ThrowException are not the last thing in the function, you must check the return value (if ThrowIfError) and call `return ...;` if an exception was created.

The `CString` class provides a helper for converting between Java and C++ strings and managing memory.
This greatly simplifies the JNI code where strings are involved.

With all of the above, see existing code for examples.

## Testing

All new code should be covered by unit tests.
The tests should validate that the bindings work and handle expected potential misuse (e.g. invalid arguments) 
and edge cases that should be handled in the Java or JNI code. 
Validating the correctness of the GenAI API implementation is not the responsibility of the Java bindings unit tests.


## Formatting

https://github.com/microsoft/onnxruntime/blob/main/docs/Coding_Conventions_and_Standards.md
https://google.github.io/styleguide/javaguide.html

Easiest way to format the Java code is to run the gradlew task with `:spotlessApply`. See Debugging.md for the full commands.
Easiest way to format the C++ code is to use clang-format with /.clang-format.
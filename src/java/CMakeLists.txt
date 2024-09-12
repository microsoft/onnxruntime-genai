# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
include(FindJava)
find_package(Java REQUIRED)
include(UseJava)

if (NOT ANDROID)
  find_package(JNI REQUIRED)
endif()

set(JAVA_SRC_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
# <build output dir>/src/java (path used with add_subdirectory in root CMakeLists.txt)
set(JAVA_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})  

# Should we use onnxruntime-genai or onnxruntime-genai-static? Using onnxruntime-genai for now. 
# Add dependency on native target
set(JAVA_DEPENDS onnxruntime-genai)

set(GRADLE_EXECUTABLE "${JAVA_SRC_ROOT}/gradlew")

file(GLOB_RECURSE genai4j_gradle_files "${JAVA_SRC_ROOT}/*.gradle")
file(GLOB_RECURSE genai4j_srcs "${JAVA_SRC_ROOT}/src/main/java/ai/onnxruntime-genai/*.java")

# set gradle options that are used with multiple gradle commands
if(WIN32)
  set(GRADLE_OPTIONS --console=plain -Dorg.gradle.daemon=false)
elseif (ANDROID)
  # For Android build, we may run gradle multiple times in same build. Sometimes gradle JVM will run out of memory
  # if we keep the daemon running, so we use no-daemon to avoid that
  set(GRADLE_OPTIONS --console=plain --no-daemon)
endif()

# this jar is solely used to signaling mechanism for dependency management in CMake
# if any of the Java sources change, the jar (and generated headers) will be regenerated 
# and the onnxruntime-genai-jni target will be rebuilt
set(JAVA_OUTPUT_JAR ${JAVA_OUTPUT_DIR}/build/libs/onnxruntime-genai.jar)
set(GRADLE_ARGS clean jar -x test)

# this jar is solely used to signaling mechanism for dependency management in CMake
# if any of the Java sources change, the jar (and generated headers) will be regenerated 
# and the onnxruntime-genai-jni target will be rebuilt
set(JAVA_OUTPUT_JAR ${JAVA_SRC_ROOT}/build/libs/onnxruntime-genai.jar)
set(GRADLE_ARGS clean jar -x test)

add_custom_command(OUTPUT ${JAVA_OUTPUT_JAR} 
                   COMMAND ${GRADLE_EXECUTABLE} ${GRADLE_OPTIONS} ${GRADLE_ARGS} 
                   WORKING_DIRECTORY ${JAVA_SRC_ROOT} 
                   DEPENDS ${genai4j_gradle_files} ${genai4j_srcs})
add_custom_target(onnxruntime-genai4j DEPENDS ${JAVA_OUTPUT_JAR})

set_source_files_properties(${JAVA_OUTPUT_JAR} PROPERTIES GENERATED TRUE)
set_property(TARGET onnxruntime-genai4j APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${JAVA_OUTPUT_DIR}")

# Specify the JNI native sources
file(GLOB genai4j_native_src
    "${JAVA_SRC_ROOT}/src/main/native/*.cpp"
    "${JAVA_SRC_ROOT}/src/main/native/*.h"
    "${SRC_ROOT}/ort_genai_c.h"
    )
    
add_library(onnxruntime-genai-jni SHARED ${genai4j_native_src})
set_property(TARGET onnxruntime-genai-jni PROPERTY CXX_STANDARD 17)
add_dependencies(onnxruntime-genai-jni onnxruntime-genai4j)
# the JNI headers are generated in the genai4j target
target_include_directories(onnxruntime-genai-jni PRIVATE ${SRC_ROOT}
                                                           ${JAVA_SRC_ROOT}/build/headers 
                                                           ${JNI_INCLUDE_DIRS})
target_link_libraries(onnxruntime-genai-jni PUBLIC onnxruntime-genai)

set(JAVA_PACKAGE_OUTPUT_DIR ${JAVA_OUTPUT_DIR}/build)
file(MAKE_DIRECTORY ${JAVA_PACKAGE_OUTPUT_DIR})

if (WIN32)
  set(JAVA_PLAT "win")
elseif (APPLE)
  set(JAVA_PLAT "osx")
elseif (LINUX)
  set(JAVA_PLAT "linux")
elseif (ANDROID)
  set(JAVA_PLAT "android")
else()
  message(FATAL_ERROR "GenAI with Java is not currently supported on this platform")
endif()

# Set platform and arch for packaging
if (genai_target_platform STREQUAL "x64")
  set(JNI_ARCH x64)
elseif (genai_target_platform STREQUAL "arm64")
  set(JNI_ARCH aarch64)
else()
  message(FATAL_ERROR "GenAI with Java is not currently supported on this platform")
endif()

# Similar to Nuget schema
set(JAVA_OS_ARCH ${JAVA_PLAT}-${JNI_ARCH})

# expose native libraries to the gradle build process
set(JAVA_NATIVE_LIB_DIR ${JAVA_OUTPUT_DIR}/native-lib)
set(JAVA_PACKAGE_LIB_DIR ${JAVA_NATIVE_LIB_DIR}/ai/onnxruntime/genai/native/${JAVA_OS_ARCH})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_LIB_DIR})

# Add the native genai library to the native-lib dir
add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                      $<TARGET_FILE:onnxruntime-genai> 
                      ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime-genai>)

# Add the JNI bindings to the native-jni dir
add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                      $<TARGET_FILE:onnxruntime-genai-jni>
                      ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_FILE_NAME:onnxruntime-genai-jni>)

if (${CMAKE_HOST_WIN32})
  # we need to use `call` with gradlew on Windows otherwise the POST_BUILD commands will be prematurely terminated
  # after the first gradlew command.
  set(POST_BUILD_GRADLE_EXECUTABLE "call" ${GRADLE_EXECUTABLE})
else()
  set(POST_BUILD_GRADLE_EXECUTABLE ${GRADLE_EXECUTABLE})
endif()

# run the build process
set(GRADLE_ARGS cmakeBuild -DcmakeBuildDir=${JAVA_OUTPUT_DIR} -DnativeLibDir=${JAVA_PACKAGE_LIB_DIR})
add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                   COMMAND ${POST_BUILD_GRADLE_EXECUTABLE} ${GRADLE_OPTIONS} ${GRADLE_ARGS}
                   WORKING_DIRECTORY ${JAVA_SRC_ROOT})

if (ANDROID)
  # <build dir>/src/java/build/android
  set(ANDROID_PACKAGE_OUTPUT_DIR ${JAVA_PACKAGE_OUTPUT_DIR}/android)
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_OUTPUT_DIR})

  # dirs to assemble the AAR contents in. <build dir>/src/java/android
  set(ANDROID_PACKAGE_DIR ${JAVA_OUTPUT_DIR}/android)
  set(ANDROID_PACKAGE_HEADERS_DIR ${ANDROID_PACKAGE_DIR}/headers)
  set(ANDROID_PACKAGE_ABI_DIR ${ANDROID_PACKAGE_DIR}/${ANDROID_ABI})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_HEADERS_DIR})
  file(MAKE_DIRECTORY ${ANDROID_PACKAGE_ABI_DIR})

  # copy C/C++ API headers to be packed into Android AAR package  
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD 
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_ROOT}/ort_genai.h ${ANDROID_PACKAGE_HEADERS_DIR})
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD 
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SRC_ROOT}/ort_genai_c.h ${ANDROID_PACKAGE_HEADERS_DIR})

  # Copy onnxruntime-genai.so and onnxruntime-genai-jni.so for building Android AAR package
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                       $<TARGET_FILE:onnxruntime-genai>
                       ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime-genai>)

  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                       $<TARGET_FILE:onnxruntime-genai-jni> 
                       ${ANDROID_PACKAGE_ABI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime-genai-jni>)

  # Generate the Android AAR package
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E echo "Generating Android AAR package..."
                     COMMAND ${POST_BUILD_GRADLE_EXECUTABLE}
                       build
                       -b build-android.gradle -c settings-android.gradle
                       -DjniLibsDir=${ANDROID_PACKAGE_DIR} -DbuildDir=${ANDROID_PACKAGE_OUTPUT_DIR}
                       -DminSdkVer=${ANDROID_MIN_SDK} -DheadersDir=${ANDROID_PACKAGE_HEADERS_DIR}
                     WORKING_DIRECTORY ${JAVA_SRC_ROOT})

  # unit tests
  set(ANDROID_TEST_SRC_ROOT ${JAVA_SRC_ROOT}/src/test/android)
  set(ANDROID_TEST_PACKAGE_DIR ${JAVA_OUTPUT_DIR}/androidtest)

  # copy the android test project to the build output so we can assemble all the pieces to build it
  file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_DIR})
  file(GLOB android_test_files "${ANDROID_TEST_SRC_ROOT}/*")
  file(COPY ${android_test_files} DESTINATION ${ANDROID_TEST_PACKAGE_DIR})

  set(ANDROID_TEST_PACKAGE_LIB_DIR ${ANDROID_TEST_PACKAGE_DIR}/app/libs)  
  set(ANDROID_TEST_PACKAGE_APP_ASSETS_DIR ${ANDROID_TEST_PACKAGE_DIR}/app/src/main/assets)
  file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_LIB_DIR})
  file(MAKE_DIRECTORY ${ANDROID_TEST_PACKAGE_APP_ASSETS_DIR})

  # Copy the test model to the assets folder in the test app
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
                       ${REPO_ROOT}/test/test_models/hf-internal-testing/tiny-random-gpt2-fp32
                       ${ANDROID_TEST_PACKAGE_APP_ASSETS_DIR}/model)

  # Copy the Android AAR package we built to the libs folder of our test app
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                       ${ANDROID_PACKAGE_OUTPUT_DIR}/outputs/aar/onnxruntime-genai-debug.aar
                       ${ANDROID_TEST_PACKAGE_LIB_DIR}/onnxruntime-genai.aar)

  # Build Android test apk for java package
  add_custom_command(TARGET onnxruntime-genai-jni POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E echo "Building and running Android test for Android AAR package..."
                     COMMAND ${POST_BUILD_GRADLE_EXECUTABLE} clean assembleDebug assembleDebugAndroidTest
                       -DminSdkVer=${ANDROID_MIN_SDK} --stacktrace
                     WORKING_DIRECTORY ${ANDROID_TEST_PACKAGE_DIR})
endif()

if (ENABLE_TESTS)
  message(STATUS "Adding Java tests")
  if (WIN32)
    # On windows ctest requires a test to be an .exe(.com) file
    # With gradle wrapper we get gradlew.bat. We delegate execution to a separate .cmake file
    # That can handle both .exe and .bat
    add_test(NAME onnxruntime-genai4j_test 
             COMMAND ${CMAKE_COMMAND} 
               -DGRADLE_EXECUTABLE=${GRADLE_EXECUTABLE} 
               -DBIN_DIR=${JAVA_OUTPUT_DIR} 
               -DJAVA_SRC_ROOT=${JAVA_SRC_ROOT}
               -DJAVA_PACKAGE_LIB_DIR=${JAVA_PACKAGE_LIB_DIR}
               -P ${JAVA_SRC_ROOT}/windows-unittests.cmake)
  else()
    add_test(NAME onnxruntime-genai4j_test 
             COMMAND ${GRADLE_EXECUTABLE} cmakeCheck 
               -DcmakeBuildDir=${JAVA_OUTPUT_DIR} -DnativeLibDir=${JAVA_PACKAGE_LIB_DIR}
             WORKING_DIRECTORY ${JAVA_SRC_ROOT})
  endif()

  set_property(TEST onnxruntime-genai4j_test APPEND PROPERTY DEPENDS onnxruntime-genai-jni)
endif()

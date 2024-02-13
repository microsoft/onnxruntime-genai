# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.26)
project(Generators LANGUAGES C CXX)
# All options are defined in cmake/options.cmake
include(cmake/options.cmake)
# All external dependencies are defined in cmake/external
include(cmake/external/onnxruntime_external_deps.cmake)
# All top-level variables are defined in cmake/variables.cmake
include(cmake/variables.cmake)
# Checking if CUDA is supported
include(CheckLanguage)
if(USE_CUDA)
  check_language(CUDA)
endif()
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 9)
  message(FATAL_ERROR "GCC version must be greater than or equal to 9")
endif()

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 10)
  add_compile_definitions(USE_CXX17=1)
  message("USE C++17")
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 20)
endif()

# CUDA Being enabled will make it not a debug build without this option, so all of the C++ headers will complain
# about a mismatch with the actual debug headers and it'll fail to link. I don't know why this happens, or if this is the best fix.
if(USE_CUDA AND CMAKE_CUDA_COMPILER AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(_DEBUG=1)
endif()

file(GLOB onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_FILES}")

message("------------------Building Generators------------------")
add_subdirectory("${CMAKE_SOURCE_DIR}/src")

message("------------------Building Tests------------------")
add_subdirectory("${CMAKE_SOURCE_DIR}/test")
# Visual C++ - /MP Multiple process build, /WX Warnings as errors
# Enable:
#   C5038: data member 'member1' will be initialized after data member 'member2'
# Disable:
#   C4100: 'this_': unreferenced formal parameter
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /w15038 /wd4100 /W4 /WX")
endif()




# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.26)
include(FetchContent)
include(CMakeDependentOption)
project(Generators LANGUAGES C CXX)
option(USE_CUDA "Build with CUDA support" ON)
option(USE_TOKENIZER "Build with Tokenizer support" ON)
option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_PYTHON "Enable python buildings" ON)
cmake_dependent_option(BUILD_WHEEL "Build the python wheel" ON "ENABLE_PYTHON" OFF)

include(cmake/external/onnxruntime_external_deps.cmake)
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
# Visual C++ - /MP Multiple process build, /WX Warnings as errors
# Enable:
#   C5038: data member 'member1' will be initialized after data member 'member2'
# Disable:
#   C4100: 'this_': unreferenced formal parameter
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /w15038 /wd4100 /W4 /WX")
endif()

set(ORT_HEADER_DIR ${CMAKE_SOURCE_DIR}/ort/include)
set(ORT_LIB_DIR ${CMAKE_SOURCE_DIR}/ort/lib)

# CUDA Being enabled will make it not a debug build without this option, so all of the C++ headers will complain
# about a mismatch with the actual debug headers and it'll fail to link. I don't know why this happens, or if this is the best fix.
if(USE_CUDA AND CMAKE_CUDA_COMPILER AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(_DEBUG=1)
endif()

if(WIN32)
  set(ONNXRUNTIME_LIB "onnxruntime.lib")
  set(ONNXRUNTIME_FILES "onnxruntime*.dll")
  set(ONNXRUNTIME_EXTENSIONS_LIB "tfmtok_c.lib")
  set(ONNXRUNTIME_EXTENSIONS_FILES "tfmtok_c.dll")
elseif(APPLE)
  set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
  set(ONNXRUNTIME_FILES "libonnxruntime*.dylib")
else()
  set(ONNXRUNTIME_LIB "libonnxruntime.so")
  set(ONNXRUNTIME_FILES "libonnxruntime*.so*")
  set(ONNXRUNTIME_EXTENSIONS_LIB "tfmtok_c.so")
endif()

if(USE_TOKENIZER)
  add_subdirectory("${CMAKE_SOURCE_DIR}/src/tokenizer")
  message("------------------Using Tokenizer------------------")
endif()

add_subdirectory("${CMAKE_SOURCE_DIR}/src")
message("------------------Building Generators------------------")

if(ENABLE_TESTS)
  add_subdirectory("${CMAKE_SOURCE_DIR}/test")
  message("------------------Enabling tests------------------")
endif()

if(ENABLE_PYTHON)
  add_subdirectory("${CMAKE_SOURCE_DIR}/src/python")
  message("------------------Enabling Python Wheel------------------")
endif()

# Visual C++ - /MP Multiple process build, /WX Warnings as errors
# Enable:
#   C5038: data member 'member1' will be initialized after data member 'member2'
# Disable:
#   C4100: 'this_': unreferenced formal parameter
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /w15038 /wd4100 /W4 /WX")
endif()




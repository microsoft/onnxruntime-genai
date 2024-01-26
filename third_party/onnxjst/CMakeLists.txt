# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.18)
project(onnxjst LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fetch Content of simdjson
include(FetchContent)

FetchContent_Declare(
  simdjson
  GIT_REPOSITORY https://github.com/simdjson/simdjson.git
  GIT_TAG v3.6.3
)

FetchContent_MakeAvailable(simdjson)

file(GLOB JST_BASE_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/base/*.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/base/*.h
)

file(GLOB SIMDJSON_SRC
  ${simdjson_SOURCE_DIR}/singleheader/simdjson.h
  ${simdjson_SOURCE_DIR}/singleheader/simdjson.cpp
)

add_library(onnxjst STATIC
  ${JST_BASE_SRC}
  ${SIMDJSON_SRC}
)

target_include_directories(onnxjst PUBLIC ${simdjson_SOURCE_DIR}/singleheader/)

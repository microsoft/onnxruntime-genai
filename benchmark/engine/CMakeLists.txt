# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
)
FetchContent_MakeAvailable(nlohmann_json)

set(engine_benchmark_srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/scenario_dispatcher.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/benchmark_types.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/scenario_base.h
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/scenario_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/decode_baseline.h
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/decode_baseline.cpp
)

add_executable(engine_benchmark ${engine_benchmark_srcs})

target_include_directories(engine_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
  ${ORT_HEADER_DIR}
)

target_compile_features(engine_benchmark PRIVATE cxx_std_17)
target_compile_definitions(engine_benchmark PRIVATE ORT_GENAI_BENCH_GENAI_VERSION="${VERSION_INFO}")

target_link_libraries(engine_benchmark PRIVATE
  onnxruntime-genai
  ${ONNXRUNTIME_LIB}
  nlohmann_json::nlohmann_json
)

target_link_directories(engine_benchmark PRIVATE ${ORT_LIB_DIR})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${engine_benchmark_srcs})

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
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/scenario_base.h
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/scenario_base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/decode_baseline.h
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/decode_baseline.cpp
)

add_executable(engine_benchmark ${engine_benchmark_srcs})

target_include_directories(engine_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${nlohmann_json_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)

target_compile_features(engine_benchmark PRIVATE cxx_std_17)
target_compile_definitions(engine_benchmark PRIVATE ORT_GENAI_BENCH_GENAI_VERSION="${VERSION_INFO}")

target_link_libraries(engine_benchmark PRIVATE
  onnxruntime-genai
)

target_link_directories(engine_benchmark PRIVATE ${ORT_LIB_DIR})

# The benchmark runs against the pinned libraries staged next to the executable, so keep the link
# directories out of the RPATH; otherwise the build tree's ONNX Runtime wins over $ORIGIN.
if(UNIX AND NOT APPLE)
  set_target_properties(engine_benchmark PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
  )
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${engine_benchmark_srcs})

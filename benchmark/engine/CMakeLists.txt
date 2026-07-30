cmake_minimum_required(VERSION 3.16)
project(engine_benchmark_prototype LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ORT_GENAI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src" CACHE PATH "Directory containing ort_genai.h")
set(ORT_GENAI_LIB_DIR "" CACHE PATH "Directory containing onnxruntime-genai.lib and onnxruntime.lib")

set(engine_benchmark_srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmark.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scenarios/decode_baseline.cpp
)

add_executable(engine_benchmark_prototype ${engine_benchmark_srcs})

target_include_directories(engine_benchmark_prototype PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ORT_GENAI_INCLUDE_DIR}
)

find_library(ONNXRUNTIME_GENAI_LIBRARY NAMES onnxruntime-genai HINTS ${ORT_GENAI_LIB_DIR})
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime HINTS ${ORT_GENAI_LIB_DIR})

if (NOT ONNXRUNTIME_GENAI_LIBRARY OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "Could not find ONNX Runtime GenAI libraries. Set ORT_GENAI_LIB_DIR.")
endif()

target_link_libraries(engine_benchmark_prototype PRIVATE
  ${ONNXRUNTIME_GENAI_LIBRARY}
  ${ONNXRUNTIME_LIBRARY}
)

if (WIN32 AND ORT_GENAI_LIB_DIR)
  add_custom_command(TARGET engine_benchmark_prototype POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_GENAI_LIB_DIR}/onnxruntime-genai.dll" "$<TARGET_FILE_DIR:engine_benchmark_prototype>/onnxruntime-genai.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_GENAI_LIB_DIR}/onnxruntime.dll" "$<TARGET_FILE_DIR:engine_benchmark_prototype>/onnxruntime.dll"
  )
endif()

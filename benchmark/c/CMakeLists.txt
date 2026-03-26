# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(model_benchmark_srcs
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/options.h
  ${CMAKE_CURRENT_SOURCE_DIR}/options.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/resource_utils.h
)

# add platform-specific source files
if(UNIX)
  list(APPEND model_benchmark_srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/posix/resource_utils.cpp
  )
elseif(WIN32)
  list(APPEND model_benchmark_srcs
       ${CMAKE_CURRENT_SOURCE_DIR}/windows/resource_utils.cpp
  )
endif()

add_executable(model_benchmark ${model_benchmark_srcs})

target_link_libraries(model_benchmark PRIVATE onnxruntime-genai)
target_include_directories(model_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${model_benchmark_srcs})

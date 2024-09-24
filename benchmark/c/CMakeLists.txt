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

target_include_directories(model_benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src  # directory containing the ort_genai headers
)

target_link_libraries(model_benchmark PRIVATE onnxruntime-genai-static ${ONNXRUNTIME_LIB})

target_link_directories(model_benchmark PRIVATE ${ORT_LIB_DIR})

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${model_benchmark_srcs})

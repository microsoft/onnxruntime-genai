# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required (VERSION 3.11)
include(FetchContent)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

FetchContent_Declare(GSL
    GIT_REPOSITORY "https://github.com/microsoft/GSL"
    GIT_TAG "v4.0.0"
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(GSL)

FetchContent_Declare(safeint
    GIT_REPOSITORY "https://github.com/dcleblanc/SafeInt"
)

FetchContent_MakeAvailable(safeint)
# add_library(safeint_interface INTERFACE)
# target_include_directories(safeint_interface INTERFACE ${safeint_SOURCE_DIR})
# message(${safeint_SOURCE_DIR})

project ("Generators")

include_directories(Generators ${CMAKE_SOURCE_DIR}/ort)
link_directories(Generators ${CMAKE_SOURCE_DIR}/ort)

# Add source to this project's executable.
add_executable (Generators "Generators.cpp" "Generators.h" "main.cpp" "Tests.cpp" "greedy_search_impl_base.h"  "Sequences.h" "TensorShape.h" "span_utils.h" "generate_impl_base.h" "logits_processor.h" "generation_device_helper.cpp" "generation_device_helper.h" "greedy_search_impl_gpt.h" "feeds_fetches_manager.h" "subgraph_gpt.h" "subgraph_base.h" "logits_processor.cpp")
target_link_libraries(Generators PRIVATE "onnxruntime.lib" Microsoft.GSL::GSL)
target_include_directories(Generators PRIVATE ${safeint_SOURCE_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/ort/onnxruntime.dll DESTINATION ${CMAKE_BINARY_DIR})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Generators PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.

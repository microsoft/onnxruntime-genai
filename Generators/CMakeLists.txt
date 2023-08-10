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

project (Generators LANGUAGES CXX)
set(USE_CUDA ON)

include_directories(Generators ${CMAKE_SOURCE_DIR}/ort)
link_directories(Generators ${CMAKE_SOURCE_DIR}/ort)

set(GENERATORS_ROOT ${PROJECT_SOURCE_DIR})

message(STATUS "Adding source files")

file(GLOB generator_srcs CONFIGURE_DEPENDS
   "${GENERATORS_ROOT}/*.h"
   "${GENERATORS_ROOT}/*.cpp"
)

if(USE_CUDA)
    # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
    cmake_policy(SET CMP0104 OLD)
    enable_language(CUDA)
    message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

    file(GLOB generator_cuda_srcs CONFIGURE_DEPENDS
        "${GENERATORS_ROOT}/*.cu"
    )
    list(APPEND generator_srcs ${generator_cuda_srcs})
endif()

add_executable (Generators ${generator_srcs})

#set_target_properties(Generators PROPERTIES LINKER_LANGUAGE cxx)
target_link_libraries(Generators PRIVATE "onnxruntime.lib" Microsoft.GSL::GSL)
target_include_directories(Generators PRIVATE ${safeint_SOURCE_DIR})
target_include_directories(Generators PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(USE_CUDA)
    target_link_libraries(Generators PRIVATE CUDA)
    add_compile_definitions(USE_CUDA=1)

    # Why is this needed? Without these, the cuda build causes weird errors due to mismatches in these settings
    add_compile_definitions(_ITERATOR_DEBUG_LEVEL=0)
    set(CMAKE_CXX_FLAGS_RELEASE "/MT")
    set(CMAKE_CXX_FLAGS_DEBUG "/MDd")
 endif()

file(COPY ${CMAKE_SOURCE_DIR}/ort/onnxruntime.dll DESTINATION ${CMAKE_BINARY_DIR})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Generators PROPERTY CXX_STANDARD 20)
endif()

set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Generators)

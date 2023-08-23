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

set(CMAKE_BUILD_TYPE Debug)

project (Generators LANGUAGES CXX)
set(GENERATORS_ROOT ${PROJECT_SOURCE_DIR})

set(USE_CUDA 1)
add_compile_definitions(_DEBUG=1)

message(STATUS "Adding source files")

file(GLOB generator_srcs CONFIGURE_DEPENDS
   "${GENERATORS_ROOT}/*.h"
   "${GENERATORS_ROOT}/*.cpp"
)

if(USE_CUDA)
    # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
    # cmake_policy(SET CMP0104 OLD)
    enable_language(CUDA)
    message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    # set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2803 --expt-relaxed-constexpr")

    file(GLOB generator_cuda_srcs CONFIGURE_DEPENDS
        "${GENERATORS_ROOT}/*.cu"
    )
    list(APPEND generator_srcs ${generator_cuda_srcs})
    add_compile_definitions(USE_CUDA=1)
endif()

add_executable (Generators ${generator_srcs})

target_link_directories(Generators PRIVATE ${CMAKE_SOURCE_DIR}/ort)
target_include_directories(Generators PRIVATE ${CMAKE_SOURCE_DIR}/ort)

target_link_libraries(Generators PRIVATE "onnxruntime.lib" Microsoft.GSL::GSL)
target_include_directories(Generators PRIVATE ${safeint_SOURCE_DIR})

if(MSVC)
    target_compile_options(Generators PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/MP>")
endif()

 # Copy the onnxruntime binaries into the build folder so it's found on launch
file(GLOB onnxruntime_libs "${CMAKE_SOURCE_DIR}/ort/onnxruntime*.dll")
foreach(DLL_FILE ${onnxruntime_libs})
    add_custom_command(
        TARGET Generators POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${DLL_FILE}
                ${CMAKE_BINARY_DIR}/$<CONFIG>/${DLL_FILE_NAME}
    )
endforeach()

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET Generators PROPERTY CXX_STANDARD 20)
endif()

set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Generators)

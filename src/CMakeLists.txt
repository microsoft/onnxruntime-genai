# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required (VERSION 3.11)
# include(FetchContent)

# Enable Hot Reload for MSVC compilers if supported.
if(POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

if(false) # We don't use GSL anymore
FetchContent_Declare(GSL
    GIT_REPOSITORY "https://github.com/microsoft/GSL"
    GIT_TAG "v4.0.0"
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(GSL)
endif()

if(false) # We don't use SafeInt anymore
FetchContent_Declare(safeint
    GIT_REPOSITORY "https://github.com/dcleblanc/SafeInt"
)

FetchContent_MakeAvailable(safeint)
endif()

project (Generators LANGUAGES CXX)
set(GENERATORS_ROOT ${PROJECT_SOURCE_DIR})
set(MODELS_ROOT ${PROJECT_SOURCE_DIR}/models)
set(TESTS_ROOT ${PROJECT_SOURCE_DIR}/tests)

set(USE_CUDA 1)

# CUDA Being enabled will make it not a debug build without this option, so all of the C++ headers will complain
# about a mismatch with the actual debug headers and it'll fail to link. I don't know why this happens, or if this is the best fix.
if(USE_CUDA EQUAL 1 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(_DEBUG=1)
endif()

message(STATUS "Adding source files")

file(GLOB generator_srcs CONFIGURE_DEPENDS
   "${GENERATORS_ROOT}/*.h"
   "${GENERATORS_ROOT}/*.cpp"
)

file(GLOB model_srcs CONFIGURE_DEPENDS
   "${MODELS_ROOT}/*.h"
   "${MODELS_ROOT}/*.cpp"
)

file(GLOB test_srcs CONFIGURE_DEPENDS
   "${TESTS_ROOT}/*.h"
   "${TESTS_ROOT}/*.cpp"
)

if(USE_CUDA)
    # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
    cmake_policy(SET CMP0104 OLD)
    enable_language(CUDA)
    message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    # set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2803 --expt-relaxed-constexpr")

    file(GLOB generator_cuda_srcs CONFIGURE_DEPENDS
        "${GENERATORS_ROOT}/*.cu"
    )
    list(APPEND generator_srcs ${generator_cuda_srcs})

    file(GLOB model_cuda_srcs CONFIGURE_DEPENDS
        "${MODELS_ROOT}/*.cu"
    )
    list(APPEND model_srcs ${model_cuda_srcs})

    add_compile_definitions(USE_CUDA=1)
endif()

add_library (Models ${model_srcs})
target_include_directories(Models PRIVATE ${CMAKE_SOURCE_DIR}/ort)

add_library (Generators ${generator_srcs})
target_include_directories(Generators PRIVATE ${CMAKE_SOURCE_DIR}/ort)

add_executable (Tests ${test_srcs})
target_include_directories(Tests PRIVATE ${CMAKE_SOURCE_DIR}/ort)
target_link_directories(Tests PRIVATE ${CMAKE_SOURCE_DIR}/ort)
target_link_libraries(Tests PRIVATE Generators Models "onnxruntime.lib")
# Microsoft.GSL::GSL
# target_include_directories(Generators PRIVATE ${safeint_SOURCE_DIR})

# Turn on multiple processor build option for Visual C++
if(MSVC)
    target_compile_options(Tests PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/MP>")
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
  set_property(TARGET Models PROPERTY CXX_STANDARD 20)
  set_property(TARGET Tests PROPERTY CXX_STANDARD 20)
endif()

source_group("Sources" FILES ${generator_srcs})
source_group("Sources" FILES ${model_srcs})
source_group("Sources" FILES ${test_srcs})

set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Tests)

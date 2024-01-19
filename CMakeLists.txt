# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required (VERSION 3.11)
# include(FetchContent)
# Checking if CUDA is supported
find_package(CUDA)

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

set(CMAKE_CXX_STANDARD 20)
# add_compile_definitions(USE_CXX17=1)

set(GENERATORS_ROOT ${PROJECT_SOURCE_DIR}/src)
set(MODELS_ROOT ${PROJECT_SOURCE_DIR}/src/models)
set(TESTS_ROOT ${PROJECT_SOURCE_DIR}/src/tests)
set(PYBIND_ROOT ${PROJECT_SOURCE_DIR}/src/pybind)

set(USE_CUDA 1)

add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11 ${PROJECT_SOURCE_DIR}/build/pybind)

# CUDA Being enabled will make it not a debug build without this option, so all of the C++ headers will complain
# about a mismatch with the actual debug headers and it'll fail to link. I don't know why this happens, or if this is the best fix.
if(USE_CUDA EQUAL 1 AND CUDA_FOUND AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(_DEBUG=1)
endif()

message(STATUS "Adding source files")

file(GLOB generator_srcs CONFIGURE_DEPENDS
   "${GENERATORS_ROOT}/*.h"
   "${GENERATORS_ROOT}/*.cpp"
   "${MODELS_ROOT}/*.h"
   "${MODELS_ROOT}/*.cpp"
)

file(GLOB test_srcs CONFIGURE_DEPENDS
   "${TESTS_ROOT}/*.h"
   "${TESTS_ROOT}/*.cpp"
)

file(GLOB pybind_srcs CMAKE_CONFIGURE_DEPENDS
   "${PYBIND_ROOT}/*.h"
   "${PYBIND_ROOT}/*.cpp"
)

if(USE_CUDA AND CUDA_FOUND)
    # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
    cmake_policy(SET CMP0104 OLD)
    enable_language(CUDA)
    message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
    # set(CUDA_PROPAGATE_HOST_FLAGS ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2803 --expt-relaxed-constexpr")

    file(GLOB generator_cuda_srcs CONFIGURE_DEPENDS
        "${GENERATORS_ROOT}/*.cu"
        "${GENERATORS_ROOT}/*.cuh"
        "${MODELS_ROOT}/*.cu"
        "${MODELS_ROOT}/*.cuh"
    )
    list(APPEND generator_srcs ${generator_cuda_srcs})

    add_compile_definitions(USE_CUDA=1)

    include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
else()
    file(GLOB generator_cuda_srcs "${GENERATORS_ROOT}/*_cuda*.*")
    list(REMOVE_ITEM generator_srcs ${generator_cuda_srcs})
endif()

add_library (Generators ${generator_srcs})
target_include_directories(Generators PRIVATE ${CMAKE_SOURCE_DIR}/ort)

if(WIN32)
  set(ONNXRUNTIME_LIB "onnxruntime.lib")
  set(ONNXRUNTIME_FILES "onnxruntime*.dll")
elseif(APPLE)
  set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
  # TODO: Figure ONNXRUNTIME_FILES for MacOS
else()
  set(ONNXRUNTIME_LIB "libonnxruntime.so")
  set(ONNXRUNTIME_FILES "libonnxruntime*.so*")
endif()

add_executable (Tests ${test_srcs})
target_include_directories(Tests PRIVATE ${CMAKE_SOURCE_DIR}/ort)
target_link_directories(Tests PRIVATE ${CMAKE_SOURCE_DIR}/ort)
target_link_libraries(Tests PRIVATE Generators ${ONNXRUNTIME_LIB})

pybind11_add_module(pybind ${pybind_srcs})
target_include_directories(pybind PRIVATE ${CMAKE_SOURCE_DIR}/ort)
target_link_directories(pybind PRIVATE ${CMAKE_SOURCE_DIR}/ort)
target_link_libraries(pybind PRIVATE Generators ${ONNXRUNTIME_LIB})
set_target_properties(pybind PROPERTIES OUTPUT_NAME "onnxruntime_genai")

# Visual C++ - /MP Multiple process build, /WX Warnings as errors
# Enable:
#   C5038: data member 'member1' will be initialized after data member 'member2'
# Disable:
#   C4100: 'this_': unreferenced formal parameter 
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /w15038 /wd4100 /W4 /WX")
endif()

 # Copy the onnxruntime binaries into the build folder so it's found on launch
file(GLOB onnxruntime_libs "${CMAKE_SOURCE_DIR}/ort/${ONNXRUNTIME_FILES}")
foreach(DLL_FILE ${onnxruntime_libs})
    add_custom_command(
        TARGET Generators POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${DLL_FILE}
                ${CMAKE_BINARY_DIR}/$<CONFIG>/${DLL_FILE_NAME}
    )
endforeach()

# Have visual studio put all files into one single folder vs the default split of header files into a separate folder
source_group(TREE ${GENERATORS_ROOT} FILES ${generator_srcs})
source_group("Sources" FILES ${test_srcs})
source_group("Sources" FILES ${pybind_srcs})

set_property (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Tests)

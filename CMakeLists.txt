# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.26)
include(FetchContent)
include(CMakeDependentOption)
project(Generators LANGUAGES C CXX)
# All Options should be defined in cmake/options.cmake
include(cmake/options.cmake)
include(cmake/external/onnxruntime_external_deps.cmake)
include(cmake/cxx_standard.cmake)

file(READ "VERSION_INFO" ver)
set(VERSION_INFO ${ver} CACHE STRING "Set the onnxruntime-genai version info.")
message("Building onnxruntime-genai for version ${VERSION_INFO}")

# Checking if CUDA is supported
include(CheckLanguage)
add_compile_definitions(BUILDING_ORT_GENAI_C)
if(USE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
    cmake_policy(SET CMP0104 OLD)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  else()
    message(FATAL_ERROR "CUDA is not supported")
  endif()
endif()
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 8)
  message(FATAL_ERROR "GCC version must be greater than or equal to 8")
endif()

set(GENERATORS_ROOT ${PROJECT_SOURCE_DIR}/src)
set(MODELS_ROOT ${PROJECT_SOURCE_DIR}/src/models)
set(ORT_HOME ${CMAKE_SOURCE_DIR}/ort CACHE PATH "Path to the onnxruntime root directory.")
set(ORT_HEADER_DIR ${ORT_HOME}/include)
set(ORT_LIB_DIR ${ORT_HOME}/lib)
# CUDA Being enabled will make it not a debug build without this option, so all of the C++ headers will complain
# about a mismatch with the actual debug headers and it'll fail to link. I don't know why this happens, or if this is the best fix.
if(USE_CUDA AND CMAKE_CUDA_COMPILER AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(_DEBUG=1)
endif()

message(STATUS "Adding source files")

file(GLOB generator_srcs CONFIGURE_DEPENDS
  "${GENERATORS_ROOT}/*.h"
  "${GENERATORS_ROOT}/*.cpp"
  "${MODELS_ROOT}/*.h"
  "${MODELS_ROOT}/*.cpp"
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
  # cmake_policy(SET CMP0104 OLD)
  # enable_language(CUDA)
  # message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  # set(CUDA_PROPAGATE_HOST_FLAGS ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2803 --expt-relaxed-constexpr")
  file(GLOB generator_cuda_srcs CONFIGURE_DEPENDS
    "${GENERATORS_ROOT}/*.cu"
    "${GENERATORS_ROOT}/*.cuh"
    "${MODELS_ROOT}/*.cu"
    "${MODELS_ROOT}/*.cuh"
  )
  file(GLOB test_cuda_srcs CONFIGURE_DEPENDS
    "${TESTS_ROOT}/*.cu"
    "${TESTS_ROOT}/*.cuh"
  )
  list(APPEND test_srcs ${test_cuda_srcs})
  list(APPEND generator_srcs ${generator_cuda_srcs})
  add_compile_definitions(USE_CUDA=1)
  include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
elseif(USE_CUDA)
  # USE_CUDA is true but cmake could not find the cuda compiler
  message(FATAL_ERROR "USE_CUDA is ON but no cuda compiler was found.")
else()
  file(GLOB generator_cuda_srcs "${GENERATORS_ROOT}/*_cuda*.*")
  list(REMOVE_ITEM generator_srcs ${generator_cuda_srcs})
endif()

if(ENABLE_TESTS AND TEST_PHI2)
  add_compile_definitions(TEST_PHI2=1)
else()
  add_compile_definitions(TEST_PHI2=0)
endif()

add_library(onnxruntime-genai SHARED ${generator_srcs})
add_library(onnxruntime-genai-static STATIC ${generator_srcs})
target_include_directories(onnxruntime-genai PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai-static PRIVATE ${ORT_HEADER_DIR})

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(onnxruntime-genai PRIVATE "/CETCOMPAT")
  target_compile_options(onnxruntime-genai PRIVATE "/sdl")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(onnxruntime-genai-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(WIN32)
  set(ONNXRUNTIME_LIB "onnxruntime.dll")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "onnxruntime_providers_cuda.dll")
  set(ONNXRUNTIME_FILES "onnxruntime*.dll")
  set(ONNXRUNTIME_EXTENSIONS_LIB "tfmtok_c.lib")
  set(ONNXRUNTIME_EXTENSIONS_FILES "tfmtok_c.dll")
elseif(APPLE)
  set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "libonnxruntime_providers_cuda.dylib")
  set(ONNXRUNTIME_FILES "libonnxruntime*.dylib")
else()
  set(ONNXRUNTIME_LIB "libonnxruntime.so")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "libonnxruntime_providers_cuda.so")
  set(ONNXRUNTIME_FILES "libonnxruntime*.so*")
  set(ONNXRUNTIME_EXTENSIONS_LIB "tfmtok_c.so")
endif()

if(NO_TOKENIZEROOT)
  add_compile_definitions(NO_TOKENIZER=1)
  message("----------------Tokenizer Disabled------------------")
else()
  add_subdirectory("${CMAKE_SOURCE_DIR}/src/tokenizer")
  target_include_directories(onnxruntime-genai PRIVATE ${TOKENIZER_ROOT})
  target_include_directories(onnxruntime-genai-static PUBLIC ${TOKENIZER_ROOT})
  target_link_libraries(onnxruntime-genai PRIVATE tokenizer)
  target_link_libraries(onnxruntime-genai-static PUBLIC tokenizer)
endif()

if(ENABLE_TESTS)
  add_subdirectory("${CMAKE_SOURCE_DIR}/test")
  message("------------------Enabling tests------------------")
endif()

if(ENABLE_PYTHON)
  add_subdirectory("${CMAKE_SOURCE_DIR}/src/python")
  message("------------------Enabling Python Wheel------------------")
endif()

if(NOT EXISTS "${ORT_LIB_DIR}/${ONNXRUNTIME_LIB}")
  message(FATAL_ERROR "Expected the ONNX Runtime library to be found at ${ORT_LIB_DIR}/${ONNXRUNTIME_LIB}. Actual: Not found.")
endif()
if(NOT EXISTS "${ORT_HEADER_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "Expected the ONNX Runtime C API header to be found at \"${ORT_HEADER_DIR}/onnxruntime_c_api.h\". Actual: Not found.")
endif()
if(USE_CUDA AND NOT EXISTS "${ORT_LIB_DIR}/${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
  message(FATAL_ERROR "Expected the ONNX Runtime providers cuda library to be found at ${ORT_LIB_DIR}/${ONNXRUNTIME_PROVIDERS_CUDA_LIB}. Actual: Not found.")
endif()

file(GLOB onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_FILES}")
target_link_directories(onnxruntime-genai PRIVATE ${ORT_LIB_DIR})
target_link_libraries(onnxruntime-genai PRIVATE ${ONNXRUNTIME_LIB})

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  set_target_properties(onnxruntime-genai PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(onnxruntime-genai PRIVATE cublasLt cublas curand cufft cudart)
  #  onnxruntime-genai-static is statically linked under Windows
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(onnxruntime-genai-static PROPERTIES LINKER_LANGUAGE CUDA)
    target_link_libraries(onnxruntime-genai-static PRIVATE cublasLt cublas curand cufft cudart)
  endif()
endif()

# Visual C++ - /MP Multiple process build, /WX Warnings as errors
# Enable:
#   C5038: data member 'member1' will be initialized after data member 'member2'
# Disable:
#   C4100: 'this_': unreferenced formal parameter
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /w15038 /wd4100 /W4 /WX")
endif()

# Copy the onnxruntime binaries into the build folder so it's found on launch
file(GLOB onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_FILES}")
foreach(DLL_FILE ${onnxruntime_libs})
  add_custom_command(
    TARGET onnxruntime-genai POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL_FILE} $<TARGET_FILE_DIR:onnxruntime-genai>
  )
endforeach()

# Have visual studio put all files into one single folder vs the default split of header files into a separate folder
source_group(TREE ${GENERATORS_ROOT} FILES ${generator_srcs})

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.26)
include(FetchContent)
include(CMakeDependentOption)
project(Generators LANGUAGES C CXX)

# All Options should be defined in cmake/options.cmake This must be included before any other cmake file is included
include(cmake/options.cmake)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR  "GCC version must be greater than or equal to 11")
endif()

include(cmake/external/onnxruntime_external_deps.cmake)
# All Global variables, including GLOB, for the top level CMakeLists.txt should be defined here
include(cmake/global_variables.cmake)
# Checking if CUDA is supported
include(cmake/check_cuda.cmake)
# Checking if DML is supported
include(cmake/check_dml.cmake)

include(cmake/cxx_standard.cmake)

add_compile_definitions(BUILDING_ORT_GENAI_C)
if(MSVC)
  # set updated value for __cplusplus macro instead of 199711L
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
endif()

if(ENABLE_TESTS)
  # call enable_testing so we can add tests from subdirectories (e.g. test and src/java)
  # it applies recursively to all subdirectories
  enable_testing()
  if (TEST_PHI2)
    add_compile_definitions(TEST_PHI2=1)
  else()
    add_compile_definitions(TEST_PHI2=0)
  endif()
endif()

if(USE_CUDA)
  cmake_dependent_option(OCOS_USE_FLASH_ATTENTION "Build flash attention kernel for scaled dot product attention" ON "NOT WIN32" OFF)
  option(OCOS_USE_MEMORY_EFFICIENT_ATTENTION "Build memory efficient attention kernel for scaled dot product attention" ON)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.6)
    message(STATUS "Turn off flash attention and memory efficient attention since CUDA compiler version < 11.6")
    set(OCOS_USE_FLASH_ATTENTION OFF)
    set(OCOS_USE_MEMORY_EFFICIENT_ATTENTION OFF)
  endif()
  if(OCOS_USE_FLASH_ATTENTION)
    add_compile_definitions(OCOS_USE_FLASH_ATTENTION)
  endif()
  if (OCOS_USE_MEMORY_EFFICIENT_ATTENTION)
    add_compile_definitions(OCOS_USE_MEMORY_EFFICIENT_ATTENTION)
  endif()
  list(APPEND generator_srcs src/custom_ops/paged_attention_impl.cu src/custom_ops/genai_ops2.cc)
endif()

add_library(onnxruntime-genai SHARED ${generator_srcs})
add_library(onnxruntime-genai-static STATIC ${generator_srcs})
target_include_directories(onnxruntime-genai PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai-static PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/include)
target_include_directories(onnxruntime-genai PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/shared/api/)
target_include_directories(onnxruntime-genai-static PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/include)
target_include_directories(onnxruntime-genai-static PUBLIC ${onnxruntime_extensions_SOURCE_DIR}/shared/api/)
if(USE_CUDA)
target_include_directories(onnxruntime-genai PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/operators/cuda)
target_include_directories(onnxruntime-genai-static PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/operators/cuda)
endif()
target_link_libraries(onnxruntime-genai PRIVATE onnxruntime_extensions)
target_link_libraries(onnxruntime-genai-static PUBLIC onnxruntime_extensions)
target_link_directories(onnxruntime-genai PRIVATE ${ORT_LIB_DIR})
target_link_libraries(onnxruntime-genai PRIVATE ${ONNXRUNTIME_LIB})

set_target_properties(onnxruntime-genai PROPERTIES FOLDER "Sources")
set_target_properties(onnxruntime-genai-static PROPERTIES FOLDER "Sources")
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${generator_srcs})

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  set_target_properties(onnxruntime-genai PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(onnxruntime-genai PRIVATE cublasLt cublas curand cufft cudart)
  #  onnxruntime-genai-static is statically linked under Windows
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(onnxruntime-genai-static PROPERTIES LINKER_LANGUAGE CUDA)
    target_link_libraries(onnxruntime-genai-static PRIVATE cublasLt cublas curand cufft cudart)
  endif()
endif()

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(onnxruntime-genai PRIVATE "/CETCOMPAT")
  target_compile_options(onnxruntime-genai PRIVATE "/sdl")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(onnxruntime-genai-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


if(USE_DML)
  list(APPEND onnxruntime_libs "${ORT_LIB_DIR}/DirectML.dll")
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${WIL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>/directx)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai-static PUBLIC $<TARGET_PROPERTY:${WIL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai-static PUBLIC $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>/directx)
  target_include_directories(onnxruntime-genai-static PUBLIC $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(onnxruntime-genai PRIVATE d3d12.lib dxcore.lib dxguid.lib dxgi.lib)
  target_link_libraries(onnxruntime-genai-static PUBLIC d3d12.lib dxcore.lib dxguid.lib dxgi.lib)

  get_filename_component(PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps ABSOLUTE)
  set(DXC_PACKAGE_DIR ${PACKAGES_DIR}/Microsoft.Direct3D.DXC.1.7.2308.12)
  set(NUGET_CONFIG ${PROJECT_SOURCE_DIR}/nuget.config)
  set(PACKAGES_CONFIG ${PROJECT_SOURCE_DIR}/packages.config)

  add_custom_command(
    OUTPUT
    ${DXC_PACKAGE_DIR}/build/native/bin/x64/dxc.exe
    DEPENDS
    ${PACKAGES_CONFIG}
    ${NUGET_CONFIG}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nuget/src/nuget restore ${PACKAGES_CONFIG} -PackagesDirectory ${PACKAGES_DIR} -ConfigFile ${NUGET_CONFIG}
    VERBATIM
  )

  add_custom_target(
    RESTORE_PACKAGES ALL
    DEPENDS
    ${DXC_PACKAGE_DIR}/build/native/bin/x64/dxc.exe
  )

  add_dependencies(RESTORE_PACKAGES nuget)
  add_dependencies(onnxruntime-genai RESTORE_PACKAGES)
  add_dependencies(onnxruntime-genai-static RESTORE_PACKAGES)
endif()

if(ENABLE_TESTS)
  message("------------------Enabling tests------------------")
  add_subdirectory("${REPO_ROOT}/test")
endif()

if(ENABLE_PYTHON)
  message("------------------Enabling Python Wheel------------------")
  add_subdirectory("${SRC_ROOT}/python")
endif()

if (ENABLE_JAVA)
  message("------------------Enabling Java Jar------------------")
  add_subdirectory("${SRC_ROOT}/java")
endif()


if(ENABLE_MODEL_BENCHMARK)
  message("------------------Enabling model benchmark------------------")
  add_subdirectory("${REPO_ROOT}/benchmark/c")
endif()

# Copy the onnxruntime binaries into the build folder so it's found on launch
foreach(DLL_FILE ${onnxruntime_libs})
  add_custom_command(
    TARGET onnxruntime-genai POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL_FILE} $<TARGET_FILE_DIR:onnxruntime-genai>
  )
endforeach()

# Have visual studio put all files into one single folder vs the default split of header files into a separate folder
source_group(TREE ${GENERATORS_ROOT} FILES ${generator_srcs})

include(cmake/package.cmake)

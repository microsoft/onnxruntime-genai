# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # For xcframework support
  cmake_minimum_required(VERSION 3.28)
else()
  cmake_minimum_required(VERSION 3.26)
endif()
include(FetchContent)
include(CMakeDependentOption)
project(Generators LANGUAGES C CXX)

# All Options should be defined in cmake/options.cmake This must be included before any other cmake file is included
include(cmake/options.cmake)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR  "GCC version must be greater than or equal to 11")
endif()

# Avoid warning of Calling FetchContent_Populate(Lib) is deprecated temporarily
# TODO: find a better way to handle the header-only 3rd party deps
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
  cmake_policy(SET CMP0169 OLD)
endif()

if(MSVC)
  # DLL initialization errors due to old conda msvcp140.dll dll are a result of the new MSVC compiler
  # See https://developercommunity.visualstudio.com/t/Access-violation-with-std::mutex::lock-a/10664660#T-N10668856
  # Remove this definition once the conda msvcp140.dll dll is updated.
  add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)

  # Add compiler options for security standards
  # Note: Must use separate generator expressions — do NOT use "C,CXX" in COMPILE_LANGUAGE.

  # Explicitly enable (/GS) stack protection to pass BinSkim checks
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/GS>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GS>")

  # Enable Control Flow Guard on compiler and linker to pass BinSkim checks
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/guard:cf>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/guard:cf>")
  add_link_options(/guard:cf /DYNAMICBASE)

  # Enable CET shadow stack unless building for ARM64 (not supported there)
  message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    message(STATUS "CET shadow stack enabled (/CETCOMPAT)")
    add_link_options(/CETCOMPAT)
  else()
    message(STATUS "CET shadow stack skipped for ARM64 build")
  endif()

  # Enable Spectre mitigations for C and C++ compilations only (avoid CUDA nvcc that is not supported there)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C>:/Qspectre>"
    "$<$<COMPILE_LANGUAGE:CXX>:/Qspectre>"
  )

  # Use updated value for __cplusplus macro (e.g. 201703L for C++17) instead of default 199711L
  # Required by some libraries/tools that check __cplusplus for feature support
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)

  add_compile_options(
    # Suppress warning C5038: data member will be initialized after another
    # Common when member initializer list is written out-of-order vs. declaration
    "$<$<COMPILE_LANGUAGE:C>:/w15038>"
    "$<$<COMPILE_LANGUAGE:CXX>:/w15038>"

    # Suppress warning C4100: unreferenced formal parameter
    # Often appears in template or virtual function overrides where a param is unused
    "$<$<COMPILE_LANGUAGE:C>:/wd4100>"
    "$<$<COMPILE_LANGUAGE:CXX>:/wd4100>"

    # Enable warning level 4 (more aggressive than default /W3)
    # Captures more potential bugs or code smells
    "$<$<COMPILE_LANGUAGE:C>:/W4>"
    "$<$<COMPILE_LANGUAGE:CXX>:/W4>"

    # Treat all warnings as errors (fail build on any warning)
    # Combine carefully with /W4 — suppress any known false positives
    "$<$<COMPILE_LANGUAGE:C>:/WX>"
    "$<$<COMPILE_LANGUAGE:CXX>:/WX>"
  )
endif()

include(cmake/ortlib.cmake)
include(cmake/external/onnxruntime_external_deps.cmake)
# All Global variables, including GLOB, for the top level CMakeLists.txt should be defined here
include(cmake/global_variables.cmake)
# Checking if CUDA is supported
include(cmake/check_cuda.cmake)
# Checking if ROCm is supported
include(cmake/check_rocm.cmake)
# Checking if DML is supported
include(cmake/check_dml.cmake)

include(cmake/cxx_standard.cmake)

add_compile_definitions(BUILDING_ORT_GENAI_C)

add_compile_definitions(USE_GUIDANCE=$<BOOL:${USE_GUIDANCE}>)

# Suggested by https://gitlab.kitware.com/cmake/cmake/-/issues/20132
# MacCatalyst is not well supported in CMake
# The error that can emerge without this flag can look like:
# "clang : error : overriding '-mmacosx-version-min=11.0' option with '-target x86_64-apple-ios14.0-macabi' [-Werror,-Woverriding-t-option]"
if (PLATFORM_NAME STREQUAL "macabi")
  add_compile_options(-Wno-overriding-t-option)
  add_link_options(-Wno-overriding-t-option)
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

if(ENABLE_TRACING)
  message(STATUS "Tracing is enabled.")
  add_compile_definitions(ORTGENAI_ENABLE_TRACING)
endif()

find_package(Threads REQUIRED)

if(WIN32)
  add_library(onnxruntime-genai SHARED ${generator_srcs} "${GENERATORS_ROOT}/dll/onnxruntime-genai.rc")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_INFO=\"${VERSION_INFO}\")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MAJOR=${VERSION_MAJOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MINOR=${VERSION_MINOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_PATCH=${VERSION_PATCH})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_SUFFIX=${VERSION_SUFFIX})
  target_compile_definitions(onnxruntime-genai PRIVATE FILE_NAME=\"onnxruntime-genai.dll\")
else()
  add_library(onnxruntime-genai SHARED ${generator_srcs})
endif()

target_include_directories(onnxruntime-genai PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/shared/api)
target_link_libraries(onnxruntime-genai PRIVATE onnxruntime_extensions)
target_link_directories(onnxruntime-genai PRIVATE ${ORT_LIB_DIR})
target_link_libraries(onnxruntime-genai PRIVATE Threads::Threads)

# The genai library itself is always embedded in the shared library
list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai>")

# we keep the shared libraries disconnected on Android as they will come from separate AARs and we don't want to force
# the ORT version to match in both.
if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND (NOT BUILD_APPLE_FRAMEWORK) AND (NOT MAC_CATALYST)))
  add_compile_definitions(_ORT_GENAI_USE_DLOPEN)
else()
  target_link_libraries(onnxruntime-genai PRIVATE ${ONNXRUNTIME_LIB})
endif()

if(APPLE)
target_link_libraries(onnxruntime-genai PRIVATE "-framework Foundation" "-framework CoreML")
endif()


# Build all source files using CUDA as a separate shared library we dynamically load at runtime
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  # Suppress nvcc warning 1650 (result of call is not used) and deprecated GPU targets warning for cuda 12.8 or later.
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress;1650;-Wno-deprecated-gpu-targets>")
  
  add_library(onnxruntime-genai-cuda SHARED ${generator_cudalib_srcs})
  target_include_directories(onnxruntime-genai-cuda PRIVATE ${ORT_HEADER_DIR})
  target_include_directories(onnxruntime-genai-cuda PRIVATE ${GENERATORS_ROOT})
  target_link_libraries(onnxruntime-genai-cuda PRIVATE cublasLt cublas curand cufft cudart)
  set_target_properties(onnxruntime-genai-cuda PROPERTIES LINKER_LANGUAGE CUDA)
  add_dependencies(onnxruntime-genai onnxruntime-genai-cuda)
  source_group(TREE ${GENERATORS_ROOT}/cuda FILES ${generator_cudalib_srcs})
  list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai-cuda>")
  if(APPLE)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${GENERATORS_ROOT}/cuda/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${GENERATORS_ROOT}/cuda/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${GENERATORS_ROOT}/cuda/symbols.def")
  else()
    message(FATAL_ERROR "${target} unknown platform, need to specify shared library exports for it")
  endif()
endif()


if(USE_GUIDANCE)
  target_include_directories(onnxruntime-genai PUBLIC ${llguidance_SOURCE_DIR}/parser/)
  target_link_libraries(onnxruntime-genai PRIVATE llguidance)
  if (WIN32)
    # bcrypt is needed for the rust std lib
    target_link_libraries(onnxruntime-genai PRIVATE bcrypt)
  endif()
endif()

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(onnxruntime-genai PRIVATE "/CETCOMPAT")
  target_compile_options(onnxruntime-genai PRIVATE "/sdl")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(onnxruntime-genai PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(onnxruntime-genai PRIVATE dl)  # For dlopen & co
endif()

if(USE_DML)
  list(APPEND ortgenai_embed_libs "${D3D12_LIB_DIR}/D3D12Core.dll")
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${WIL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>/directx)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai PRIVATE ${DML_HEADER_DIR})
  target_include_directories(onnxruntime-genai PRIVATE ${D3D12_HEADER_DIR})
  target_link_directories(onnxruntime-genai PRIVATE ${DML_LIB_DIR})
  target_link_directories(onnxruntime-genai PRIVATE ${D3D12_LIB_DIR})
  target_link_libraries(onnxruntime-genai PRIVATE d3d12.lib dxcore.lib dxguid.lib dxgi.lib)

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
endif()

if(ANDROID)
  # strip the binary if it's not a build with debug info
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_RELEASE -s)
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_MINSIZEREL -s)
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

# Have visual studio put all files into one single folder vs the default split of header files into a separate folder
source_group(TREE ${GENERATORS_ROOT} FILES ${generator_srcs})

include(cmake/package.cmake)

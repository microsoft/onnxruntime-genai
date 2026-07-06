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
  # Add compiler options for security standards
  # Note: Must use separate generator expressions — do NOT use "C,CXX" in COMPILE_LANGUAGE.

  # Explicitly enable (/GS) stack protection to pass BinSkim checks
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/GS>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GS>")

  # Enable Control Flow Guard on compiler and linker to pass BinSkim checks
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/guard:cf>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/guard:cf>")
  add_link_options(/guard:cf /DYNAMICBASE)

  # Determine the target platform (from MSBuild generator)
  if (CMAKE_VS_PLATFORM_NAME)
    set(onnxruntime_target_platform ${CMAKE_VS_PLATFORM_NAME})
  else()
    set(onnxruntime_target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()

  # Enable CET shadow stack unless building for ARM64 (not supported there)
  if (onnxruntime_target_platform STREQUAL "x64" OR
      onnxruntime_target_platform STREQUAL "Win32" OR
      onnxruntime_target_platform STREQUAL "x86")
    add_link_options("/CETCOMPAT")
    message(STATUS "CET shadow stack enabled (/CETCOMPAT).")
  else()
    message(STATUS "CET shadow stack skipped.")
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

    # Suppress warning C4819: file contains character that cannot be represented in current code page
    "$<$<COMPILE_LANGUAGE:C>:/wd4819>"
    "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>"

    # Suppress warning C4996: use of deprecated function or variable
    "$<$<COMPILE_LANGUAGE:C>:/wd4996>"
    "$<$<COMPILE_LANGUAGE:CXX>:/wd4996>"

    # Enable warning level 4 (more aggressive than default /W3)
    # Captures more potential bugs or code smells
    "$<$<COMPILE_LANGUAGE:C>:/W4>"
    "$<$<COMPILE_LANGUAGE:CXX>:/W4>"

    # Treat all warnings as errors (fail build on any warning)
    # Combine carefully with /W4 — suppress any known false positives
    "$<$<COMPILE_LANGUAGE:C>:/WX>"
    "$<$<COMPILE_LANGUAGE:CXX>:/WX>"
  )

  # MSVC 19.50+ (VS 2026) introduced C4875: non-string literal argument to [[gsl::suppress]] is
  # deprecated. The GSL version used by onnxruntime_extensions (and other deps) hasn't adopted
  # string-literal syntax yet. Suppress globally so third-party headers build cleanly.
  # Also silence the STL1011 static assertion for <experimental/coroutine> being deprecated
  # (ocos_operators in onnxruntime_extensions still uses it; the fix is upstream).
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.50")
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:C>:/wd4875>"
      "$<$<COMPILE_LANGUAGE:CXX>:/wd4875>"
    )
    add_compile_definitions(_SILENCE_EXPERIMENTAL_COROUTINE_DEPRECATION_WARNINGS)
  endif()
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

  if (TEST_QWEN_2_5)
    add_compile_definitions(TEST_QWEN_2_5=1)
  else()
    add_compile_definitions(TEST_QWEN_2_5=0)
  endif()

  if (USE_WEBGPU)
    add_compile_definitions(USE_WEBGPU=1)
  else()
    add_compile_definitions(USE_WEBGPU=0)
  endif()

endif()

if(ENABLE_TRACING)
  message(STATUS "Tracing is enabled.")
  add_compile_definitions(ORTGENAI_ENABLE_TRACING)
endif()

find_package(Threads REQUIRED)

# genai is split into an OBJECT library (all sources + their compile/link usage requirements) and
# the SHARED library that ships it. White-box test binaries (e.g. reinit_tests) link the object
# library directly to reach internal symbols; the shipped DLL and every other consumer keep linking
# the SHARED onnxruntime-genai target. Usage requirements are PUBLIC on the object library so both
# the shared library and such tests inherit them. The shared library links the object library
# PRIVATE, so DLL consumers (python / java / benchmark / unit_tests) do not inherit genai's private
# dependencies -- they just link the DLL as before.
add_library(onnxruntime-genai-obj OBJECT ${generator_srcs})
target_include_directories(onnxruntime-genai-obj PUBLIC ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai-obj PUBLIC ${onnxruntime_extensions_SOURCE_DIR}/shared/api)
target_link_libraries(onnxruntime-genai-obj PUBLIC onnxruntime_extensions)
target_link_directories(onnxruntime-genai-obj PUBLIC ${ORT_LIB_DIR})
target_link_libraries(onnxruntime-genai-obj PUBLIC Threads::Threads)

if(WIN32)
  add_library(onnxruntime-genai SHARED "${GENERATORS_ROOT}/dll/onnxruntime-genai.rc")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_INFO=\"${VERSION_INFO}\")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MAJOR=${VERSION_MAJOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MINOR=${VERSION_MINOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_PATCH=${VERSION_PATCH})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_SUFFIX=${VERSION_SUFFIX})
  target_compile_definitions(onnxruntime-genai PRIVATE FILE_NAME=\"onnxruntime-genai.dll\")
else()
  add_library(onnxruntime-genai SHARED)
endif()
# Bring in the object library's compiled objects and its (PUBLIC) usage requirements. PRIVATE so
# the requirements are not re-exported to consumers of the shared library.
target_link_libraries(onnxruntime-genai PRIVATE onnxruntime-genai-obj)

# Group the genai library targets under one Visual Studio solution folder so the object library and
# the shared library it produces appear together rather than as loose entries at the solution root.
# Cosmetic only (IDE generators); no effect on the build.
set_target_properties(onnxruntime-genai-obj onnxruntime-genai PROPERTIES FOLDER "GenAI")

# The genai library itself is always embedded in the shared library
list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai>")

# we keep the shared libraries disconnected on Android as they will come from separate AARs and we don't want to force
# the ORT version to match in both.
if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND (NOT BUILD_APPLE_FRAMEWORK) AND (NOT MAC_CATALYST)))
  add_compile_definitions(_ORT_GENAI_USE_DLOPEN)
else()
  target_link_libraries(onnxruntime-genai-obj PUBLIC ${ONNXRUNTIME_LIB})
  if(USE_WINML)
    target_link_options(onnxruntime-genai PRIVATE "/DELAYLOAD:${ONNXRUNTIME_LIB}")
  endif()
endif()

if(APPLE)
target_link_libraries(onnxruntime-genai-obj PUBLIC "-framework Foundation" "-framework CoreML")
endif()


# Build all source files using CUDA as a separate shared library we dynamically load at runtime
if((USE_CUDA OR USE_TRT_RTX) AND CMAKE_CUDA_COMPILER)
  # Suppress nvcc warnings:
  # 1650 = "result of call is not used"
  # 221  = "floating-point value does not fit in required floating-point type"
  # Also suppress deprecated GPU targets warnings for CUDA 12.8+
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=1650>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=221>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/wd4996>
  )

  # nvcc compiles CUDA code with cl.exe, so we need to pass the /Zc:preprocessor option to avoid compilation errors
  if(WIN32 AND MSVC AND MSVC_VERSION GREATER_EQUAL 1925)
    add_compile_options(
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/Zc:preprocessor>
    )
  endif()
  
  if(WIN32)
    add_library(onnxruntime-genai-cuda SHARED ${generator_cudalib_srcs} "${GENERATORS_ROOT}/dll/onnxruntime-genai.rc")
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_INFO=\"${VERSION_INFO}\")
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_MAJOR=${VERSION_MAJOR})
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_MINOR=${VERSION_MINOR})
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_PATCH=${VERSION_PATCH})
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE VERSION_SUFFIX=${VERSION_SUFFIX})
    target_compile_definitions(onnxruntime-genai-cuda PRIVATE FILE_NAME=\"onnxruntime-genai-cuda.dll\")
  else()
    add_library(onnxruntime-genai-cuda SHARED ${generator_cudalib_srcs})
  endif()
  target_include_directories(onnxruntime-genai-cuda PRIVATE ${ORT_HEADER_DIR})
  target_include_directories(onnxruntime-genai-cuda PRIVATE ${GENERATORS_ROOT})
  target_link_libraries(onnxruntime-genai-cuda PRIVATE cublasLt cublas curand cufft cudart)
  set_target_properties(onnxruntime-genai-cuda PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(onnxruntime-genai-cuda PROPERTIES FOLDER "GenAI")
  add_dependencies(onnxruntime-genai onnxruntime-genai-cuda)
  source_group(TREE ${GENERATORS_ROOT}/cuda FILES ${generator_cudalib_srcs})
  list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai-cuda>")
  if(APPLE)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${GENERATORS_ROOT}/cuda/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${GENERATORS_ROOT}/cuda/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-DEF:\"${GENERATORS_ROOT}/cuda/symbols.def\"")
  else()
    message(FATAL_ERROR "${target} unknown platform, need to specify shared library exports for it")
  endif()
endif()


if(USE_GUIDANCE)
  target_include_directories(onnxruntime-genai-obj PUBLIC ${llguidance_SOURCE_DIR}/parser/)
  target_link_libraries(onnxruntime-genai-obj PUBLIC llguidance)
  if (WIN32)
    # bcrypt is needed for the rust std lib
    target_link_libraries(onnxruntime-genai-obj PUBLIC bcrypt)
  endif()
  if(MSVC)
    # The Rust llguidance static library is always compiled against the release MSVC CRT
    # (Rust has no debug CRT concept). The .lib embeds /DEFAULTLIB directives for the release
    # CRT and /NODEFAULTLIB directives that suppress the debug CRT (msvcrtd, ucrtd, vcruntimed).
    # In Debug builds, C++ code (e.g. onnxruntime-extensions) is compiled with /MDd and references
    # debug-only CRT functions like _CrtDbgReport (in ucrtd.lib). Because the Rust .lib suppresses
    # ucrtd.lib via its embedded /NODEFAULTLIB, _CrtDbgReport becomes unresolved.
    #
    # Fix: explicitly add the debug CRT import libraries in Debug builds. Explicitly specified
    # libraries are not affected by /NODEFAULTLIB directives (those only suppress /DEFAULTLIB
    # auto-linking). Also suppress the conflicting release CRT to avoid LNK4098 warnings.
    # PUBLIC/INTERFACE so any binary linking the object library (the shared lib and white-box tests)
    # gets the same fixups.
    target_link_libraries(onnxruntime-genai-obj PUBLIC
      $<$<CONFIG:Debug>:msvcrtd.lib>
      $<$<CONFIG:Debug>:ucrtd.lib>
      $<$<CONFIG:Debug>:vcruntimed.lib>
    )
    target_link_options(onnxruntime-genai-obj INTERFACE
      $<$<CONFIG:Debug>:/NODEFAULTLIB:msvcrt.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:ucrt.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:vcruntime.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:libcmt.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:libucrt.lib>
      $<$<CONFIG:Debug>:/NODEFAULTLIB:libvcruntime.lib>
    )
  endif()
endif()

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_compile_options(onnxruntime-genai-obj PRIVATE "/sdl")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(onnxruntime-genai-obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(onnxruntime-genai-obj PUBLIC dl)  # For dlopen & co
endif()

if(USE_DML)
  list(APPEND ortgenai_embed_libs "${D3D12_LIB_DIR}/D3D12Core.dll")
  target_include_directories(onnxruntime-genai-obj PRIVATE $<TARGET_PROPERTY:${WIL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai-obj PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>/directx)
  target_include_directories(onnxruntime-genai-obj PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai-obj PRIVATE ${DML_HEADER_DIR})
  target_include_directories(onnxruntime-genai-obj PRIVATE ${D3D12_HEADER_DIR})
  target_link_directories(onnxruntime-genai-obj PUBLIC ${DML_LIB_DIR})
  target_link_directories(onnxruntime-genai-obj PUBLIC ${D3D12_LIB_DIR})
  target_link_libraries(onnxruntime-genai-obj PUBLIC d3d12.lib dxcore.lib dxguid.lib dxgi.lib)

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
  add_dependencies(onnxruntime-genai-obj RESTORE_PACKAGES)
endif()

if(ANDROID)
  # strip the binary if it's not a build with debug info
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_RELEASE -s)
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_MINSIZEREL -s)

  # Build shared libraries with support for 16 KB page size on Android
  # https://source.android.com/docs/core/architecture/16kb-page-size/16kb#build-lib-16kb-alignment
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-z,max-page-size=16384")
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

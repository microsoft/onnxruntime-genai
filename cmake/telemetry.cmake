# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# CMake module for 1DS C++ SDK (MAT) telemetry integration.
#
# The 1DS SDK is obtained one of two ways, in priority order:
#
#   1. A caller-supplied MSTelemetry::mat CONFIG package, when already available in the
#      configured CMake environment.
#
#   2. FetchContent source build (default): when the package is not available, the
#      SDK source (pinned in cmake/deps.txt) is downloaded and built locally.
#      This matches onnxruntime-genai's dependency model (ORT_HOME + FetchContent) and works
#      on supported platforms since GenAI uses 1DS everywhere.
#
# Either way this module defines the INTERFACE target `onnxruntime-genai-telemetry`, which
# the main library links, so the rest of the build is agnostic to how the SDK was obtained.

if(NOT ENABLE_TELEMETRY)
  return()
endif()

if(ANDROID AND NOT ENABLE_JAVA)
  message(WARNING
    "Android telemetry requires the host app to initialize the 1DS Java HttpClient before using GenAI. "
    "Build with ENABLE_JAVA=ON to package the automatic AAR initializer.")
endif()

# ---------------------------------------------------------------------------
# Path 1: caller-supplied package
# ---------------------------------------------------------------------------
if(NOT ANDROID)
  find_package(MSTelemetry CONFIG QUIET)
endif()
if(NOT ANDROID AND TARGET MSTelemetry::mat)
  message(STATUS "Telemetry: using the caller-supplied MSTelemetry::mat package.")

  add_library(onnxruntime-genai-telemetry INTERFACE)
  target_link_libraries(onnxruntime-genai-telemetry INTERFACE MSTelemetry::mat)

  # A static package lets the linker dead-strip unused SDK code. A shared package ships the full
  # MAT library and its dependencies, which GenAI does not package.
  get_target_property(_mat_type MSTelemetry::mat TYPE)
  if(_mat_type STREQUAL "SHARED_LIBRARY")
    message(FATAL_ERROR
      "Telemetry requires a static caller-supplied MSTelemetry::mat target because GenAI packages "
      "do not bundle the MAT shared library or its runtime dependencies.")
  endif()

  return()
endif()

# ---------------------------------------------------------------------------
# Path 2: FetchContent source build
# ---------------------------------------------------------------------------
message(STATUS "Telemetry: MSTelemetry::mat not found; building the 1DS SDK from source via FetchContent.")

include(FetchContent)

# Use the SDK's canonical build options. GenAI consumes only the C++ library and supplies all
# dependencies needed by its packaged static/shared target.
set(MATSDK_BUILD_HEADERS ON CACHE BOOL "Build 1DS SDK headers" FORCE)
set(MATSDK_BUILD_LIBRARY ON CACHE BOOL "Build 1DS SDK library" FORCE)
set(MATSDK_BUILD_TEST_TOOL OFF CACHE BOOL "Disable 1DS SDK test tool" FORCE)
set(MATSDK_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable 1DS SDK unit tests" FORCE)
set(MATSDK_BUILD_FUNC_TESTS OFF CACHE BOOL "Disable 1DS SDK functional tests" FORCE)
set(MATSDK_BUILD_PRIVACYGUARD OFF CACHE BOOL "Disable 1DS privacy guard module" FORCE)
set(MATSDK_BUILD_CDS OFF CACHE BOOL "Disable 1DS CDS module" FORCE)
set(MATSDK_BUILD_LIVEEVENTINSPECTOR OFF CACHE BOOL "Disable 1DS live event inspector" FORCE)
set(MATSDK_BUILD_SIGNALS OFF CACHE BOOL "Disable 1DS signals module" FORCE)
set(MATSDK_BUILD_SANITIZER OFF CACHE BOOL "Disable 1DS sanitizer module" FORCE)
set(MATSDK_BUILD_AZMON OFF CACHE BOOL "Disable 1DS Azure Monitor module" FORCE)
set(MATSDK_BUILD_OBJC_WRAPPER OFF CACHE BOOL "Disable 1DS ObjC wrapper" FORCE)
set(MATSDK_BUILD_SWIFT_WRAPPER OFF CACHE BOOL "Disable 1DS Swift wrapper" FORCE)
set(MATSDK_BUILD_JNI_WRAPPER OFF CACHE BOOL "Disable 1DS JNI wrapper" FORCE)
set(MATSDK_BUILD_PACKAGE OFF CACHE BOOL "Disable 1DS package generation" FORCE)
set(MATSDK_BUILD_APPLE_HTTP ${APPLE} CACHE BOOL "Build the 1DS Apple HTTP client" FORCE)
set(MATSDK_ANDROID_HTTP_CLIENT JAVA CACHE STRING "Use the 1DS Java HTTP bridge on Android" FORCE)
set(MATSDK_CURL_TLS_BACKEND MBEDTLS CACHE STRING "Use mbedTLS for 1DS curl" FORCE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(MATSDK_CURL_PROVIDER FETCH CACHE STRING "Build the SDK's pinned curl with mbedTLS" FORCE)
else()
  set(MATSDK_CURL_PROVIDER SYSTEM CACHE STRING "Use the platform HTTP transport" FORCE)
endif()
if(APPLE)
  set(MATSDK_SQLITE_PROVIDER SYSTEM CACHE STRING "Use Apple's system SQLite" FORCE)
  set(MATSDK_ZLIB_PROVIDER SYSTEM CACHE STRING "Use Apple's system libz" FORCE)
else()
  set(MATSDK_SQLITE_PROVIDER MINIMAL CACHE STRING "Build the SDK's minimal private SQLite" FORCE)
  set(MATSDK_ZLIB_PROVIDER VENDORED CACHE STRING "Build the SDK's private zlib" FORCE)
endif()

# BUILD_SHARED_LIBS is a global that onnxruntime-genai's own targets read after this module, and the SDK
# selects mat's library type from it. Save it and restore it after the SDK is configured. Desktop and
# Apple builds use a dead-strippable static library. Android Java/AAR builds use libmat.so so the
# SDK's Java HTTP bridge and GenAI resolve against the same process-wide SDK state; native-only
# Android builds stay static and safely disable telemetry when no Java context is available.
set(_ortgenai_build_shared_libs_saved "${BUILD_SHARED_LIBS}")
if(ANDROID AND ENABLE_JAVA)
  set(BUILD_SHARED_LIBS ON CACHE BOOL "Build the Android 1DS SDK as a shared library" FORCE)
else()
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build the 1DS SDK as a static library" FORCE)
endif()

FetchContent_Declare(
  cpp_client_telemetry
  URL ${DEP_URL_cpp_client_telemetry}
  URL_HASH SHA1=${DEP_SHA1_cpp_client_telemetry}
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(cpp_client_telemetry)
target_compile_definitions(mat PRIVATE MATSDK_DISABLE_LOGGING)
if(ANDROID)
  target_compile_definitions(mat PRIVATE ANDROID_SUPPRESS_LOGCAT)
endif()

if(ANDROID)
  set(ORTGENAI_TELEMETRY_ANDROID_JAVA_SOURCE_DIR
    "${cpp_client_telemetry_SOURCE_DIR}/lib/android_build/maesdk/src/main/java")
  set(ORTGENAI_TELEMETRY_LICENSE_FILE "${cpp_client_telemetry_SOURCE_DIR}/LICENSE")
  target_sources(mat PRIVATE
    "${PROJECT_SOURCE_DIR}/cmake/telemetry/android_telemetry_bridge.cpp")
endif()
foreach(_ortgenai_1ds_cache_var
    MATSDK_BUILD_HEADERS
    MATSDK_BUILD_LIBRARY
    MATSDK_BUILD_TEST_TOOL
    MATSDK_BUILD_UNIT_TESTS
    MATSDK_BUILD_FUNC_TESTS
    MATSDK_BUILD_PRIVACYGUARD
    MATSDK_BUILD_CDS
    MATSDK_BUILD_LIVEEVENTINSPECTOR
    MATSDK_BUILD_SIGNALS
    MATSDK_BUILD_SANITIZER
    MATSDK_BUILD_AZMON
    MATSDK_BUILD_OBJC_WRAPPER
    MATSDK_BUILD_SWIFT_WRAPPER
    MATSDK_BUILD_JNI_WRAPPER
    MATSDK_BUILD_PACKAGE
    MATSDK_BUILD_APPLE_HTTP
    MATSDK_ANDROID_HTTP_CLIENT
    MATSDK_CURL_PROVIDER
    MATSDK_CURL_TLS_BACKEND
    MATSDK_SQLITE_PROVIDER
    MATSDK_ZLIB_PROVIDER)
  unset(${_ortgenai_1ds_cache_var} CACHE)
endforeach()

if(NOT TARGET mat)
  message(FATAL_ERROR "Telemetry: the 1DS SDK 'mat' target was not created by FetchContent.")
endif()

# Vendored 1DS dependencies emit unavoidable narrowing warnings under the Apple warning policy.
# Suppress them only for third-party targets so GenAI sources retain the warning.
if(APPLE)
  foreach(_ortgenai_apple_telemetry_target sqlite3_bundled zlib_bundled mat)
    if(TARGET ${_ortgenai_apple_telemetry_target})
      target_compile_options(${_ortgenai_apple_telemetry_target} PRIVATE -Wno-shorten-64-to-32)
    endif()
  endforeach()
endif()
if(TARGET sqlite3_bundled)
  target_compile_options(sqlite3_bundled PRIVATE
    $<$<COMPILE_LANG_AND_ID:C,GNU>:-Wno-error=stringop-overread>)
endif()

# Guard SDK warnings that the consumer treats as errors without weakening warnings for GenAI targets.
if(MSVC)
  get_target_property(_ortgenai_mat_opts mat COMPILE_OPTIONS)
  if(_ortgenai_mat_opts)
    list(REMOVE_ITEM _ortgenai_mat_opts
      "$<$<COMPILE_LANGUAGE:C>:/w15038>"
      "$<$<COMPILE_LANGUAGE:CXX>:/w15038>")
    set_target_properties(mat PROPERTIES COMPILE_OPTIONS "${_ortgenai_mat_opts}")
  endif()
  target_compile_options(mat PRIVATE /EHsc /wd5038)
else()
  # Guard the SDK's bundled nlohmann/json.hpp use of infinity() against any -ffast-math /
  # -ffinite-math-only in the inherited flags.
  target_compile_options(mat PRIVATE
    -fno-finite-math-only
    -Wno-unused-const-variable
    $<$<CXX_COMPILER_ID:GNU>:-Wno-reorder>
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-reorder-ctor>)
endif()

# Restore the caller's BUILD_SHARED_LIBS now that the SDK targets are configured.
set(BUILD_SHARED_LIBS "${_ortgenai_build_shared_libs_saved}" CACHE BOOL "Restored after building 1DS SDK" FORCE)

add_library(onnxruntime-genai-telemetry INTERFACE)
target_link_libraries(onnxruntime-genai-telemetry INTERFACE mat)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # curl is an implementation detail absorbed into the GenAI shared library, not a public GenAI API.
  target_link_options(onnxruntime-genai-telemetry INTERFACE "LINKER:--exclude-libs,libcurl.a")
endif()

# `mat` already exports lib/include/public as a PUBLIC build-interface include, which covers the
# LogManager / LogManagerProvider headers genai uses; also add include/mat so any transitive SDK headers
# resolve, matching ONNX Runtime's telemetry include set. These are build-tree paths, which is fine:
# onnxruntime-genai links this target PRIVATE and never installs/exports it.
target_include_directories(onnxruntime-genai-telemetry INTERFACE
  ${cpp_client_telemetry_SOURCE_DIR}/lib/include/public
  ${cpp_client_telemetry_SOURCE_DIR}/lib/include/mat)

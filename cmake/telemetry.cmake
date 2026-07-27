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
#      SDK source (pinned in cmake/deps.txt) is downloaded, patched, and built locally.
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

# The FetchContent fallback relies on genai's patch for a correct source-root include path and portable
# static packaging. Require the patch tool up front with a clear message rather than failing obscurely.
find_program(ORTGENAI_PATCH_EXECUTABLE NAMES patch)
if(NOT ORTGENAI_PATCH_EXECUTABLE AND WIN32)
  find_program(ORTGENAI_GIT_EXECUTABLE NAMES git)
  if(ORTGENAI_GIT_EXECUTABLE)
    get_filename_component(_ortgenai_git_dir "${ORTGENAI_GIT_EXECUTABLE}" DIRECTORY)
    foreach(_ortgenai_patch_candidate
        "${_ortgenai_git_dir}/../usr/bin/patch.exe"
        "${_ortgenai_git_dir}/patch.exe")
      if(EXISTS "${_ortgenai_patch_candidate}")
        set(ORTGENAI_PATCH_EXECUTABLE "${_ortgenai_patch_candidate}")
        break()
      endif()
    endforeach()
  endif()
endif()
if(NOT ORTGENAI_PATCH_EXECUTABLE)
  message(FATAL_ERROR
    "ENABLE_TELEMETRY requires the 'patch' tool to build the 1DS SDK from source when "
    "MSTelemetry::mat is not already available. On Windows, patch ships with Git in <Git>/usr/bin.")
endif()

# The 1DS SDK reads these generic option() names from its own CMakeLists. Disable its tests and the
# optional modules whose source may be absent from the release archive; genai uses the C++ API directly.
set(BUILD_UNIT_TESTS OFF CACHE BOOL "Disable 1DS SDK unit tests" FORCE)
set(BUILD_FUNC_TESTS OFF CACHE BOOL "Disable 1DS SDK functional tests" FORCE)
set(BUILD_PRIVACYGUARD OFF CACHE BOOL "Disable 1DS privacy guard module" FORCE)
set(BUILD_SANITIZER OFF CACHE BOOL "Disable 1DS sanitizer module" FORCE)
set(BUILD_OBJC_WRAPPER OFF CACHE BOOL "Disable 1DS ObjC wrapper" FORCE)
set(BUILD_SWIFT_WRAPPER OFF CACHE BOOL "Disable 1DS Swift wrapper" FORCE)

# The 1DS project does not infer its iOS source selection on the normal FetchContent path.
if(IOS OR CMAKE_SYSTEM_NAME STREQUAL "iOS")
  if(NOT PLATFORM_NAME OR NOT IPHONEOS_DEPLOYMENT_TARGET OR NOT IOS_ARCH)
    message(FATAL_ERROR
      "iOS telemetry requires PLATFORM_NAME (iphoneos/iphonesimulator), IOS_ARCH, and "
      "IPHONEOS_DEPLOYMENT_TARGET to configure the 1DS SDK.")
  endif()
  set(BUILD_IOS ON CACHE BOOL "Build the 1DS SDK for iOS" FORCE)
  set(IOS_PLAT "${PLATFORM_NAME}" CACHE STRING "1DS iOS platform" FORCE)
  set(IOS_DEPLOYMENT_TARGET "${IPHONEOS_DEPLOYMENT_TARGET}" CACHE STRING "1DS iOS deployment target" FORCE)
  set(FORCE_RESET_OSX_DEPLOYMENT_TARGET OFF CACHE BOOL "Preserve the caller's iOS deployment target" FORCE)
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

# Build sqlite3 and zlib from the SDK's vendored sources (MATSDK_BUNDLE_VENDORED_DEPS) on every
# platform so the build is self-contained and the resulting static `mat` is complete: it references
# sqlite3 and the SDK's symbol-prefixed zlib (act_z_*), which the bundled targets provide. This avoids
# depending on system libsqlite3/zlib dev packages (Linux) or absolute Homebrew archive paths (macOS),
# and on Windows it supplies the sqlite3/zlib object code that mat would otherwise leave unresolved
# (ONNX Runtime never needed this because it uses ETW, not the 1DS SDK, on Windows).
set(MATSDK_BUNDLE_VENDORED_DEPS ON CACHE BOOL "Build 1DS SDK vendored sqlite3/zlib dependencies" FORCE)

set(_ortgenai_telemetry_patch
  "${CMAKE_COMMAND}"
  "-DSOURCE_DIR=<SOURCE_DIR>"
  "-DPATCH_EXECUTABLE=${ORTGENAI_PATCH_EXECUTABLE}"
  "-DPATCH_FILE=${PROJECT_SOURCE_DIR}/cmake/patches/cpp_client_telemetry/cpp_client_telemetry.patch"
  "-DUSE_BINARY=${WIN32}"
  "-P"
  "${PROJECT_SOURCE_DIR}/cmake/patches/cpp_client_telemetry/apply_patch.cmake")

FetchContent_Declare(
  cpp_client_telemetry
  URL ${DEP_URL_cpp_client_telemetry}
  URL_HASH SHA1=${DEP_SHA1_cpp_client_telemetry}
  PATCH_COMMAND ${_ortgenai_telemetry_patch}
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(cpp_client_telemetry)
if(ANDROID)
  set(ORTGENAI_TELEMETRY_ANDROID_JAVA_SOURCE_DIR
    "${cpp_client_telemetry_SOURCE_DIR}/lib/android_build/maesdk/src/main/java")
  set(ORTGENAI_TELEMETRY_LICENSE_FILE "${cpp_client_telemetry_SOURCE_DIR}/LICENSE")
  target_sources(mat PRIVATE
    "${PROJECT_SOURCE_DIR}/cmake/telemetry/android_telemetry_bridge.cpp")
endif()
foreach(_ortgenai_1ds_cache_var
    BUILD_UNIT_TESTS
    BUILD_FUNC_TESTS
    BUILD_PRIVACYGUARD
    BUILD_SANITIZER
    BUILD_OBJC_WRAPPER
    BUILD_SWIFT_WRAPPER
    MATSDK_BUNDLE_VENDORED_DEPS)
  unset(${_ortgenai_1ds_cache_var} CACHE)
endforeach()

if(NOT TARGET mat)
  message(FATAL_ERROR "Telemetry: the 1DS SDK 'mat' target was not created by FetchContent.")
endif()

# The SDK's CMakeLists uses include_directories(${CMAKE_CURRENT_SOURCE_DIR}) (patched from
# ${CMAKE_SOURCE_DIR}) to locate its bundled nlohmann/, sqlite/, and zlib/ headers. Under FetchContent
# that variable points at genai's root, so add the SDK's real source dir as an include path.
target_include_directories(mat PRIVATE ${cpp_client_telemetry_SOURCE_DIR})

# The bundled sqlite3_bundled/zlib_bundled targets expose their vendored header dirs as PUBLIC includes
# with build-tree paths. install(EXPORT) rejects a build/source-tree include path on an exported target,
# so scope them to the build tree only.
foreach(_ortgenai_bundled_dep sqlite3_bundled zlib_bundled)
  if(TARGET ${_ortgenai_bundled_dep})
    get_target_property(_ortgenai_bundled_inc ${_ortgenai_bundled_dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(_ortgenai_bundled_inc)
      set_target_properties(${_ortgenai_bundled_dep} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${_ortgenai_bundled_inc}>")
    endif()
  endif()
endforeach()

# The SDK's bundled sqlite3/zlib targets carry GCC/Clang-only flags that MSVC either warns on or rejects
# outright: zlib is compiled with Z_HAVE_UNISTD_H (MSVC has no <unistd.h>, so zconf.h fails to include
# it), and sqlite3 with -fno-finite-math-only / -Wno-unused-function (MSVC reads the latter as /Wno-...
# and errors D8021). Strip these on Windows -- none are needed there. This keeps genai's copy of the
# patch identical to ONNX Runtime's, whose FetchContent path is non-Windows only and so never hit this.
if(WIN32)
  if(TARGET zlib_bundled)
    get_target_property(_ortgenai_zlib_defs zlib_bundled COMPILE_DEFINITIONS)
    if(_ortgenai_zlib_defs)
      list(REMOVE_ITEM _ortgenai_zlib_defs Z_HAVE_UNISTD_H)
      set_target_properties(zlib_bundled PROPERTIES COMPILE_DEFINITIONS "${_ortgenai_zlib_defs}")
    endif()
  endif()
  if(TARGET sqlite3_bundled)
    get_target_property(_ortgenai_sqlite_opts sqlite3_bundled COMPILE_OPTIONS)
    if(_ortgenai_sqlite_opts)
      list(REMOVE_ITEM _ortgenai_sqlite_opts -fno-finite-math-only -Wno-unused-function)
      set_target_properties(sqlite3_bundled PROPERTIES COMPILE_OPTIONS "${_ortgenai_sqlite_opts}")
    endif()
  endif()
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
  target_compile_options(mat PRIVATE /wd5038)
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

# `mat` already exports lib/include/public as a PUBLIC build-interface include, which covers the
# LogManager / LogManagerProvider headers genai uses; also add include/mat so any transitive SDK headers
# resolve, matching ONNX Runtime's telemetry include set. These are build-tree paths, which is fine:
# onnxruntime-genai links this target PRIVATE and never installs/exports it.
target_include_directories(onnxruntime-genai-telemetry INTERFACE
  ${cpp_client_telemetry_SOURCE_DIR}/lib/include/public
  ${cpp_client_telemetry_SOURCE_DIR}/lib/include/mat)

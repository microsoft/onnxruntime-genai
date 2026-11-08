# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.28)
project(TelemetryAppleDependencies LANGUAGES CXX)

if(NOT MODE MATCHES "^(missing|preserved|sqlite-preserved|zlib-preserved)$")
  message(FATAL_ERROR "Expected an Apple dependency regression mode")
endif()

# Configure the real integration module on any host; this fixture models the SDK's vcpkg branch
# without downloading dependencies or requiring an Apple compiler/sysroot.
set(APPLE TRUE)
set(ENABLE_TELEMETRY ON)
set(CMAKE_DISABLE_FIND_PACKAGE_MSTelemetry TRUE)
set(BUILD_SHARED_LIBS ON)
set(BUILD_VERSION "caller-version")
foreach(dependency SQLite3 ZLIB)
  if(dependency STREQUAL "SQLite3")
    set(target SQLite3::SQLite3)
    set(preserve_mode sqlite-preserved)
  else()
    set(target ZLIB::ZLIB)
    set(preserve_mode zlib-preserved)
  endif()
  if(MODE STREQUAL "preserved" OR MODE STREQUAL preserve_mode)
    add_library(${target} INTERFACE IMPORTED)
    set_target_properties(${target} PROPERTIES
      INTERFACE_LINK_LIBRARIES "caller-${dependency}"
      INTERFACE_COMPILE_DEFINITIONS "CALLER_${dependency}")
  endif()
endforeach()

set(sdk_dir "${CMAKE_CURRENT_BINARY_DIR}/sdk")
file(MAKE_DIRECTORY "${sdk_dir}/lib/pal" "${sdk_dir}/lib/api")
file(WRITE "${sdk_dir}/lib/pal/PAL.cpp" [=[
        static const std::string sdkVersion(EVTSDK_VERSION_PREFIX "-" OS_NAME "-C++-" ECS_SUPP "-" BUILD_VERSION_STR);
        return sdkVersion;
]=])
file(WRITE "${sdk_dir}/lib/api/AuthTokensController.cpp" [=[
#include "AuthTokensController.hpp"
        static std::mutex mutex;
        return mutex;
]=])
file(WRITE "${sdk_dir}/CMakeLists.txt" [=[
set(MATSDK_USING_VCPKG ON)
if(NOT MATSDK_SQLITE_PROVIDER STREQUAL "SYSTEM" OR NOT MATSDK_ZLIB_PROVIDER STREQUAL "SYSTEM")
  message(FATAL_ERROR "Apple dependencies must use SYSTEM providers")
endif()
if(NOT TARGET SQLite3::SQLite3 OR NOT TARGET ZLIB::ZLIB)
  message(FATAL_ERROR "Canonical Apple targets must exist before the SDK detects vcpkg")
endif()
add_library(mat STATIC lib/placeholder.cpp)
set_property(TARGET mat PROPERTY SOURCES "pal/PAL.cpp;api/AuthTokensController.cpp")
target_link_libraries(mat PUBLIC SQLite3::SQLite3 ZLIB::ZLIB)
]=])
set(FETCHCONTENT_SOURCE_DIR_CPP_CLIENT_TELEMETRY "${sdk_dir}" CACHE PATH "" FORCE)
set(DEP_URL_cpp_client_telemetry "https://unused.invalid/sdk.zip")
set(DEP_SHA1_cpp_client_telemetry "0000000000000000000000000000000000000000")
include("${GENAI_SOURCE_DIR}/cmake/telemetry.cmake")

foreach(dependency SQLite3 ZLIB)
  if(dependency STREQUAL "SQLite3")
    set(target SQLite3::SQLite3)
    set(expected sqlite3)
    set(preserve_mode sqlite-preserved)
  else()
    set(target ZLIB::ZLIB)
    set(expected z)
    set(preserve_mode zlib-preserved)
  endif()
  if(MODE STREQUAL "preserved" OR MODE STREQUAL preserve_mode)
    set(expected "caller-${dependency}")
    get_target_property(definitions ${target} INTERFACE_COMPILE_DEFINITIONS)
    if(NOT definitions STREQUAL "CALLER_${dependency}")
      message(FATAL_ERROR "Caller-supplied ${target} properties were overwritten")
    endif()
  endif()
  get_target_property(links ${target} INTERFACE_LINK_LIBRARIES)
  get_target_property(imported ${target} IMPORTED)
  get_target_property(type ${target} TYPE)
  if(NOT links STREQUAL expected OR NOT imported OR NOT type STREQUAL "INTERFACE_LIBRARY")
    message(FATAL_ERROR "Unexpected ${target}: ${links}, ${imported}, ${type}")
  endif()
endforeach()
if(NOT BUILD_SHARED_LIBS OR NOT BUILD_VERSION STREQUAL "caller-version")
  message(FATAL_ERROR "Telemetry did not restore caller build settings")
endif()
message(STATUS "PASS: Apple SYSTEM dependencies with vcpkg (${MODE})")

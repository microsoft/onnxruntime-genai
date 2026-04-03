# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# CMake module for 1DS C++ SDK (MAT) telemetry integration.
#
# The 1DS SDK is consumed via the official vcpkg port "cpp-client-telemetry",
# which installs the MSTelemetry CMake CONFIG package exposing the
# MSTelemetry::mat target.
#
# To build with telemetry enabled, configure with the vcpkg toolchain and
# activate the "telemetry" manifest feature, e.g.:
#   cmake -DENABLE_TELEMETRY=ON \
#         -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake \
#         -DVCPKG_MANIFEST_FEATURES=telemetry ...
# (CMakeLists.txt activates the "telemetry" manifest feature automatically when
#  ENABLE_TELEMETRY is set before project().)

if(NOT ENABLE_TELEMETRY)
  return()
endif()

message(STATUS "Telemetry is enabled. Using 1DS SDK via the cpp-client-telemetry vcpkg port.")

find_package(MSTelemetry CONFIG REQUIRED)

# Minimum-binary-footprint guidance.
#
# The single biggest lever is STATIC linkage: with a dynamic triplet the full
# mat.dll (plus sqlite3.dll / zlib.dll) is shipped regardless of how little of
# the SDK we use, whereas a static triplet links only the referenced SDK code
# into onnxruntime-genai and lets the linker dead-strip the rest (the SDK is
# built with function/data sections in >= 3.10.173.1; the consumer enables
# /OPT:REF,ICF // --gc-sections // -dead_strip in CMakeLists.txt). The json1
# SQLite extension is also dropped via the root vcpkg.json (~52 KB).
#
# Configure with a static triplet to realize this, e.g.:
#   -DVCPKG_TARGET_TRIPLET=x64-windows-static-md   (Windows, dynamic CRT)
#   -DVCPKG_TARGET_TRIPLET=x64-linux               (Linux)
if(TARGET MSTelemetry::mat)
  get_target_property(_mat_type MSTelemetry::mat TYPE)
  if(_mat_type STREQUAL "SHARED_LIBRARY")
    message(STATUS
      "Telemetry: MSTelemetry::mat is a SHARED library, so mat.dll (and sqlite3/zlib) "
      "will be shipped. For minimum binary footprint, configure with a static triplet "
      "(e.g. -DVCPKG_TARGET_TRIPLET=x64-windows-static-md) to statically link and "
      "dead-strip the unused SDK code.")
  endif()
endif()

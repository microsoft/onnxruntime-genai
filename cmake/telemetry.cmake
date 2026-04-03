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

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT DEFINED SOURCE_DIR OR
   NOT DEFINED PATCH_EXECUTABLE OR
   NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "SOURCE_DIR, PATCH_EXECUTABLE, and PATCH_FILE are required.")
endif()

set(_marker_file "${SOURCE_DIR}/lib/CMakeLists.txt")
if(NOT EXISTS "${_marker_file}")
  message(FATAL_ERROR "1DS source marker file not found: ${_marker_file}")
endif()

file(READ "${_marker_file}" _marker_contents)
if(_marker_contents MATCHES "MATSDK_BUNDLE_VENDORED_DEPS")
  message(STATUS "1DS patch is already applied.")
  return()
endif()

set(_patch_command "${PATCH_EXECUTABLE}")
if(USE_BINARY)
  list(APPEND _patch_command --binary)
endif()
list(APPEND _patch_command -l -p1 -i "${PATCH_FILE}")

execute_process(
  COMMAND ${_patch_command}
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE _patch_result
  OUTPUT_VARIABLE _patch_output
  ERROR_VARIABLE _patch_error)

if(NOT _patch_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to patch the 1DS SDK (exit ${_patch_result}).\n"
    "${_patch_output}${_patch_error}")
endif()

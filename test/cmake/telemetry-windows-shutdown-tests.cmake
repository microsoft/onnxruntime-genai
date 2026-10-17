# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT WIN32 OR NOT ENABLE_TELEMETRY OR CMAKE_CROSSCOMPILING)
  return()
endif()

# An archive lets the fixture compile telemetry.cpp with its private test peer while reusing
# production dependencies without pulling in a second copy of telemetry.cpp's object.
add_library(telemetry_windows_shutdown_core STATIC $<TARGET_OBJECTS:onnxruntime-genai-obj>)
target_link_libraries(telemetry_windows_shutdown_core INTERFACE onnxruntime-genai-obj)
add_library(telemetry_windows_shutdown_fixture SHARED
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/telemetry/telemetry-windows-shutdown-fixture.cpp")
target_link_libraries(telemetry_windows_shutdown_fixture PRIVATE telemetry_windows_shutdown_core)
target_include_directories(telemetry_windows_shutdown_fixture PRIVATE
  "${CMAKE_SOURCE_DIR}/src" "$<TARGET_PROPERTY:onnxruntime-genai-obj,INCLUDE_DIRECTORIES>")
target_compile_definitions(telemetry_windows_shutdown_fixture PRIVATE
  $<TARGET_PROPERTY:onnxruntime-genai-obj,COMPILE_DEFINITIONS>)
add_dependencies(telemetry_windows_shutdown_fixture onnxruntime-genai)
set_target_properties(telemetry_windows_shutdown_fixture PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:onnxruntime-genai>" FOLDER "Tests")

add_executable(telemetry_windows_shutdown_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/telemetry/telemetry-windows-shutdown-tests.cpp")
add_dependencies(telemetry_windows_shutdown_tests telemetry_windows_shutdown_fixture)
set_target_properties(telemetry_windows_shutdown_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:onnxruntime-genai>" FOLDER "Tests")
foreach(mode process-exit locked-process-exit shutdown-locked-process-exit normal-destruction explicit-shutdown explicit-unload)
  add_test(NAME TelemetryWindowsShutdown-${mode}
    COMMAND telemetry_windows_shutdown_tests ${mode} "$<TARGET_FILE:telemetry_windows_shutdown_fixture>")
  set_tests_properties(TelemetryWindowsShutdown-${mode} PROPERTIES TIMEOUT 20)
endforeach()

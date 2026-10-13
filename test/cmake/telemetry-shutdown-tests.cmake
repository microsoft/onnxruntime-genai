# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT ENABLE_TELEMETRY OR CMAKE_CROSSCOMPILING OR ANDROID OR IOS)
  return()
endif()

set(TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR "${cpp_client_telemetry_SOURCE_DIR}" CACHE PATH
  "Matching 1DS source tree for deterministic shutdown regressions (also supports static CONFIG packages)")
if(NOT TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR)
  message(STATUS "Telemetry shutdown regression requires TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR for CONFIG packages")
  return()
endif()
if(TARGET mat)
  set(shutdown_sdk_target mat)
else()
  set(shutdown_sdk_target MSTelemetry::mat)
endif()

set(shutdown_sdk_dir "${CMAKE_CURRENT_BINARY_DIR}/telemetry_shutdown_sdk")
file(MAKE_DIRECTORY "${shutdown_sdk_dir}")
foreach(source callbacks/DebugSource.cpp pal/WorkerThread.cpp api/Logger.cpp)
  set(source_path "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/${source}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_path}")
  file(READ "${source_path}" instrumented)
  if(source STREQUAL "callbacks/DebugSource.cpp")
    string(REGEX MATCHALL "DE_LOCKGUARD\\(stateLock\\(\\)\\)" sites "${instrumented}")
    list(LENGTH sites count)
    if(NOT count EQUAL 5)
      message(FATAL_ERROR "Update shutdown regression debug-lock instrumentation")
    endif()
    string(REPLACE "DE_LOCKGUARD(stateLock());"
      "TelemetryTestBeforeDebugLock();\n        (void)stateLock();\n        TelemetryTestDebugLockInitialized();\n        DE_LOCKGUARD(stateLock());"
      instrumented "${instrumented}")
    string(PREPEND instrumented
      "extern \"C\" void TelemetryTestBeforeDebugLock();\nextern \"C\" void TelemetryTestDebugLockInitialized();\n")
  elseif(source STREQUAL "pal/WorkerThread.cpp")
    string(REGEX MATCHALL "threadToJoin\\.join\\(\\)" sites "${instrumented}")
    list(LENGTH sites count)
    if(NOT count EQUAL 2)
      message(FATAL_ERROR "Update shutdown regression native-join instrumentation")
    endif()
    string(REPLACE "threadToJoin.join();"
      "threadToJoin.join();\n                    TelemetryTestWorkerJoined(this);"
      instrumented "${instrumented}")
    set(start_site "m_hThread = std::thread(WorkerThread::threadFunc, static_cast<void*>(this));")
    string(FIND "${instrumented}" "${start_site}" start_position)
    if(start_position EQUAL -1)
      message(FATAL_ERROR "Update shutdown regression worker-start instrumentation")
    endif()
    string(REPLACE "${start_site}" "TelemetryTestWorkerStarted(this);\n            ${start_site}"
      instrumented "${instrumented}")
    string(PREPEND instrumented
      "extern \"C\" void TelemetryTestWorkerStarted(void*);\nextern \"C\" void TelemetryTestWorkerJoined(void*);\n")
  else()
    # Consume uploads only in this executable, before they reach the queue or collector.
    set(log_site "void Logger::LogEvent(EventProperties const& properties)\n    {")
    string(FIND "${instrumented}" "${log_site}" log_position)
    if(log_position EQUAL -1)
      message(FATAL_ERROR "Update shutdown regression collector-event interception")
    endif()
    string(REPLACE "${log_site}"
      "${log_site}\n        TelemetryTestCollectorEvent(properties.GetName().c_str());\n        return;"
      instrumented "${instrumented}")
    string(PREPEND instrumented "extern \"C\" void TelemetryTestCollectorEvent(const char*);\n")
  endif()
  get_filename_component(name "${source}" NAME)
  file(WRITE "${shutdown_sdk_dir}/${name}" "${instrumented}")
endforeach()

add_library(telemetry_shutdown_sdk OBJECT
  "${shutdown_sdk_dir}/DebugSource.cpp" "${shutdown_sdk_dir}/WorkerThread.cpp" "${shutdown_sdk_dir}/Logger.cpp")
target_link_libraries(telemetry_shutdown_sdk PRIVATE ${shutdown_sdk_target})
target_include_directories(telemetry_shutdown_sdk PRIVATE
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib" "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/api"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include/mat")
if(TARGET mat)
  target_compile_definitions(telemetry_shutdown_sdk PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
  target_compile_options(telemetry_shutdown_sdk PRIVATE $<TARGET_PROPERTY:mat,COMPILE_OPTIONS>)
  target_include_directories(telemetry_shutdown_sdk PRIVATE $<TARGET_PROPERTY:mat,INCLUDE_DIRECTORIES>)
endif()
add_executable(telemetry_shutdown_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/telemetry/telemetry-shutdown-tests.cpp"
  $<TARGET_OBJECTS:telemetry_shutdown_sdk>)
target_link_libraries(telemetry_shutdown_tests PRIVATE onnxruntime-genai-obj ${shutdown_sdk_target})
target_include_directories(telemetry_shutdown_tests PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib" "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include/mat")
add_dependencies(telemetry_shutdown_tests onnxruntime-genai)
if(WIN32)
  add_custom_command(TARGET telemetry_shutdown_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ORT_LIB_DIR}/onnxruntime.dll"
      "$<TARGET_FILE_DIR:telemetry_shutdown_tests>/onnxruntime.dll")
endif()
set_target_properties(telemetry_shutdown_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:onnxruntime-genai>" FOLDER "Tests")

foreach(mode retained threaded recreated explicit prime-only disabled ci-disabled)
  set(home "${CMAKE_CURRENT_BINARY_DIR}/telemetry-shutdown-${mode}")
  file(MAKE_DIRECTORY "${home}")
  set(disabled 0)
  set(ci 0)
  if(mode STREQUAL "disabled")
    set(disabled 1)
  elseif(mode STREQUAL "ci-disabled")
    set(ci 1)
  endif()
  add_test(NAME TelemetryShutdown-${mode}
    COMMAND ${CMAKE_COMMAND} -E env
      "HOME=${home}" "XDG_CACHE_HOME=${home}" "APPDATA=${home}" "LOCALAPPDATA=${home}"
      "CI=${ci}" TF_BUILD=0 GITHUB_ACTIONS=0 GITLAB_CI=0 CIRCLECI=0 TRAVIS=0
      JENKINS_URL=0 CODEBUILD_BUILD_ID=0 BUILDKITE=0 TEAMCITY_VERSION=0 APPVEYOR=0
      BITBUCKET_BUILD_NUMBER=0 SYSTEM_TEAMFOUNDATIONCOLLECTIONURI=0 ORT_RUNNING_UNIT_TESTS=0
      "ORT_DISABLE_TELEMETRY=${disabled}" $<TARGET_FILE:telemetry_shutdown_tests> ${mode})
  set_tests_properties(TelemetryShutdown-${mode} PROPERTIES TIMEOUT 30)
endforeach()

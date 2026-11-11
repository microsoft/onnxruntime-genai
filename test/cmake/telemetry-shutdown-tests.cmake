# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT ENABLE_TELEMETRY OR CMAKE_CROSSCOMPILING OR ANDROID OR IOS)
  return()
endif()

foreach(mode missing preserved sqlite-preserved zlib-preserved sqlite-alternate sqlite-both)
  add_test(NAME TelemetryAppleDependencies-${mode}
    COMMAND ${CMAKE_COMMAND}
      -S "${CMAKE_CURRENT_LIST_DIR}/telemetry-dependencies"
      -B "${CMAKE_CURRENT_BINARY_DIR}/telemetry-apple-dependencies-${mode}"
      "-DGENAI_SOURCE_DIR=${CMAKE_SOURCE_DIR}" "-DMODE=${mode}")
endforeach()

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
foreach(source callbacks/DebugSource.cpp pal/WorkerThread.cpp api/Logger.cpp pal/PAL.cpp
    api/AuthTokensController.cpp http/HttpRequestEncoder.cpp)
  set(source_path "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/${source}")
  if(source STREQUAL "pal/PAL.cpp" AND ORTGENAI_TELEMETRY_PAL_SOURCE)
    set(source_path "${ORTGENAI_TELEMETRY_PAL_SOURCE}")
  elseif(source STREQUAL "api/AuthTokensController.cpp" AND ORTGENAI_TELEMETRY_AUTH_SOURCE)
    set(source_path "${ORTGENAI_TELEMETRY_AUTH_SOURCE}")
  endif()
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
      "void TelemetryTestBeforeDebugLock();\nextern \"C\" void TelemetryTestDebugLockInitialized();\n")
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
  elseif(source STREQUAL "api/Logger.cpp")
    # Observe real decoration, queuing, persistence, and encoding; replace only HTTP transport.
    set(log_site "void Logger::LogEvent(EventProperties const& properties)\n    {")
    string(FIND "${instrumented}" "${log_site}" log_position)
    if(log_position EQUAL -1)
      message(FATAL_ERROR "Update shutdown regression collector-event interception")
    endif()
    string(REPLACE "${log_site}"
      "${log_site}\n        TelemetryTestCollectorEvent(properties.GetName().c_str());"
      instrumented "${instrumented}")
    set(decorated_site "DispatchEvent(DebugEvent(DebugEventType::EVT_LOG_EVENT, size_t(latency), size_t(0), static_cast<void*>(&record), sizeof(record)));")
    string(FIND "${instrumented}" "${decorated_site}" decorated_position)
    if(decorated_position EQUAL -1)
      message(FATAL_ERROR "Update telemetry client-identity decoration interception")
    endif()
    string(REPLACE "${decorated_site}"
      "TelemetryTestDecoratedEvent(properties.GetName().c_str(), record);\n        ${decorated_site}"
      instrumented "${instrumented}")
    string(PREPEND instrumented
      "namespace CsProtocol { struct Record; }\nextern \"C\" void TelemetryTestCollectorEvent(const char*);\nvoid TelemetryTestDecoratedEvent(const char*, const CsProtocol::Record&);\n")
  elseif(source STREQUAL "pal/PAL.cpp")
    # Keep destructor observers for the original/immortal-string negative controls.
    string(REPLACE "using SdkVersion = std::string;" "using SdkVersion = TelemetryTestSdkVersion;"
      instrumented "${instrumented}")
    string(REPLACE "static const std::string sdkVersion(" "static const TelemetryTestSdkVersion sdkVersion("
      instrumented "${instrumented}")
    string(REPLACE "return *sdkVersion;" "TelemetryTestVersionAccess();\n        return *sdkVersion;"
      instrumented "${instrumented}")
    string(REPLACE "return sdkVersion;" "TelemetryTestVersionAccess();\n        return sdkVersion;"
      instrumented "${instrumented}")
    set(version_entry "const char* PlatformAbstractionLayer::getSdkVersion() const\n    {")
    string(REPLACE "${version_entry}" "${version_entry}\n        TelemetryTestVersionAccess();"
      instrumented "${instrumented}")
    string(REGEX MATCHALL "TelemetryTestVersionAccess\\(\\)" sites "${instrumented}")
    list(LENGTH sites count)
    if(NOT count EQUAL 1)
      message(FATAL_ERROR "Update shutdown regression SDK-version lifetime instrumentation")
    endif()
    string(PREPEND instrumented [=[
#include <string>
extern "C" void TelemetryTestVersionDestroyed();
extern "C" void TelemetryTestVersionAccess();
struct TelemetryTestSdkVersion : std::string {
    using std::string::string;
    ~TelemetryTestSdkVersion() { TelemetryTestVersionDestroyed(); }
};
]=])
  elseif(source STREQUAL "api/AuthTokensController.cpp")
    # A real destructor observer makes the libc++ failure deterministic on every host.
    string(REPLACE "static std::mutex mutex;" "static TelemetryTestAuthMutex mutex;"
      instrumented "${instrumented}")
    string(REPLACE "using AuthMutex = std::mutex;" "using AuthMutex = TelemetryTestAuthMutex;"
      instrumented "${instrumented}")
    string(REPLACE "return mutex;" "TelemetryTestAuthMutexAccess();\n        return mutex;"
      instrumented "${instrumented}")
    string(REPLACE "return *mutex;" "TelemetryTestAuthMutexAccess();\n        return *mutex;"
      instrumented "${instrumented}")
    string(REGEX MATCHALL "TelemetryTestAuthMutexAccess\\(\\)" sites "${instrumented}")
    list(LENGTH sites count)
    if(NOT count EQUAL 1)
      message(FATAL_ERROR "Update shutdown regression auth-mutex lifetime instrumentation")
    endif()
    string(PREPEND instrumented [=[
#include <mutex>
extern "C" void TelemetryTestAuthMutexDestroyed();
extern "C" void TelemetryTestAuthMutexAccess();
struct TelemetryTestAuthMutex : std::mutex {
    ~TelemetryTestAuthMutex() { TelemetryTestAuthMutexDestroyed(); }
};
]=])
  else()
    set(encode_site "bool HttpRequestEncoder::handleEncode(EventsUploadContextPtr const& ctx)\n    {")
    string(FIND "${instrumented}" "${encode_site}" encode_position)
    if(encode_position EQUAL -1)
      message(FATAL_ERROR "Update shutdown regression encoder instrumentation")
    endif()
    string(REPLACE "${encode_site}" "${encode_site}\n        TelemetryTestBeforeEncode();"
      instrumented "${instrumented}")
    string(PREPEND instrumented "extern \"C\" void TelemetryTestBeforeEncode();\n")
  endif()
  get_filename_component(name "${source}" NAME)
  file(WRITE "${shutdown_sdk_dir}/${name}" "${instrumented}")
endforeach()

add_library(telemetry_shutdown_sdk OBJECT
  "${shutdown_sdk_dir}/DebugSource.cpp" "${shutdown_sdk_dir}/WorkerThread.cpp" "${shutdown_sdk_dir}/Logger.cpp"
  "${shutdown_sdk_dir}/PAL.cpp" "${shutdown_sdk_dir}/AuthTokensController.cpp"
  "${shutdown_sdk_dir}/HttpRequestEncoder.cpp")
target_link_libraries(telemetry_shutdown_sdk PRIVATE ${shutdown_sdk_target})
target_include_directories(telemetry_shutdown_sdk PRIVATE
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib" "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/api"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/pal" "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/http"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include/mat")
if(TARGET mat)
  target_compile_definitions(telemetry_shutdown_sdk PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
  target_compile_options(telemetry_shutdown_sdk PRIVATE $<TARGET_PROPERTY:mat,COMPILE_OPTIONS>)
  target_include_directories(telemetry_shutdown_sdk PRIVATE $<TARGET_PROPERTY:mat,INCLUDE_DIRECTORIES>)
endif()
if(ORTGENAI_TELEMETRY_SDK_INCLUDE_DIR)
  target_include_directories(telemetry_shutdown_sdk BEFORE PRIVATE "${ORTGENAI_TELEMETRY_SDK_INCLUDE_DIR}")
endif()
add_executable(telemetry_shutdown_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/telemetry/telemetry-shutdown-tests.cpp"
  $<TARGET_OBJECTS:telemetry_shutdown_sdk>)
target_link_libraries(telemetry_shutdown_tests PRIVATE onnxruntime-genai-obj ${shutdown_sdk_target})
target_include_directories(telemetry_shutdown_tests PRIVATE
  "${CMAKE_SOURCE_DIR}/src"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib" "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include"
  "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include/mat")
if(ORTGENAI_TELEMETRY_SDK_INCLUDE_DIR)
  target_include_directories(telemetry_shutdown_tests BEFORE PRIVATE "${ORTGENAI_TELEMETRY_SDK_INCLUDE_DIR}")
endif()
add_dependencies(telemetry_shutdown_tests onnxruntime-genai)
set_target_properties(telemetry_shutdown_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:onnxruntime-genai>" FOLDER "Tests")

# Exercise real DLL/DSO unload so a heap-backed "immortal" string cannot hide behind live roots.
if(WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(telemetry_version_reload_fixture SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/telemetry/telemetry-version-reload-fixture.cpp")
  target_link_libraries(telemetry_version_reload_fixture PRIVATE ${shutdown_sdk_target})
  target_include_directories(telemetry_version_reload_fixture PRIVATE
    "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib"
    "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include"
    "${TELEMETRY_SHUTDOWN_SDK_SOURCE_DIR}/lib/include/mat")
  if(TARGET mat)
    target_compile_definitions(telemetry_version_reload_fixture PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
  endif()
  if(ORTGENAI_TELEMETRY_SDK_INCLUDE_DIR)
    target_include_directories(telemetry_version_reload_fixture BEFORE PRIVATE "${ORTGENAI_TELEMETRY_SDK_INCLUDE_DIR}")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(TARGET mat)
      # GNU-unique symbols would mark the fixture NODELETE and mask unload leaks.
      target_compile_options(mat PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fno-gnu-unique>)
    endif()
    target_link_options(telemetry_version_reload_fixture PRIVATE
      "LINKER:--exclude-libs,ALL" "LINKER:-Bsymbolic")
  endif()
  add_executable(telemetry_version_reload_tests
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/telemetry/telemetry-version-reload-tests.cpp")
  target_compile_features(telemetry_version_reload_fixture PRIVATE cxx_std_17)
  target_compile_features(telemetry_version_reload_tests PRIVATE cxx_std_17)
  target_link_libraries(telemetry_version_reload_tests PRIVATE ${CMAKE_DL_LIBS})
  add_dependencies(telemetry_version_reload_tests telemetry_version_reload_fixture)
  add_test(NAME TelemetryVersionReload
    COMMAND telemetry_version_reload_tests $<TARGET_FILE:telemetry_version_reload_fixture>)
  set_tests_properties(TelemetryVersionReload PROPERTIES TIMEOUT 30)
endif()

set(shutdown_modes retained threaded recreated explicit prime-only disabled ci-disabled prime-failed prime-allocation-failed upload-at-exit upload-after-reinit)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND shutdown_modes upload-overridden-argv0)
endif()
foreach(mode IN LISTS shutdown_modes)
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
  if(mode MATCHES "^upload-")
    set_tests_properties(TelemetryShutdown-${mode} PROPERTIES TIMEOUT 60)
  endif()
endforeach()

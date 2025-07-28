# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(ORT_HOME)
  # If ORT_HOME is specified at build time, use ORT_HOME to get the onnxruntime headers and libraries
  message(STATUS "Using ONNX Runtime from ${ORT_HOME}")

  if (ANDROID)
    # Paths are based on the directory structure of the ORT Android AAR.
    set(ORT_HEADER_DIR ${ORT_HOME}/headers)
    set(ORT_LIB_DIR ${ORT_HOME}/jni/${ANDROID_ABI})
  elseif (IOS OR MAC_CATALYST)
    set(ORT_HEADER_DIR ${ORT_HOME}/Headers)
    set(ORT_LIB_DIR ${ORT_HOME}/)
  elseif (CMAKE_SYSTEM_NAME MATCHES "AIX")
    set(ORT_HEADER_DIR ${ORT_HOME}/include/onnxruntime)
    set(ORT_LIB_DIR ${ORT_HOME}/lib)
  else()
    set(ORT_HEADER_DIR ${ORT_HOME}/include)
    set(ORT_LIB_DIR ${ORT_HOME}/lib)
  endif()
else()
  # If ORT_HOME is not specified, download the onnxruntime headers and libraries from the nightly feed
  set(ORT_VERSION "1.22.0")
  set(ORT_FEED_ORG_NAME "aiinfra")
  set(ORT_FEED_PROJECT "2692857e-05ef-43b4-ba9c-ccf1c22c437c")
  set(ORT_NIGHTLY_FEED_ID "7982ae20-ed19-4a35-a362-a96ac99897b7")

  if (USE_DML)
    set(ORT_VERSION "1.22.0")
    set(ORT_PACKAGE_NAME "Microsoft.ML.OnnxRuntime.DirectML")
  elseif(USE_CUDA)
    set(ORT_VERSION "1.22.0")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(ORT_PACKAGE_NAME "Microsoft.ML.OnnxRuntime.Gpu.Linux")
    elseif(WIN32)
      set(ORT_PACKAGE_NAME "Microsoft.ML.OnnxRuntime.Gpu.Windows")
    else()
      message(FATAL_ERROR "Unsupported platform for CUDA")
    endif()
  elseif(USE_ROCM)
    set(ORT_VERSION "1.22.0")
    set(ORT_PACKAGE_NAME "Microsoft.ML.OnnxRuntime.Rocm")
  else()
    set(ORT_PACKAGE_NAME "Microsoft.ML.OnnxRuntime")
  endif()

  set(ORT_FETCH_URL "https://pkgs.dev.azure.com/${ORT_FEED_ORG_NAME}/${ORT_FEED_PROJECT}/_apis/packaging/feeds/${ORT_NIGHTLY_FEED_ID}/nuget/packages/${ORT_PACKAGE_NAME}/versions/${ORT_VERSION}/content?api-version=6.0-preview.1")

  message(STATUS "Using ONNX Runtime package ${ORT_PACKAGE_NAME} version ${ORT_VERSION}")

  FetchContent_Declare(
    ortlib
    URL ${ORT_FETCH_URL}
  )
  FetchContent_makeAvailable(ortlib)

  if(USE_DML)
    set(ORT_HEADER_DIR ${ortlib_SOURCE_DIR}/build/native/include)
  elseif(USE_CUDA)
    set(ORT_HEADER_DIR ${ortlib_SOURCE_DIR}/buildTransitive/native/include)
  else()
    set(ORT_HEADER_DIR ${ortlib_SOURCE_DIR}/build/native/include)
  endif()

  if(ANDROID)
    file(ARCHIVE_EXTRACT INPUT ${ortlib_SOURCE_DIR}/runtimes/android/native/onnxruntime.aar DESTINATION ${ortlib_SOURCE_DIR}/runtimes/android/native/)
    set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/android/native/jni/${ANDROID_ABI})
  elseif (IOS OR MAC_CATALYST)
    file(ARCHIVE_EXTRACT INPUT ${ortlib_SOURCE_DIR}/runtimes/ios/native/onnxruntime.xcframework.zip DESTINATION ${ortlib_SOURCE_DIR}/runtimes/ios/native/)
    set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/ios/native/)
  else()
    set(ORT_BINARY_PLATFORM "x64")
    if (APPLE)
      if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(ORT_BINARY_PLATFORM "arm64")
      endif()
      set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/osx-${ORT_BINARY_PLATFORM}/native)
    elseif(WIN32)
      if (CMAKE_GENERATOR_PLATFORM)
        if (CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64EC" OR CMAKE_GENERATOR_PLATFORM STREQUAL "arm64")
          set(ORT_BINARY_PLATFORM "arm64")
        endif()
      elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
        set(ORT_BINARY_PLATFORM "arm64")
      endif()
      set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/win-${ORT_BINARY_PLATFORM}/native)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(ORT_BINARY_PLATFORM "arm64")
      endif()
      set(ORT_LIB_DIR ${ortlib_SOURCE_DIR}/runtimes/linux-${ORT_BINARY_PLATFORM}/native)
    else()
      message(FATAL_ERROR "Auto download ONNX Runtime for this platform is not supported.")
    endif()
  endif()
endif()

# Download DML headers and libraries
if(USE_DML)
  set(DML_VERSION "1.15.2")
  set(DML_PACKAGE_NAME "Microsoft.AI.DirectML")
  set(DML_FETCH_URL "https://www.nuget.org/api/v2/package/${DML_PACKAGE_NAME}/${DML_VERSION}")

  FetchContent_Declare(
    dmllib
    URL ${DML_FETCH_URL}
  )
  FetchContent_makeAvailable(dmllib)
  set(DML_HEADER_DIR ${dmllib_SOURCE_DIR}/build/native/include)

  set(DML_BINARY_PLATFORM "x64")
  if (CMAKE_GENERATOR_PLATFORM)
    if (CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "arm64")
      set(DML_BINARY_PLATFORM "arm64")
    elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64EC")
      set(DML_BINARY_PLATFORM "arm64ec")
    endif()
  elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    set(DML_BINARY_PLATFORM "arm64")
  endif()

  set(DML_LIB_DIR ${dmllib_SOURCE_DIR}/runtimes/bin/${DML_BINARY_PLATFORM}-win/native)

  set(D3D12_VERSION "1.614.1")
  set(D3D12_PACKAGE_NAME "Microsoft.Direct3D.D3D12")
  set(D3D12_FETCH_URL "https://www.nuget.org/api/v2/package/${D3D12_PACKAGE_NAME}/${D3D12_VERSION}")
  FetchContent_Declare(
    d3d12lib
    URL ${D3D12_FETCH_URL}
  )
  FetchContent_makeAvailable(d3d12lib)
  set(D3D12_HEADER_DIR ${d3d12lib_SOURCE_DIR}/build/native/include)

  set(D3D12_LIB_DIR ${d3d12lib_SOURCE_DIR}/build/native/bin/${DML_BINARY_PLATFORM})
endif()

# onnxruntime-extensions can use the same onnxruntime headers
set(ONNXRUNTIME_INCLUDE_DIR ${ORT_HEADER_DIR})
set(ONNXRUNTIME_LIB_DIR ${ORT_LIB_DIR})

message(STATUS "ORT_HEADER_DIR: ${ORT_HEADER_DIR}")
message(STATUS "ORT_LIB_DIR: ${ORT_LIB_DIR}")

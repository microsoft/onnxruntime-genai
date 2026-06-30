# All Global variables for the top level CMakeLists.txt should be defined here

# Define the Version Info
file(READ "VERSION_INFO" ver)
set(VERSION_INFO ${ver})

# Example:
# VERSION_INFO: 0.4.0-dev
# VERSION_STR: 0.4.0
# VERSION_SUFFIX: dev
# VERSION_MAJOR: 0
# VERSION_MINOR: 4
# VERSION_PATCH: 0
string(REPLACE "-" ";" VERSION_LIST ${VERSION_INFO})
list(GET VERSION_LIST 0 VERSION_STR)
# Check if it is a stable or dev version
list(LENGTH VERSION_LIST VERSION_LIST_LENGTH)
if(VERSION_LIST_LENGTH GREATER 1)
    list(GET VERSION_LIST 1 VERSION_SUFFIX)
else()
    set(VERSION_SUFFIX "")  # Set VERSION_SUFFIX to empty if stable version
endif()
string(REPLACE "." ";" VERSION_LIST ${VERSION_STR})
list(GET VERSION_LIST 0 VERSION_MAJOR)
list(GET VERSION_LIST 1 VERSION_MINOR)
list(GET VERSION_LIST 2 VERSION_PATCH)


# Define the project directories
set(REPO_ROOT ${PROJECT_SOURCE_DIR})
set(SRC_ROOT ${REPO_ROOT}/src)
set(GENERATORS_ROOT ${SRC_ROOT})
set(MODELS_ROOT ${SRC_ROOT}/models)
set(ENGINE_ROOT ${SRC_ROOT}/engine)

# Define the dependency libraries

if(WIN32)
  set(ONNXRUNTIME_LIB "onnxruntime.dll")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "onnxruntime_providers_cuda.dll")
elseif(APPLE)
  if(IOS OR MAC_CATALYST)
    add_library(onnxruntime IMPORTED STATIC)
    if(PLATFORM_NAME STREQUAL "macabi")
      # The xcframework in cmake doesn't seem to support MacCatalyst.
      # Without manually setting the target framework, cmake will be confused and looking for wrong libraries.
      # The error looks like: 'Unable to find suitable library in: Info.plist for system name "Darwin"'
      set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION ${ORT_LIB_DIR}/onnxruntime.xcframework/ios-arm64_x86_64-maccatalyst/onnxruntime.framework)
    else()
      set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION ${ORT_LIB_DIR}/onnxruntime.xcframework)
    endif()
    set(ONNXRUNTIME_LIB onnxruntime)
  else()
    set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
    set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "libonnxruntime_providers_cuda.dylib")
  endif()
else()
  #In AIX, only CPU inferencing is supported
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    set(ONNXRUNTIME_LIB "libonnxruntime.a")
  else()
    set(ONNXRUNTIME_LIB "libonnxruntime.so")
  endif()
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "libonnxruntime_providers_cuda.so")
endif()

file(GLOB generator_srcs CONFIGURE_DEPENDS
  "${GENERATORS_ROOT}/*.h"
  "${GENERATORS_ROOT}/*.cpp"
  "${GENERATORS_ROOT}/cpu/*.h"
  "${GENERATORS_ROOT}/cpu/*.cpp"
  "${GENERATORS_ROOT}/qnn/*.h"
  "${GENERATORS_ROOT}/qnn/*.cpp"
  "${GENERATORS_ROOT}/webgpu/*.h"
  "${GENERATORS_ROOT}/webgpu/*.cpp"
  "${GENERATORS_ROOT}/openvino/*.h"
  "${GENERATORS_ROOT}/openvino/*.cpp"
  "${GENERATORS_ROOT}/ryzenai/*.h"
  "${GENERATORS_ROOT}/ryzenai/*.cpp"
  "${GENERATORS_ROOT}/cuda/session_options.h"
  "${GENERATORS_ROOT}/cuda/session_options.cpp"
  "${GENERATORS_ROOT}/nvtensorrtrtx/*.h"
  "${GENERATORS_ROOT}/nvtensorrtrtx/*.cpp"
  "${GENERATORS_ROOT}/vitisai/*.h"
  "${GENERATORS_ROOT}/vitisai/*.cpp"
  "${GENERATORS_ROOT}/dml/session_options.h"
  "${GENERATORS_ROOT}/dml/session_options.cpp"
  "${MODELS_ROOT}/*.h"
  "${MODELS_ROOT}/*.cpp"
  "${ENGINE_ROOT}/*.h"
  "${ENGINE_ROOT}/*.cpp"
  "${ENGINE_ROOT}/decoders/*.h"
  "${ENGINE_ROOT}/decoders/*.cpp"
)

set(ortgenai_embed_libs "") # shared libs that will be embedded inside the onnxruntime-genai package

if (IOS OR MAC_CATALYST)
  if (NOT EXISTS "${ORT_LIB_DIR}/onnxruntime.xcframework")
    message(FATAL_ERROR "Expected the ONNX Runtime XCFramework to be found at ${ORT_LIB_DIR}/onnxruntime.xcframework. Actual: Not found.")
  endif()
elseif (USE_WINML)
    message(STATUS "Using WinML, does NOT include ONNX Runtime library, is provied by Windows.")
else()
  if(NOT EXISTS "${ORT_LIB_DIR}/${ONNXRUNTIME_LIB}")
    message(FATAL_ERROR "Expected the ONNX Runtime library to be found at ${ORT_LIB_DIR}/${ONNXRUNTIME_LIB}. Actual: Not found.")
  endif()
endif()

if(NOT EXISTS "${ORT_HEADER_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "Expected the ONNX Runtime C API header to be found at \"${ORT_HEADER_DIR}/onnxruntime_c_api.h\". Actual: Not found.")
endif()


# normalize the target platform to x64 or arm64. additional architectures can be added as needed.
include(${CMAKE_CURRENT_LIST_DIR}/target_platform.cmake)

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

# Define the dependency libraries

if(WIN32)
  set(ONNXRUNTIME_LIB "onnxruntime.dll")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "onnxruntime_providers_cuda.dll")
  set(ONNXRUNTIME_PROVIDERS_ROCM_LIB "onnxruntime_providers_rocm.dll")
elseif(APPLE)
  set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "libonnxruntime_providers_cuda.dylib")
  set(ONNXRUNTIME_PROVIDERS_ROCM_LIB "libonnxruntime_providers_rocm.dylib")
else()
  set(ONNXRUNTIME_LIB "libonnxruntime.so")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "libonnxruntime_providers_cuda.so")
  set(ONNXRUNTIME_PROVIDERS_ROCM_LIB "libonnxruntime_providers_rocm.so")
endif()

file(GLOB generator_srcs CONFIGURE_DEPENDS
  "${GENERATORS_ROOT}/*.h"
  "${GENERATORS_ROOT}/*.cpp"
  "${MODELS_ROOT}/*.h"
  "${MODELS_ROOT}/*.cpp"
)

set(ortgenai_embed_libs "") # shared libs that will be embedded inside the onnxruntime-genai package

if(NOT EXISTS "${ORT_LIB_DIR}/${ONNXRUNTIME_LIB}")
  message(FATAL_ERROR "Expected the ONNX Runtime library to be found at ${ORT_LIB_DIR}/${ONNXRUNTIME_LIB}. Actual: Not found.")
endif()
if(NOT EXISTS "${ORT_HEADER_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "Expected the ONNX Runtime C API header to be found at \"${ORT_HEADER_DIR}/onnxruntime_c_api.h\". Actual: Not found.")
endif()


# normalize the target platform to x64 or arm64. additional architectures can be added as needed.
if (MSVC)
  if (CMAKE_VS_PLATFORM_NAME)
    # cross-platform generator
    set(genai_target_platform ${CMAKE_VS_PLATFORM_NAME})
  else()
    set(genai_target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()

  if (genai_target_platform STREQUAL "arm64")
    # pass
  elseif (genai_target_platform STREQUAL "ARM64" OR
          genai_target_platform STREQUAL "ARM64EC")
    set(genai_target_platform "arm64")
  elseif (genai_target_platform STREQUAL "x64" OR 
          genai_target_platform STREQUAL "x86_64" OR 
          genai_target_platform STREQUAL "AMD64" OR 
          CMAKE_GENERATOR MATCHES "Win64")
    set(genai_target_platform "x64")
  else()
    message(FATAL_ERROR "Unsupported architecture. CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
elseif(APPLE)
  # TODO: do we need to support CMAKE_OSX_ARCHITECTURES having multiple values?
  set(_apple_target_arch ${CMAKE_OSX_ARCHITECTURES})
  if (NOT _apple_target_arch)
    set(_apple_target_arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
  endif()

  if (_apple_target_arch STREQUAL "arm64")
    set(genai_target_platform "arm64")
  elseif (_apple_target_arch STREQUAL "x86_64")
    set(genai_target_platform "x64")
  else()
    message(FATAL_ERROR "Unsupported architecture. ${_apple_target_arch}")
  endif()
elseif(ANDROID)
  if (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(genai_target_platform "arm64")
  elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
    set(genai_target_platform "x64")
  else()
    message(FATAL_ERROR "Unsupported architecture. CMAKE_ANDROID_ARCH_ABI: ${CMAKE_ANDROID_ARCH_ABI}")
  endif()
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
    set(genai_target_platform "arm64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
    set(genai_target_platform "arm64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
    set(genai_target_platform "x64")
  else()
    message(FATAL_ERROR "Unsupported architecture. CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()

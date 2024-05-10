# All Global variables for the top level CMakeLists.txt should be defined here

# Define the Version Info
file(READ "VERSION_INFO" ver)
set(VERSION_INFO ${ver} CACHE STRING "Set the onnxruntime-genai version info.")
message("Building onnxruntime-genai for version ${VERSION_INFO}")


# Define the project directories
set(GENERATORS_ROOT ${PROJECT_SOURCE_DIR}/src)
set(MODELS_ROOT ${PROJECT_SOURCE_DIR}/src/models)
set(ORT_HOME ${CMAKE_SOURCE_DIR}/ort CACHE PATH "Path to the onnxruntime root directory.")

if (ANDROID)
  # Paths are based on the directory structure of the ORT Android AAR.
  set(ORT_HEADER_DIR ${ORT_HOME}/headers)
  set(ORT_LIB_DIR ${ORT_HOME}/jni/${ANDROID_ABI})
else()
  set(ORT_HEADER_DIR ${ORT_HOME}/include)
  set(ORT_LIB_DIR ${ORT_HOME}/lib)
endif()

# Define the dependency libraries

if(WIN32)
  set(ONNXRUNTIME_LIB "onnxruntime.dll")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "onnxruntime_providers_cuda.dll")
  set(ONNXRUNTIME_ALL_SHARED_LIBS "onnxruntime*.dll")
  set(ONNXRUNTIME_EXTENSIONS_LIB "tfmtok_c.lib")
  set(ONNXRUNTIME_EXTENSIONS_FILES "tfmtok_c.dll")
elseif(APPLE)
  set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "libonnxruntime_providers_cuda.dylib")
  set(ONNXRUNTIME_ALL_SHARED_LIBS "libonnxruntime*.dylib")
else()
  set(ONNXRUNTIME_LIB "libonnxruntime.so")
  set(ONNXRUNTIME_PROVIDERS_CUDA_LIB "libonnxruntime_providers_cuda.so")
  set(ONNXRUNTIME_ALL_SHARED_LIBS "libonnxruntime*.so*")
  set(ONNXRUNTIME_EXTENSIONS_LIB "tfmtok_c.so")
endif()

file(GLOB generator_srcs CONFIGURE_DEPENDS
  "${GENERATORS_ROOT}/*.h"
  "${GENERATORS_ROOT}/*.cpp"
  "${MODELS_ROOT}/*.h"
  "${MODELS_ROOT}/*.cpp"
)

file(GLOB onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_ALL_SHARED_LIBS}")

if(NOT EXISTS "${ORT_LIB_DIR}/${ONNXRUNTIME_LIB}")
  message(FATAL_ERROR "Expected the ONNX Runtime library to be found at ${ORT_LIB_DIR}/${ONNXRUNTIME_LIB}. Actual: Not found.")
endif()
if(NOT EXISTS "${ORT_HEADER_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "Expected the ONNX Runtime C API header to be found at \"${ORT_HEADER_DIR}/onnxruntime_c_api.h\". Actual: Not found.")
endif()
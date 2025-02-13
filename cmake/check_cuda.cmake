# Checking if CUDA is supported
include(CheckLanguage)

if(USE_CUDA)
  # Temporary add -allow-unsupported-compiler
  # Do this before enable_cuda
  if(WIN32 AND NOT CMAKE_CUDA_FLAGS_INIT)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  endif()

  enable_language(CUDA)
  message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  if(CMAKE_CUDA_COMPILER)
    # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
    cmake_policy(SET CMP0104 OLD)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  else()
    message(FATAL_ERROR "CUDA is not supported")
  endif()
endif()

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 8)
  message(FATAL_ERROR "GCC version must be greater than or equal to 8")
endif()

# CUDA Being enabled will make it not a debug build without this option, so all of the C++ headers will complain
# about a mismatch with the actual debug headers and it'll fail to link. I don't know why this happens, or if this is the best fix.
if(USE_CUDA AND CMAKE_CUDA_COMPILER AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_definitions(_DEBUG=1)
endif()

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
  # cmake_policy(SET CMP0104 OLD)
  # enable_language(CUDA)
  # message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  # set(CUDA_PROPAGATE_HOST_FLAGS ON)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2803 --expt-relaxed-constexpr")

  file(GLOB generator_cudalib_srcs CONFIGURE_DEPENDS
    "${GENERATORS_ROOT}/cuda/*.cpp"
    "${GENERATORS_ROOT}/cuda/*.h"
    "${GENERATORS_ROOT}/cuda/*.cu"
    "${GENERATORS_ROOT}/cuda/*.cuh"
  )

  add_compile_definitions(USE_CUDA=1)
  include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
elseif(USE_CUDA)
  # USE_CUDA is true but cmake could not find the cuda compiler
  message(FATAL_ERROR "USE_CUDA is ON but no cuda compiler was found.")
else()
  file(GLOB generator_cuda_srcs "${GENERATORS_ROOT}/*_cuda*.*")
  list(REMOVE_ITEM generator_srcs ${generator_cuda_srcs})
  add_compile_definitions(USE_CUDA=0)
endif()

if(USE_CUDA AND NOT EXISTS "${ORT_LIB_DIR}/${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
  message(FATAL_ERROR "Expected the ONNX Runtime providers cuda library to be found at ${ORT_LIB_DIR}/${ONNXRUNTIME_PROVIDERS_CUDA_LIB}. Actual: Not found.")
endif()
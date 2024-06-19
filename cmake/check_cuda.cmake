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

  if(WIN32 AND NOT CMAKE_CUDA_FLAGS_INIT)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} /DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100 /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 /DNTDDI_VERSION=0x0A000000 
              -Xcompiler=\"/MP /guard:cf /Qspectre\"")
  endif()

  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=sm_30") # K series
  endif()
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12)
    # 37, 50 still work in CUDA 11 but are marked deprecated and will be removed in future CUDA version.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_37,code=sm_37") # K80
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52") # M60
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75") # T series
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80") # A series
  endif()
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_90,code=sm_90") # H series
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2803 --expt-relaxed-constexpr")

  file(GLOB generator_cuda_srcs CONFIGURE_DEPENDS
    "${GENERATORS_ROOT}/*.cu"
    "${GENERATORS_ROOT}/*.cuh"
    "${MODELS_ROOT}/*.cu"
    "${MODELS_ROOT}/*.cuh"
  )
  file(GLOB test_cuda_srcs CONFIGURE_DEPENDS
    "${TESTS_ROOT}/*.cu"
    "${TESTS_ROOT}/*.cuh"
  )
  list(APPEND test_srcs ${test_cuda_srcs})
  list(APPEND generator_srcs ${generator_cuda_srcs})
  list(APPEND onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
  add_compile_definitions(USE_CUDA=1)
  include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
elseif(USE_CUDA)
  # USE_CUDA is true but cmake could not find the cuda compiler
  message(FATAL_ERROR "USE_CUDA is ON but no cuda compiler was found.")
else()
  file(GLOB generator_cuda_srcs "${GENERATORS_ROOT}/*_cuda*.*")
  list(REMOVE_ITEM generator_srcs ${generator_cuda_srcs})
endif()

if(USE_CUDA AND NOT EXISTS "${ORT_LIB_DIR}/${ONNXRUNTIME_PROVIDERS_CUDA_LIB}")
  message(FATAL_ERROR "Expected the ONNX Runtime providers cuda library to be found at ${ORT_LIB_DIR}/${ONNXRUNTIME_PROVIDERS_CUDA_LIB}. Actual: Not found.")
endif()
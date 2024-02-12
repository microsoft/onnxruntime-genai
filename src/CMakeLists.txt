cmake_minimum_required(VERSION 3.26)

file(GLOB generator_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/models/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/models/*.cpp"
)
file(GLOB generator_cuda_exclusion_srcs "${CMAKE_CURRENT_SOURCE_DIR}/*_cuda*.*")
list(REMOVE_ITEM generator_srcs ${generator_cuda_exclusion_srcs})

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  # Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
  cmake_policy(SET CMP0104 OLD)
  enable_language(CUDA)
  message(STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  # set(CUDA_PROPAGATE_HOST_FLAGS ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=2803 --expt-relaxed-constexpr")
  file(GLOB generator_cuda_srcs CONFIGURE_DEPENDS
    "${GENERATORS_ROOT}/*.cu"
    "${GENERATORS_ROOT}/*.cuh"
    "${MODELS_ROOT}/*.cu"
    "${MODELS_ROOT}/*.cuh"
  )
  list(APPEND generator_srcs ${generator_cuda_srcs})
  list(APPEND generator_srcs ${generator_cuda_exclusion_srcs})
  add_compile_definitions(USE_CUDA=1)
  include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
elseif(USE_CUDA)
  message(STATUS "USE_CUDA is ON but no cuda compiler was found, disabling USE_CUDA.")
  set(USE_CUDA OFF)
endif()
add_library(onnxruntime-genai SHARED ${generator_srcs})
add_library(onnxruntime-genai-static STATIC ${generator_srcs})
target_include_directories(onnxruntime-genai PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai-static PRIVATE ${ORT_HEADER_DIR})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(onnxruntime-genai-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  set_target_properties(onnxruntime-genai PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(onnxruntime-genai PRIVATE cublasLt cublas cudnn curand cufft cudart)
  #  onnxruntime-genai-static is statically linked under Windows
  if(NOT WIN32)
    set_target_properties(onnxruntime-genai-static PROPERTIES LINKER_LANGUAGE CUDA)
    target_link_libraries(onnxruntime-genai-static PRIVATE cublasLt cublas cudnn curand cufft cudart)
  endif()
endif()

if(USE_TOKENIZER)
  add_compile_definitions(USE_TOKENIZER=1)
  target_include_directories(onnxruntime-genai PRIVATE ${TOKENIZER_ROOT})
  target_include_directories(onnxruntime-genai-static PRIVATE ${TOKENIZER_ROOT})
  target_link_libraries(onnxruntime-genai PRIVATE tokenizer)
  target_link_libraries(onnxruntime-genai-static PRIVATE tokenizer)
endif()
# Copy the onnxruntime binaries into the build folder so it's found on launch
file(GLOB onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_FILES}")
foreach(DLL_FILE ${onnxruntime_libs})
  add_custom_command(
    TARGET onnxruntime-genai POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL_FILE} $<TARGET_FILE_DIR:onnxruntime-genai>
  )
endforeach()

# Have visual studio put all files into one single folder vs the default split of header files into a separate folder
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${generator_srcs})
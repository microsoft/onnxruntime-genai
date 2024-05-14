# usage:
# mkdir -p build 
# cd build
# cmake -S ../ -B ./ -DCMAKE_CUDA_ARCHITECTURES=86 # run 'nvidia-smi  --query-gpu=compute_cap --format=csv,noheader,nounits' to get cuda_arch
# cmake --build .
cmake_minimum_required(VERSION 3.25)
project(genai_custom_ops VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(CMakeDependentOption)

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA related
find_package(CUDAToolkit)
enable_language(CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_STANDARD 17)
cmake_dependent_option(OCOS_USE_FLASH_ATTENTION "Build flash attention kernel for scaled dot product attention" ON "NOT WIN32" OFF)
option(OCOS_USE_MEMORY_EFFICIENT_ATTENTION "Build memory efficient attention kernel for scaled dot product attention" ON)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.6)
  message(STATUS "Turn off flash attention and memory efficient attention since CUDA compiler version < 11.6")
  set(OCOS_USE_FLASH_ATTENTION OFF)
  set(OCOS_USE_MEMORY_EFFICIENT_ATTENTION OFF)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror default-stream-launch")
endif()

if(NOT WIN32)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
endif()

# Options passed to cudafe
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=bad_friend_decl\"")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\"")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=expr_has_no_effect\"")

add_compile_definitions(USE_CUDA)

if (OCOS_USE_FLASH_ATTENTION)
  message(STATUS "Enable flash attention")
  add_compile_definitions(OCOS_USE_FLASH_ATTENTION)
endif()
if (OCOS_USE_MEMORY_EFFICIENT_ATTENTION)
  message(STATUS "Enable memory efficient attention")
  add_compile_definitions(OCOS_USE_MEMORY_EFFICIENT_ATTENTION)
endif()

#file(GLOB CUSTOMOPS "")
add_library(genai_custom_ops SHARED fast_gelu2.h fast_gelu_impl2.cuh fast_gelu_impl2.cu genai_ops.cc genai_ops_registry.cc paged_attention_impl.cu)

target_include_directories(genai_custom_ops PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(genai_custom_ops PUBLIC cudart cublas cufft)

target_include_directories(genai_custom_ops PUBLIC 
    "~/ort_pkg_19/include"
    "../onnxruntime-extensions/base"
    "../onnxruntime-extensions/include"
    "../onnxruntime-extensions/include/custom_op"
    "../build/_deps/gsl-src/include"
    "../onnxruntime-extensions/operators/cuda")

target_link_libraries(genai_custom_ops PUBLIC "~/ort_pkg_19/lib/libonnxruntime.so")
# usage:
# mkdir -p build 
# cd build
# cmake -S ../ -B ./ -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Debug # run 'nvidia-smi  --query-gpu=compute_cap --format=csv,noheader,nounits' to get cuda_arch
# cmake --build . --config Debug --parallel 4
cmake_minimum_required(VERSION 3.25)
project(genai_custom_ops_test VERSION 1.0)

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
add_compile_definitions(OCOS_USE_FLASH_ATTENTION)
add_compile_definitions(OCOS_USE_MEMORY_EFFICIENT_ATTENTION)

file(GLOB TARGET_SRC ../../src/custom_ops/paged_attention_impl.cu ../../src/custom_ops/genai_ops2.cc genai_custom_ops_registry.cc)
file(GLOB_RECURSE TARGET_SRC_ATTENTION_LIB "../../build/Linux/Debug/_deps/onnxruntime_extensions-src/operators/cuda/attention_lib/*.*")
list(APPEND TARGET_SRC ${TARGET_SRC_ATTENTION_LIB})
add_library(genai_custom_ops_test SHARED ${TARGET_SRC})

target_include_directories(genai_custom_ops_test PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(genai_custom_ops_test PUBLIC cudart cublas cufft)

target_include_directories(genai_custom_ops_test PUBLIC 
    "~/ort_pkg_19/include"
    "../../build/Linux/Debug/_deps/onnxruntime_extensions-src/base"
    "../../build/Linux/Debug/_deps/onnxruntime_extensions-src/include"
    "../../build/Linux/Debug/_deps/onnxruntime_extensions-src/include/custom_op"
    "../../build/Linux/Debug/_deps/onnxruntime_extensions-src/operators/cuda"
    "../../build/Linux/Debug/_deps/cutlass-src/include"
    "../../build/Linux/Debug/_deps/cutlass-src/examples")

target_link_libraries(genai_custom_ops_test PUBLIC "~/ort_pkg_19/lib/libonnxruntime.so")
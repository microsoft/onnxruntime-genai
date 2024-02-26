enable_testing()
set(TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
file(GLOB test_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# google unit test
add_executable(unit_tests main.cpp c_api.cpp tests.cpp sampling_benchmark.cpp sampling_tests.cpp)
target_include_directories(unit_tests PRIVATE
  ${ORT_HEADER_DIR}
  ${CMAKE_SOURCE_DIR}/src
)

target_link_directories(unit_tests PRIVATE ${ORT_LIB_DIR})
target_link_libraries(unit_tests PRIVATE
  onnxruntime-genai-static
  ${ONNXRUNTIME_LIB}
  GTest::gtest_main
)
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  file(GLOB cuda_test_srcs CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
  )
  set_target_properties(unit_tests PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(unit_tests PRIVATE cublasLt cublas curand cufft cudart)
  target_sources(unit_tests PRIVATE ${cuda_test_srcs})
endif()

if(NO_TOKENIZER)
  add_compile_definitions(NO_TOKENIZER=1)
else()
  target_include_directories(unit_tests PRIVATE ${TOKENIZER_ROOT})
  target_link_libraries(unit_tests PRIVATE tokenizer)
endif()

source_group("Sources" FILES ${test_srcs})
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT unit_tests)
#set_property(TARGET unit_tests PROPERTY TARGET_RUNTIME_LIBRARY_DIRS ${ORT_LIB_DIR})
include(GoogleTest)

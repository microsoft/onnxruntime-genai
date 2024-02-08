set(TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
file(GLOB test_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
add_executable(unit_tests ${test_srcs})
target_include_directories(unit_tests PRIVATE ${ORT_HEADER_DIR})
target_link_directories(unit_tests PRIVATE ${ORT_LIB_DIR})
message(ONNXRUNTIME_LIB: ${ONNXRUNTIME_LIB})
target_link_libraries(unit_tests PRIVATE onnxruntime-genai-static ${ONNXRUNTIME_LIB})

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  set_target_properties(unit_tests PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(unit_tests PRIVATE cublasLt cublas cudnn curand cufft cudart)
endif()

if(USE_TOKENIZER)
  target_include_directories(unit_tests PRIVATE ${TOKENIZER_ROOT})
endif()

enable_testing()

target_include_directories(
  unit_tests
  PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(
  unit_tests
  PRIVATE
  GTest::gtest_main
)
source_group("Sources" FILES ${test_srcs})
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT unit_tests)

#include(GoogleTest)
#gtest_discover_tests(unit_tests)

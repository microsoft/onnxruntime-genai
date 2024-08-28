include(${CMAKE_SOURCE_DIR}/cmake/cxx_standard.cmake)

set(TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
file(GLOB test_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# google unit test
add_executable(unit_tests
  main.cpp
  c_api_tests.cpp
  model_tests.cpp
  sampling_tests.cpp
  sampling_benchmark.cpp
  )

add_executable(test_engine test_engine.cpp) 
  
target_include_directories(unit_tests PRIVATE
  ${ORT_HEADER_DIR}
  ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(test_engine PRIVATE
  ${ORT_HEADER_DIR}
  ${CMAKE_SOURCE_DIR}/src
)

target_link_directories(unit_tests PRIVATE ${ORT_LIB_DIR})
target_link_directories(test_engine PRIVATE ${ORT_LIB_DIR})
target_link_libraries(unit_tests PRIVATE
  onnxruntime-genai-static
  ${ONNXRUNTIME_LIB}
  GTest::gtest_main
)
target_link_libraries(test_engine PRIVATE
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
  set_target_properties(test_engine PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(test_engine PRIVATE cublasLt cublas curand cufft cudart)
  target_sources(test_engine PRIVATE ${cuda_test_srcs})
endif()

set(TEST_MODEL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_models/")
set(TEST_MODEL_DES_DIR "$<TARGET_FILE_DIR:unit_tests>/test_models/")
add_custom_command(TARGET unit_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_MODEL_SRC_DIR} ${TEST_MODEL_DES_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${onnxruntime_libs} $<TARGET_FILE_DIR:unit_tests>
)

add_compile_definitions(MODEL_PATH="${TEST_MODEL_DES_DIR}")
set_target_properties(unit_tests PROPERTIES FOLDER "Tests")
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${test_srcs})
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT unit_tests)

include(GoogleTest)



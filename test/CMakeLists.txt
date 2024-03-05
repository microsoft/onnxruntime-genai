enable_testing()

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 10)
  add_compile_definitions(USE_CXX17=1)
  message("TEST IS USE C++17")
  set(CMAKE_CXX_STANDARD 17)
elseif(USE_CUDA AND CMAKE_CUDA_COMPILER AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12)
  add_compile_definitions(USE_CXX17=1)
  message(" TEST IS USE C++17 Because of CUDA Version is less than 12")
  set(CMAKE_CXX_STANDARD 17)
else()
  message("TEST IS USE C++20")
  set(CMAKE_CXX_STANDARD 20)
endif()

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

if(NOT WIN32)
  target_link_libraries(unit_tests PRIVATE stdc++fs)
endif()

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
file(GLOB onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_FILES}")
set(TEST_MODEL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_models/")
set(TEST_MODEL_DES_DIR "$<TARGET_FILE_DIR:unit_tests>/test_models/")
add_custom_command(TARGET unit_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_MODEL_SRC_DIR} ${TEST_MODEL_DES_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy ${onnxruntime_libs} $<TARGET_FILE_DIR:unit_tests>
)

add_compile_definitions(MODEL_PATH="${TEST_MODEL_DES_DIR}")
source_group("Sources" FILES ${test_srcs})
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT unit_tests)
include(GoogleTest)

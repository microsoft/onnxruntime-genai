# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if (IOS)
  find_package(XCTest REQUIRED)
endif()

set(TEST_SRC_DIR ${CMAKE_SOURCE_DIR}/test)
set(TEST_INC_DIR ${ORT_HEADER_DIR} ${CMAKE_SOURCE_DIR}/src)

include(${CMAKE_SOURCE_DIR}/cmake/cxx_standard.cmake)

set(TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
file(GLOB test_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

set(_UT_SOURCES
  main.cpp
  c_api_tests.cpp
  model_tests.cpp
  sampling_tests.cpp
  sampling_benchmark.cpp
)

if (IOS)
  add_executable(unit_tests ${_UT_SOURCES} ${TEST_SRC_DIR}/xctest/ortgenaitestmain.mm)
else()
  # google unit test
  add_executable(unit_tests ${_UT_SOURCES})
endif()

target_include_directories(unit_tests PRIVATE ${TEST_INC_DIR})

target_link_directories(unit_tests PRIVATE ${ORT_LIB_DIR})
target_link_libraries(unit_tests PRIVATE
  onnxruntime-genai-static
  GTest::gtest_main
)

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
target_link_libraries(unit_tests PRIVATE ${ONNXRUNTIME_LIB})
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

set(TEST_MODEL_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_models/")
set(TEST_MODEL_DES_DIR "$<TARGET_FILE_DIR:unit_tests>/test_models/")
add_custom_command(TARGET unit_tests POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TEST_MODEL_SRC_DIR} ${TEST_MODEL_DES_DIR}
)

if (NOT IOS)
  # On iOS static xcframework is used
  add_custom_command(TARGET unit_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${onnxruntime_libs} $<TARGET_FILE_DIR:unit_tests>
  )
endif()

add_compile_definitions(MODEL_PATH="${TEST_MODEL_DES_DIR}")
set_target_properties(unit_tests PROPERTIES FOLDER "Tests")
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${test_srcs})
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT unit_tests)

if (IOS)
  set_target_properties(unit_tests PROPERTIES FOLDER "ONNXRuntimeGenAITest"
  MACOSX_BUNDLE_BUNDLE_NAME unit_tests
  MACOSX_BUNDLE_GUI_IDENTIFIER com.onnxruntimegenai.utest
  MACOSX_BUNDLE_LONG_VERSION_STRING ${VERSION_STR}
  MACOSX_BUNDLE_BUNDLE_VERSION ${VERSION_STR}
  MACOSX_BUNDLE_SHORT_VERSION_STRING ${VERSION_STR}
  XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES "YES"
  XCODE_ATTRIBUTE_ENABLE_BITCODE "NO"
  XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")

  xctest_add_bundle(unit_tests_xc unit_tests
  ${TEST_SRC_DIR}/xctest/ortgenaixctest.m
  ${TEST_SRC_DIR}/xctest/xcgtest.mm
  ${_UT_SOURCES})

  get_target_property(srcs unit_tests_xc SOURCES)
  set(objective_c_cc_srcs ${srcs})
  list(FILTER objective_c_cc_srcs INCLUDE REGEX "\\.mm?$")
  set_property(SOURCE ${objective_c_cc_srcs} APPEND PROPERTY COMPILE_OPTIONS "-fobjc-arc")

  target_include_directories(unit_tests_xc PRIVATE ${TEST_INC_DIR})
  target_link_directories(unit_tests_xc PRIVATE ${ORT_LIB_DIR})
  target_link_libraries(unit_tests_xc PRIVATE onnxruntime-genai-static GTest::gtest)
  target_link_libraries(unit_tests_xc PRIVATE ${ONNXRUNTIME_LIB})
  set_target_properties(unit_tests_xc PROPERTIES FOLDER "ONNXRuntimeGenAIXCTest"
  MACOSX_BUNDLE_BUNDLE_NAME unit_tests_xc
  MACOSX_BUNDLE_GUI_IDENTIFIER com.onnxruntime.utest.unit_tests_xc
  MACOSX_BUNDLE_LONG_VERSION_STRING ${VERSION_STR}
  MACOSX_BUNDLE_BUNDLE_VERSION ${VERSION_STR}
  MACOSX_BUNDLE_SHORT_VERSION_STRING ${VERSION_STR}
  XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")

  xctest_add_test(xctest.unit_tests unit_tests_xc)
endif()

include(GoogleTest)

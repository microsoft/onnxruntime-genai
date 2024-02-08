
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

include(GoogleTest)
gtest_discover_tests(unit_tests)

# Windows requires dlls to be in the same directory as the executable
if(WIN32)
  add_custom_command(
    TARGET onnxruntime-genai POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${onnxruntime_libs} "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
  )
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

include(GoogleTest)
gtest_discover_tests(unit_tests)

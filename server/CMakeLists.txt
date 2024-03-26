include(${CMAKE_SOURCE_DIR}/cmake/cxx_standard.cmake)

add_executable(server
  server.cpp
  generator.cpp
)

target_include_directories(server PRIVATE
  ${CMAKE_SOURCE_DIR}/src/
  ${nlohmann_json_SOURCE_DIR}/single_include/nlohmann
  ${httplib_SOURCE_DIR}/
)

target_link_directories(server PRIVATE ${ORT_LIB_DIR})
target_link_libraries(server PRIVATE
  onnxruntime-genai-static
  nlohmann_json
  ${ONNXRUNTIME_LIB}
)

add_executable(model_benchmark
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(model_benchmark PRIVATE
  ${CMAKE_SOURCE_DIR}/src  # directory containing the ort_genai headers
)

target_link_libraries(model_benchmark PRIVATE onnxruntime-genai-static ${ONNXRUNTIME_LIB})

target_link_directories(model_benchmark PRIVATE ${ORT_LIB_DIR})

add_custom_command(TARGET model_benchmark POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${onnxruntime_libs} $<TARGET_FILE_DIR:model_benchmark>
)

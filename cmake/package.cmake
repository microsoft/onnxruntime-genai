set_target_properties(
  onnxruntime-genai PROPERTIES PUBLIC_HEADER
  "${CMAKE_SOURCE_DIR}/src/ort_genai_c.h;${CMAKE_SOURCE_DIR}/src/ort_genai.h"
)
install(TARGETS
  onnxruntime-genai onnxruntime-genai-static
  ARCHIVE
  LIBRARY
  PUBLIC_HEADER
)
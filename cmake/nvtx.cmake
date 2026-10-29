include_guard(GLOBAL)

if(ENABLE_NVTX)
  find_path(ORTGENAI_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
    HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ENV NVTX_ROOT ENV CUDA_PATH
    PATH_SUFFIXES include
    DOC "Directory containing the NVTX3 headers (nvtx3/nvToolsExt.h)")
  if(NOT ORTGENAI_NVTX_INCLUDE_DIR OR
     NOT EXISTS "${ORTGENAI_NVTX_INCLUDE_DIR}/nvtx3/nvToolsExt.h")
    message(FATAL_ERROR
      "ENABLE_NVTX requires NVTX3 headers. Set ORTGENAI_NVTX_INCLUDE_DIR to the directory containing nvtx3/nvToolsExt.h.")
  endif()

  add_library(onnxruntime-genai-nvtx INTERFACE)
  target_include_directories(onnxruntime-genai-nvtx SYSTEM INTERFACE "${ORTGENAI_NVTX_INCLUDE_DIR}")
  target_compile_definitions(onnxruntime-genai-nvtx INTERFACE ORTGENAI_ENABLE_NVTX)
  target_link_libraries(onnxruntime-genai-nvtx INTERFACE ${CMAKE_DL_LIBS})
endif()
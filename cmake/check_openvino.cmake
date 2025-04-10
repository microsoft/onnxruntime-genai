set(OPENVINO_EP_FOUND 0)

# First check if ONNXRUNTIME_PROVIDERS_OPENVINO_LIB var has been set.
# (global_variables.cmake only sets it for supported platforms)
if(DEFINED ONNXRUNTIME_PROVIDERS_OPENVINO_LIB)
  # Check for existence of OpenVINO EP library. This will determine
  #  whether onnxruntime has been compiled with OpenVINO EP support.
  if(EXISTS "${ORT_LIB_DIR}/${ONNXRUNTIME_PROVIDERS_OPENVINO_LIB}")
    set(OPENVINO_EP_FOUND 1)
  endif()
endif()

add_compile_definitions(USE_OPENVINO=${OPENVINO_EP_FOUND})

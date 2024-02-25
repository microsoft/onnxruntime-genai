file(GLOB python_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
pybind11_add_module(python ${python_srcs})
target_include_directories(python PRIVATE ${ORT_HEADER_DIR})
target_link_directories(python PRIVATE ${ORT_LIB_DIR})
target_link_libraries(python PRIVATE onnxruntime-genai-static ${ONNXRUNTIME_LIB})
set_target_properties(python PROPERTIES OUTPUT_NAME "onnxruntime_genai")

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  set_target_properties(python PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(python PRIVATE cublasLt cublas cudnn curand cufft cudart)
endif()
source_group("Sources" FILES ${python_srcs})

if(BUILD_WHEEL)
  set(WHEEL_FILES_DIR "${CMAKE_BINARY_DIR}/wheel")
  message("Setting up wheel files in : ${WHEEL_FILES_DIR}")
  set(WHEEL_TARGET_NAME "${WHEEL_FILES_DIR}/onnxruntime_genai")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${WHEEL_FILES_DIR}/setup.py @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in ${WHEEL_TARGET_NAME}/__init__.py @ONLY)

  # Copy over any additional python files
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/py/" DESTINATION ${WHEEL_TARGET_NAME}/)

  file(GLOB onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_FILES}")
  add_custom_command(TARGET python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:onnxruntime-genai> ${onnxruntime_libs} $<TARGET_FILE:python>
    ${WHEEL_TARGET_NAME}
    COMMENT "Copying files to wheel directory: ${WHEEL_TARGET_NAME}"
  )
  add_custom_target(PyPackageBuild ALL
    COMMAND "${PYTHON_EXECUTABLE}" -m pip wheel .
    WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
    COMMENT "Building wheel on ${WHEEL_FILES_DIR}"
  )
  add_dependencies(PyPackageBuild python)
endif()
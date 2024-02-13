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
  message("Setting up wheel files in : ${WHEEL_FILES_DIR}")
  set(TARGET_NAME "onnxruntime_genai")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${WHEEL_FILES_DIR}/setup.py @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/py/__init__.py.in ${WHEEL_FILES_DIR}/${TARGET_NAME}/__init__.py @ONLY)

  # Copy over any additional python files
  file(GLOB pyfiles "${CMAKE_CURRENT_SOURCE_DIR}/py/*.py")
  foreach(filename ${pyfiles})
    get_filename_component(target "${filename}" NAME)
    message(STATUS "Copying ${filename} to ${target}")
    configure_file("${filename}" "${WHEEL_FILES_DIR}/${TARGET_NAME}" COPYONLY)
  endforeach(filename)
  add_custom_target(PyPackageBuild ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onnxruntime-genai> ${WHEEL_FILES_DIR}/${TARGET_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy ${onnxruntime_libs} ${WHEEL_FILES_DIR}/${TARGET_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:python> ${WHEEL_FILES_DIR}/${TARGET_NAME}
    COMMAND "${PYTHON_EXECUTABLE}" -m pip wheel .
    WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
    COMMENT "Building wheel"
  )
  add_dependencies(PyPackageBuild python)
endif()
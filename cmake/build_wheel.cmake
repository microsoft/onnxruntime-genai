  set(WHEEL_FILES_DIR "${CMAKE_BINARY_DIR}/wheel")
  message("Setting up wheel files in : ${WHEEL_FILES_DIR}")
  set(TARGET_NAME "onnxruntime_genai")
  configure_file(${PYTHON_ROOT}/setup.py.in ${WHEEL_FILES_DIR}/setup.py @ONLY)
  configure_file(${PYTHON_ROOT}/py/__init__.py.in ${WHEEL_FILES_DIR}/${TARGET_NAME}/__init__.py @ONLY)
  file(GLOB onnxruntime_libs "${CMAKE_SOURCE_DIR}/ort/lib/${ONNXRUNTIME_FILES}")
  foreach(DLL_FILE ${onnxruntime_libs})
    add_custom_command(
      TARGET onnxruntime-genai-static POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${DLL_FILE} ${WHEEL_FILES_DIR}/${TARGET_NAME}/${DLL_FILE_NAME}
    )
  endforeach()


  # Copy over any additional python files
  file(GLOB pyfiles "${PYTHON_ROOT}/py/*.py")
  foreach(filename ${pyfiles})
    get_filename_component(target "${filename}" NAME)
    message(STATUS "Copying ${filename} to ${target}")
    configure_file("${filename}" "${WHEEL_FILES_DIR}/${TARGET_NAME}" COPYONLY)
  endforeach(filename)

  add_custom_target(BuildWheel ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:python> ${WHEEL_FILES_DIR}/${TARGET_NAME}
    COMMAND "${PYTHON_EXECUTABLE}" -m pip wheel .
    WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
    COMMENT "Building wheel"
  )
  add_dependencies(BuildWheel python)
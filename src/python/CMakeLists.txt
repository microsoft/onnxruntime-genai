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
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in ${WHEEL_FILES_DIR}/${TARGET_NAME}/__init__.py @ONLY)

  # Copy over any additional python files
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/py/" DESTINATION ${WHEEL_FILES_DIR}/${TARGET_NAME}/)

  # file(GLOB_RECURSE pyfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
  # foreach(filename ${pyfiles})
  #   get_filename_component(target "${filename}" NAME)
  #   message(STATUS "Copying ${filename} to ${target}")
  #   configure_file("${filename}" "${WHEEL_FILES_DIR}/${TARGET_NAME}" COPYONLY)
  # endforeach(filename)

  # Copy over all python files
  # file(GLOB_RECURSE pyfiles "${CMAKE_CURRENT_SOURCE_DIR}/*.py")
  # foreach(filename ${pyfiles})
  #   # Get current source path without filename
  #   get_filename_component(source_path_to_file "${filename}" DIRECTORY)

  #   # Get relative directory within source path without filename
  #   string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" dir_to_create source_path_to_file)

  #   # Create directory if it doesn't exist
  #   file(MAKE_DIRECTORY dir_to_create)

  #   # Get filename
  #   get_filename_component(target "${filename}" NAME)

  #   # Copy file to new directory
  #   message(STATUS "Copying ${filename} to ${WHEEL_FILES_DIR}/${TARGET_NAME}/${dir_to_create}")
  #   configure_file("${filename}" "${WHEEL_FILES_DIR}/${TARGET_NAME}/${dir_to_create}" COPYONLY)
  # endforeach(filename)

  # Copy over model builder python files
  # file(GLOB_RECURSE model_builder_files "${CMAKE_CURRENT_SOURCE_DIR}/models/*.py")
  # foreach(filename ${model_builder_files})
  #   get_filename_component(target "${filename}" NAME)
  #   message(STATUS "Current source dir is ${CMAKE_CURRENT_SOURCE_DIR}")
  #   message(STATUS "Wheel files dir is ${WHEEL_FILES_DIR}")
  #   message(STATUS "Target name is ${TARGET_NAME}")
  #   message(STATUS "Copying ${filename} to ${target}")
  #   configure_file("${filename}" "${WHEEL_FILES_DIR}/${TARGET_NAME}" COPYONLY)
  # endforeach(filename)
  # add_custom_target(copy-runtime-files ALL
  #   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/models ${CMAKE_BINARY_DIR}/models
  #   DEPENDS ${MY_TARGET})

  # Copy 'models' directory to wheel
  # file(GLOB model_builder_files CONFIGURE_DEPENDS
  #     "${CMAKE_CURRENT_SOURCE_DIR}/src/python/models/*.py"
  # )
  # add_custom_command(
  #   TARGET python POST_BUILD
  #   COMMAND ${CMAKE_COMMAND} -E make_directory ${WHEEL_FILES_DIR}/${TARGET_NAME}/models
  #   COMMAND ${CMAKE_COMMAND} -E copy
  #       ${model_builder_files}
  #       ${WHEEL_FILES_DIR}/${TARGET_NAME}/models/
  # )

  file(GLOB onnxruntime_libs "${ORT_LIB_DIR}/${ONNXRUNTIME_FILES}")
  add_custom_target(PyPackageBuild ALL
    # COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/" "${WHEEL_FILES_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onnxruntime-genai> ${WHEEL_FILES_DIR}/${TARGET_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy ${onnxruntime_libs} ${WHEEL_FILES_DIR}/${TARGET_NAME}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:python> ${WHEEL_FILES_DIR}/${TARGET_NAME}
    COMMAND "${PYTHON_EXECUTABLE}" -m pip wheel .
    WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
    COMMENT "Building wheel"
  )
  add_dependencies(PyPackageBuild python)
endif()
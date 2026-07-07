include(${REPO_ROOT}/cmake/cxx_standard.cmake)

file(GLOB python_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
find_package(Python COMPONENTS Interpreter Development)
pybind11_add_module(python ${python_srcs})
target_include_directories(python PRIVATE ${ORT_HEADER_DIR})
target_link_directories(python PRIVATE ${ORT_LIB_DIR})
target_link_libraries(python PRIVATE onnxruntime-genai)

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
  target_link_libraries(python PRIVATE ${ONNXRUNTIME_LIB})
endif()

set_target_properties(python PROPERTIES OUTPUT_NAME "onnxruntime_genai")

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(python PRIVATE "/CETCOMPAT")
  target_compile_options(python PRIVATE "/sdl" PRIVATE "/Qspectre")
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${python_srcs})

if(BUILD_WHEEL)
  set(WHEEL_FILES_DIR "${CMAKE_BINARY_DIR}/wheel")
  message("Setting up wheel files in : ${WHEEL_FILES_DIR}")
  if(USE_TRT_RTX)
    set(TARGET_NAME "onnxruntime-genai-trt-rtx")
  elseif(USE_WINML)
    set(TARGET_NAME "onnxruntime-genai-winml")
  elseif(USE_CUDA)
    set(TARGET_NAME "onnxruntime-genai-cuda")
  elseif(USE_DML)
    set(TARGET_NAME "onnxruntime-genai-directml")
  else()
    set(TARGET_NAME "onnxruntime-genai")
  endif()
  set(PACKAGE_DIR_NAME "onnxruntime_genai")
  set(WHEEL_TARGET_NAME "${WHEEL_FILES_DIR}/${PACKAGE_DIR_NAME}")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${WHEEL_FILES_DIR}/setup.py @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in ${WHEEL_TARGET_NAME}/__init__.py @ONLY)

  # Copy over any additional python files
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/py/" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/package_description.md" DESTINATION ${WHEEL_FILES_DIR}/)
  file(COPY "${CMAKE_SOURCE_DIR}/ThirdPartyNotices.txt" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION ${WHEEL_TARGET_NAME}/)

  add_custom_command(TARGET python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${ortgenai_embed_libs} $<TARGET_FILE:python>
    ${WHEEL_TARGET_NAME}
    COMMENT "Copying files to wheel directory: ${WHEEL_TARGET_NAME}"
  )

  # Generate type stubs using pybind11_stubgen if it is available.
  # CMake's find_package cannot locate Python modules, so we probe with
  # execute_process at configure time.  The Python import name uses underscores
  # ('pybind11_stubgen') while the pip package name uses hyphens
  # ('pybind11-stubgen') – both refer to the same package.
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11_stubgen"
    RESULT_VARIABLE PYBIND11_STUBGEN_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
  )
  if(PYBIND11_STUBGEN_RESULT EQUAL 0)
    set(PYBIND11_STUB_WORK_DIR "${CMAKE_BINARY_DIR}/wheel/${PACKAGE_DIR_NAME}-type-stubs")
    add_custom_command(TARGET python POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "${PYBIND11_STUB_WORK_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${WHEEL_TARGET_NAME}" "${PYBIND11_STUB_WORK_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${ORT_LIB_DIR}" "${PYBIND11_STUB_WORK_DIR}"
      COMMAND ${Python_EXECUTABLE} -m pybind11_stubgen onnxruntime_genai --output-dir "${WHEEL_TARGET_NAME}"
      COMMAND ${Python_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/fix_pyi.py" -i "${WHEEL_TARGET_NAME}/onnxruntime_genai.pyi"
      WORKING_DIRECTORY "${PYBIND11_STUB_WORK_DIR}"
      COMMENT "Generate type stubs for onnxruntime_genai"
    )
  else()
    message(STATUS "pybind11_stubgen not found, skipping type stub generation. "
                   "To enable, run: pip install pybind11-stubgen")
  endif()
  set(auditwheel_exclude_list
    "libcublas.so.11"
    "libcublas.so.12"
    "libcublasLt.so.11"
    "libcublasLt.so.12"
    "libcudart.so.11.0"
    "libcudart.so.12"
    "libcudnn.so.8"
    "libcudnn.so.9"
    "libcufft.so.10"
    "libcufft.so.11"
    "libcurand.so.10"
    "libnvinfer.so.8"
    "libnvinfer.so.10"
    "libnvinfer_plugin.so.8"
    "libnvinfer_plugin.so.10"
    "libnvonnxparser.so.8"
    "libnvonnxparser.so.10"
  )
  set(modified_exclude_list)
  foreach(item IN LISTS auditwheel_exclude_list)
    list(APPEND modified_exclude_list "--exclude" ${item})
  endforeach()

  # On Linux we need this so the python bindings can load our other shared libraries in the same folder
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(python PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")  # For macOS
    set_target_properties(python PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
  endif()

  if(MANYLINUX)
    # auditwheel produces a wheel tagged with the platform (e.g. linux_x86_64
    # or linux_aarch64) — use a glob that matches all Linux platform tags so
    # this target works on every supported manylinux architecture.
    add_custom_target(PyPackageBuild
      COMMAND ${PYTHON_EXECUTABLE} -m pip wheel --no-deps .
      COMMAND ${CMAKE_COMMAND} -E remove ${WHEEL_TARGET_NAME}/onnxruntime_genai.cpython-*
      COMMAND auditwheel repair onnxruntime_genai*linux_*.whl -w ${WHEEL_FILES_DIR} ${modified_exclude_list}
      WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
      COMMENT "Building wheel with MANYLINUX on ${WHEEL_FILES_DIR}"
      EXCLUDE_FROM_ALL
    )
  else()
    add_custom_target(PyPackageBuild
      COMMAND ${PYTHON_EXECUTABLE} -m pip wheel --no-deps .
      WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
      COMMENT "Building wheel on ${WHEEL_FILES_DIR}"
      EXCLUDE_FROM_ALL
    )
  endif()
endif()
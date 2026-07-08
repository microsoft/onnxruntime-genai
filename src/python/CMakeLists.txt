if(NOT DEFINED REPO_ROOT)
  # ---------------------------------------------------------------------------
  # Standalone (incremental) SDK build.
  #
  # This project is configured directly (cmake -S src/python) and links against a
  # prebuilt core installed via find_package(onnxruntime-genai), instead of being
  # added as a subdirectory of the top-level core build. build.py drives this via
  # `--sdk python --prebuilt_genai_home <prefix> --ort_home <ort>`.
  # ---------------------------------------------------------------------------
  cmake_minimum_required(VERSION 3.26)
  project(onnxruntime_genai_python LANGUAGES C CXX)

  get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  set(SRC_ROOT ${REPO_ROOT}/src)

  # Read the project version the same way global_variables.cmake does in-tree so
  # the wheel (setup.py.in / __init__.py.in @VERSION_INFO@) is versioned correctly.
  file(READ "${REPO_ROOT}/VERSION_INFO" VERSION_INFO)
  string(STRIP "${VERSION_INFO}" VERSION_INFO)

  find_package(onnxruntime-genai CONFIG REQUIRED)

  # Mirror the EP flavor of the prebuilt core so the wheel is named correctly.
  set(USE_CUDA ${onnxruntime-genai_USE_CUDA})
  set(USE_TRT_RTX ${onnxruntime-genai_USE_TRT_RTX})
  set(USE_DML ${onnxruntime-genai_USE_DML})
  set(USE_WINML ${onnxruntime-genai_USE_WINML})

  if(NOT DEFINED ORT_HOME)
    message(FATAL_ERROR "ORT_HOME must be set (-DORT_HOME=...) for a standalone python SDK build.")
  endif()
  # Desktop ORT layout: headers under <ORT_HOME>/include and libs under
  # <ORT_HOME>/lib. Android/iOS use different ORT package layouts (see
  # cmake/ortlib.cmake) and are not yet supported by the standalone SDK build.
  set(ORT_HEADER_DIR ${ORT_HOME}/include)
  set(ORT_LIB_DIR ${ORT_HOME}/lib)

  include(${REPO_ROOT}/cmake/target_platform.cmake)

  if(WIN32)
    set(ONNXRUNTIME_LIB "onnxruntime.dll")
  elseif(APPLE)
    set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
  else()
    set(ONNXRUNTIME_LIB "libonnxruntime.so")
  endif()

  # pybind11 (installed via pip). Locate its CMake package through the interpreter.
  # Only the Development.Module component is needed to build the extension (like the
  # in-tree pybind11_add_module path); requesting the full Development component would
  # demand an embeddable libpython, which manylinux CPython builds do not ship.
  # build.py passes -DPython_EXECUTABLE so CMake uses the exact interpreter that will
  # build the wheel (e.g. /opt/python/cp311-cp311/... in the manylinux container),
  # which is otherwise not on CMake's default search paths.
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  # The wheel-packaging custom target uses ${PYTHON_EXECUTABLE}; in-tree this is
  # provided by the parent build, so mirror it from the modern Python module here.
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE _pybind11_cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _pybind11_result)
    if(NOT _pybind11_result EQUAL 0)
      message(FATAL_ERROR "pybind11 not found. Install it with `pip install pybind11` for standalone SDK builds.")
    endif()
    find_package(pybind11 CONFIG REQUIRED PATHS "${_pybind11_cmakedir}" NO_DEFAULT_PATH)
  endif()

  # Provide the plain `onnxruntime-genai` target name the rest of this file links against.
  if(NOT TARGET onnxruntime-genai)
    add_library(onnxruntime-genai ALIAS onnxruntime-genai::onnxruntime-genai)
  endif()

  # Gather the prebuilt core shared libraries (main + optional cuda) to embed in the wheel.
  # They were installed next to the exported CMake package (see cmake/package.cmake).
  get_filename_component(_genai_install_lib_dir "${onnxruntime-genai_DIR}/../.." ABSOLUTE)
  if(WIN32)
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.dll")
  elseif(APPLE)
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.dylib")
  else()
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.so")
  endif()
endif()

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
  file(COPY "${REPO_ROOT}/ThirdPartyNotices.txt" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${REPO_ROOT}/LICENSE" DESTINATION ${WHEEL_TARGET_NAME}/)

  add_custom_command(TARGET python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${ortgenai_embed_libs} $<TARGET_FILE:python>
    ${WHEEL_TARGET_NAME}
    COMMENT "Copying files to wheel directory: ${WHEEL_TARGET_NAME}"
  )
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
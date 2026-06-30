# This CMakeLists is dual-mode:
#  - In-tree (monorepo build): pulled in via add_subdirectory from the root
#    CMakeLists, where the core target and shared variables already exist.
#  - Standalone (incremental SDK build): configured directly against a prebuilt
#    core consumed via find_package(onnxruntime-genai). This lets the wheel be
#    built without recompiling the core.
if(NOT TARGET onnxruntime-genai AND NOT TARGET onnxruntime-genai::onnxruntime-genai)
  cmake_minimum_required(VERSION 3.26)
  project(onnxruntime_genai_python LANGUAGES C CXX)

  get_filename_component(SRC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
  get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

  # Consume the prebuilt core (imports onnxruntime-genai::onnxruntime-genai).
  find_package(onnxruntime-genai REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  set(GENAI_STANDALONE_SDK ON)

  # The version is single-sourced from the repo-root VERSION_INFO file (the same
  # file the core build reads), so the wheel version matches the core.
  file(READ "${REPO_ROOT}/VERSION_INFO" VERSION_INFO)
  string(STRIP "${VERSION_INFO}" VERSION_INFO)

  # The binding compiles against ONNX Runtime headers (onnxruntime_c_api.h).
  # The SDK build resolves ORT from the single managed source and passes its
  # location in; we don't bake ORT paths into the exported core package.
  if(NOT ORT_HEADER_DIR)
    message(FATAL_ERROR "Standalone python build requires -DORT_HEADER_DIR=<onnxruntime headers>")
  endif()
endif()

# Normalize the core target name across both modes.
if(TARGET onnxruntime-genai::onnxruntime-genai)
  set(GENAI_CORE_TARGET onnxruntime-genai::onnxruntime-genai)
else()
  set(GENAI_CORE_TARGET onnxruntime-genai)
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
target_link_libraries(python PRIVATE ${GENAI_CORE_TARGET})

# Pin the C++ standard on the target so it holds in both the in-tree build
# (inherits it from the root scope) and the standalone SDK build (where the
# directory-scope default would otherwise let pybind11 fall back to C++14).
set_target_properties(python PROPERTIES CXX_STANDARD ${CMAKE_CXX_STANDARD} CXX_STANDARD_REQUIRED ON)

# In the in-tree build the execution-provider flags come from global compile
# definitions. In the standalone SDK build they come from the exported core
# package so the wheel's reported capabilities match the prebuilt core.
if(GENAI_STANDALONE_SDK)
  target_compile_definitions(python PRIVATE ${onnxruntime-genai_EP_DEFINITIONS})

  # In-tree the root CMakeLists populates ortgenai_embed_libs with the genai
  # runtime libraries the wheel must bundle. Standalone we reconstruct it from
  # the imported core target(s); ORT itself is not bundled (the wheel declares a
  # pip dependency on onnxruntime/onnxruntime-gpu instead).
  list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai::onnxruntime-genai>")
  if(TARGET onnxruntime-genai::onnxruntime-genai-cuda)
    list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai::onnxruntime-genai-cuda>")
  endif()
endif()

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
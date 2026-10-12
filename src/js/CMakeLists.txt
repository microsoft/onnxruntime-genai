# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT DEFINED REPO_ROOT)
  cmake_minimum_required(VERSION 3.26)
  project(onnxruntime_genai_node LANGUAGES CXX)
  get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  find_package(onnxruntime-genai CONFIG REQUIRED)
  set(USE_CUDA ${onnxruntime-genai_USE_CUDA})
  set(USE_TRT_RTX ${onnxruntime-genai_USE_TRT_RTX})
  set(USE_DML ${onnxruntime-genai_USE_DML})
  set(USE_WINML ${onnxruntime-genai_USE_WINML})
  if(NOT DEFINED ORT_HOME)
    message(FATAL_ERROR "ORT_HOME must be set for a standalone JavaScript SDK build.")
  endif()
  set(ORT_LIB_DIR "${ORT_HOME}/lib")
  if(NOT TARGET onnxruntime-genai)
    add_library(onnxruntime-genai ALIAS onnxruntime-genai::onnxruntime-genai)
  endif()
  get_filename_component(_genai_install_lib_dir "${onnxruntime-genai_DIR}/../.." ABSOLUTE)
  if(WIN32)
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.dll")
  elseif(APPLE)
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.dylib")
  else()
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.so*")
  endif()
else()
  set(ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai>")
  if(TARGET onnxruntime-genai-cuda)
    list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai-cuda>")
  endif()
endif()

find_program(NODE_EXECUTABLE node REQUIRED)
execute_process(
  COMMAND ${NODE_EXECUTABLE} -p "require('node-addon-api').include"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE NODE_ADDON_API_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\"" "" NODE_ADDON_API_INCLUDE "${NODE_ADDON_API_INCLUDE}")
execute_process(
  COMMAND ${NODE_EXECUTABLE} -p "require('node-api-headers').include_dir"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE NODE_API_INCLUDE
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)

add_library(onnxruntime_genai_node MODULE addon.cpp)
set_target_properties(onnxruntime_genai_node PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  PREFIX ""
  SUFFIX ".node"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/$<CONFIG>"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/$<CONFIG>")
target_include_directories(onnxruntime_genai_node PRIVATE
  "${NODE_ADDON_API_INCLUDE}"
  "${NODE_API_INCLUDE}"
  "${REPO_ROOT}/src")
target_compile_definitions(onnxruntime_genai_node PRIVATE NAPI_VERSION=8)
target_link_libraries(onnxruntime_genai_node PRIVATE onnxruntime-genai)

if(WIN32)
  file(GLOB ort_embed_libs CONFIGURE_DEPENDS "${ORT_LIB_DIR}/*.dll")
elseif(APPLE)
  file(GLOB ort_embed_libs CONFIGURE_DEPENDS "${ORT_LIB_DIR}/*.dylib")
else()
  file(GLOB ort_embed_libs CONFIGURE_DEPENDS "${ORT_LIB_DIR}/*.so" "${ORT_LIB_DIR}/*.so.*")
endif()
if(NOT ortgenai_embed_libs)
  message(FATAL_ERROR "No onnxruntime-genai shared library was found to package.")
endif()
if(NOT ort_embed_libs)
  message(FATAL_ERROR "No ONNX Runtime shared libraries were found in ${ORT_LIB_DIR}.")
endif()

set(JS_NATIVE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build/$<CONFIG>")
foreach(LIB_FILE IN LISTS ortgenai_embed_libs ort_embed_libs)
  add_custom_command(TARGET onnxruntime_genai_node POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${LIB_FILE}" "${JS_NATIVE_OUTPUT_DIR}/"
    VERBATIM)
endforeach()

if(APPLE)
  set_target_properties(onnxruntime_genai_node PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
  set(_js_origin "@loader_path")
elseif(UNIX)
  set(_js_origin "$ORIGIN")
endif()
if(APPLE OR UNIX)
  set_target_properties(onnxruntime_genai_node PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${_js_origin}")
endif()

if(MSVC)
  get_filename_component(NODE_BIN_DIR "${NODE_EXECUTABLE}" DIRECTORY)
  find_library(NODE_LIBRARY node HINTS
    "$ENV{ProgramFiles}/nodejs"
    "${NODE_BIN_DIR}"
    "${NODE_BIN_DIR}/../lib")
  if(NOT NODE_LIBRARY)
    message(FATAL_ERROR "node.lib is required to build the Node addon on Windows; set NODE_LIBRARY.")
  endif()
  target_link_libraries(onnxruntime_genai_node PRIVATE "${NODE_LIBRARY}")
  include("${REPO_ROOT}/cmake/msvc_security.cmake")
  target_compile_options(onnxruntime_genai_node PRIVATE /sdl)
endif()

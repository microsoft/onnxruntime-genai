cmake_minimum_required(VERSION 3.25)

project(ortgenaiapp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Download and make available nlohmann/json
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.12.0     # Or update to latest release
  SYSTEM
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nlohmann_json)

# Download and make available CLI11
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.6.1      # Or update to latest release
  SYSTEM
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(CLI11)

option(USE_CXX "Invoke the C++ example" ON)
option(MODEL_CHAT "Build the Model Chat example" OFF)
option(MODEL_QA "Build the Model Q&A example" OFF)
option(MODEL_MM "Build the Model Multimodal example" OFF)
option(WHISPER "Build the Whisper example" OFF)
option(NEMOTRON_SPEECH "Build the Nemotron Speech Streaming example" OFF)

if(USE_CXX)
  add_compile_definitions(USE_CXX)
endif()

macro(genai_find_dependency id component)
  find_package(${component} QUIET HINTS ${${id}_HOME})
  if(NOT ${component}_FOUND)
    if(CMAKE_SYSTEM_NAME MATCHES "AIX")
      set(${id}_LIB_FILE lib${component}.a)
    else()
      set(${id}_LIB_FILE ${CMAKE_SHARED_LIBRARY_PREFIX}${component}${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    if(NOT ${id}_INCLUDE_DIR OR NOT ${id}_LIB_DIR)
      message(FATAL_ERROR "missing required ${id}_INCLUDE_DIR and ${id}_LIB_DIR")
    endif()
    message(STATUS "${id}_INCLUDE_DIR: ${${id}_INCLUDE_DIR}")
    message(STATUS "${id}_LIB_DIR: ${${id}_LIB_DIR}")
    add_library(onnxruntime::${component} SHARED IMPORTED)
    set_target_properties(onnxruntime::${component} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${${id}_INCLUDE_DIR})
    set_target_properties(onnxruntime::${component} PROPERTIES
            IMPORTED_LOCATION "${${id}_LIB_DIR}/${${id}_LIB_FILE}")
    if(WIN32)
        set_target_properties(onnxruntime::${component} PROPERTIES
            IMPORTED_IMPLIB "${${id}_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${component}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    endif()
  endif()
endmacro()

genai_find_dependency(ORT onnxruntime)
genai_find_dependency(OGA onnxruntime-genai)

macro(genai_add_example target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE onnxruntime::onnxruntime-genai
          onnxruntime::onnxruntime nlohmann_json::nlohmann_json CLI11::CLI11)
  if(UNIX)
    set_target_properties(${target} PROPERTIES BUILD_RPATH
      "$<TARGET_FILE_DIR:onnxruntime::onnxruntime>;$<TARGET_FILE_DIR:onnxruntime::onnxruntime-genai>")
  elseif(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}> COMMAND_EXPAND_LISTS)
  endif()
endmacro()

set(EXAMPLES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

if(MODEL_CHAT)
  genai_add_example(model_chat ${EXAMPLES_SOURCE_DIR}/model_chat.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

if(MODEL_QA)
  genai_add_example(model_qa ${EXAMPLES_SOURCE_DIR}/model_qa.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

if(MODEL_MM)
  genai_add_example(model_mm ${EXAMPLES_SOURCE_DIR}/model_mm.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

if(WHISPER)
  genai_add_example(whisper ${EXAMPLES_SOURCE_DIR}/whisper.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

if(NEMOTRON_SPEECH)
  genai_add_example(nemotron_speech ${EXAMPLES_SOURCE_DIR}/nemotron_speech.cpp)
endif()

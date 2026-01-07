cmake_minimum_required(VERSION 3.18.1)

project(ortgenaiapp)
set(CMAKE_CXX_STANDARD 20)

# Download and make available nlohmann/json
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.12.0     # Or update to latest release
)
FetchContent_MakeAvailable(nlohmann_json)

option(USE_CXX "Invoke the C++ example" ON)
option(MODEL_CHAT "Build the Model Chat example" OFF)
option(MODEL_QA "Build the Model Q&A example" OFF)
option(MODEL_VISION "Build the Model Vision example" OFF)
option(PHI4-MM "Build the Phi-4 mm example" OFF)
option(WHISPER "Build the Whisper example" OFF)

if(USE_CXX)
  add_compile_definitions(USE_CXX)
endif()

if(WIN32)
  set(ONNXRUNTIME_GENAI_LIB "onnxruntime-genai.dll")
elseif(APPLE)
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.dylib")
elseif(CMAKE_SYSTEM_NAME MATCHES "AIX")
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.a")
else()
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.so")
endif()

# Set default library directory if not specified
if(NOT ORT_GENAI_LIB_DIR)
  set(ORT_GENAI_LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
endif()

file(GLOB ort_genai_libs "${ORT_GENAI_LIB_DIR}/*")

message(STATUS "ORT_GENAI_LIB_DIR: ${ORT_GENAI_LIB_DIR}")

function(prepare_executable executable)
  target_link_directories(${executable} PRIVATE ${ORT_GENAI_LIB_DIR})
  target_link_libraries(${executable} PRIVATE ${ONNXRUNTIME_GENAI_LIB})

  if (ORT_GENAI_INCLUDE_DIR)
    target_include_directories(${executable} PRIVATE ${ORT_GENAI_INCLUDE_DIR})
  else()
    target_include_directories(${executable} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  endif()

  target_link_libraries(${executable} PUBLIC onnxruntime-genai)

  foreach(DEPENDENCY_FILE ${ort_genai_libs})
    if (NOT IS_DIRECTORY ${DEPENDENCY_FILE})
      add_custom_command(
        TARGET ${executable} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DEPENDENCY_FILE} $<TARGET_FILE_DIR:${executable}>
      )
    endif()
  endforeach()
endfunction()

set(EXAMPLES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

if(MODEL_CHAT)
  add_executable(model_chat ${EXAMPLES_SOURCE_DIR}/model_chat.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(model_chat)
  target_link_libraries(model_chat PRIVATE nlohmann_json::nlohmann_json)
endif()

if(MODEL_QA)
  add_executable(model_qa ${EXAMPLES_SOURCE_DIR}/model_qa.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(model_qa)
  target_link_libraries(model_qa PRIVATE nlohmann_json::nlohmann_json)
endif()

if(MODEL_VISION)
  add_executable(model_vision ${EXAMPLES_SOURCE_DIR}/model_vision.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(model_vision)
  target_link_libraries(model_vision PRIVATE nlohmann_json::nlohmann_json)
endif()

if(PHI4-MM)
  add_executable(phi4-mm ${CMAKE_SOURCE_DIR}/src/phi4-mm.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(phi4-mm)
  target_link_libraries(phi4-mm PRIVATE nlohmann_json::nlohmann_json)
endif()

if(WHISPER)
  add_executable(whisper ${CMAKE_SOURCE_DIR}/src/whisper.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(whisper)
  target_link_libraries(whisper PRIVATE nlohmann_json::nlohmann_json)
endif()

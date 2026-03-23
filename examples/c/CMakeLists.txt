cmake_minimum_required(VERSION 3.18.1)

project(ortgenaiapp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Download and make available nlohmann/json
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.12.0     # Or update to latest release
  SYSTEM
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(nlohmann_json)

# Download and make available CLI11
include(FetchContent)
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

find_package(onnxruntime REQUIRED)
find_package(onnxruntime-genai REQUIRED)

function(genai_add_example target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE onnxruntime::onnxruntime-genai onnxruntime::onnxruntime
          nlohmann_json::nlohmann_json CLI11::CLI11)
endfunction()

set(EXAMPLES_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

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
  add_executable(nemotron_speech ${EXAMPLES_SOURCE_DIR}/nemotron_speech.cpp)
  prepare_executable(nemotron_speech)
  target_link_libraries(nemotron_speech PRIVATE nlohmann_json::nlohmann_json)
endif()

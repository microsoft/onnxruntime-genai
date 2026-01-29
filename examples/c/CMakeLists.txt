cmake_minimum_required(VERSION 3.18.1)

project(ortgenaiapp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_CXX "Invoke the C++ example" ON)
option(MODEL_CHAT "Build the Model Chat example" OFF)
option(MODEL_QA "Build the Model Q&A example without multi-turn prompting" OFF)
option(MODEL_VISION "Build the Model Vision example" OFF)
option(PHI4-MM "Build the Phi-4 mm example" OFF)
option(WHISPER "Build the Whisper example" OFF)

if(USE_CXX)
  add_compile_definitions(USE_CXX)
endif()

find_package(onnxruntime-genai REQUIRED)

function(genai_add_example target)
  add_executable(${target} ${ARGN})
  target_link_libraries(${target} PRIVATE onnxruntime::onnxruntime-genai)
endfunction()

set(EXAMPLES_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)

if(MODEL_CHAT)
  genai_add_example(model_chat ${EXAMPLES_SOURCE_DIR}/model_chat.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

if(MODEL_QA)
  genai_add_example(model_qa ${EXAMPLES_SOURCE_DIR}/model_qa.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

if(MODEL_VISION)
  genai_add_example(model_vision ${EXAMPLES_SOURCE_DIR}/model_vision.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

if(PHI4-MM)
  genai_add_example(phi4-mm ${EXAMPLES_SOURCE_DIR}/phi4-mm.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

if(WHISPER)
  genai_add_example(whisper ${EXAMPLES_SOURCE_DIR}/whisper.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
endif()

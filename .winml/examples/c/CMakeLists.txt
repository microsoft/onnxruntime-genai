cmake_minimum_required(VERSION 3.18.1)

project(ortgenaiapp)
set(CMAKE_CXX_STANDARD 20)

option(USE_CXX "Invoke the C++ example" ON)
option(MODEL_CHAT "Build the Model Chat example" OFF)
option(MODEL_QA "Build the Model Q&A example without multi-turn prompting" OFF)
option(MODEL_VISION "Build the Model Vision example" OFF)
option(PHI4-MM "Build the Phi-4 mm example" OFF)
option(WHISPER "Build the Whisper example" OFF)

if(USE_CXX)
  add_compile_definitions(USE_CXX)
endif()

set(CMAKE_C_FLAGS "/EHsc /Qspectre /MP /guard:cf /DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100 /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 /DNTDDI_VERSION=0x0A000000 /O2 /Ob1 /DNDEBUG")
set(CMAKE_CXX_FLAGS "/EHsc /Qspectre /MP /guard:cf /DWIN32 /D_WINDOWS /DWINAPI_FAMILY=100 /DWINVER=0x0A00 /D_WIN32_WINNT=0x0A00 /DNTDDI_VERSION=0x0A000000 /O2 /Ob1 /DNDEBUG")

if(USE_CUDA)
  include(CheckLanguage)
  enable_language(CUDA)
endif()

set(ORT_GENAI_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)

if(WIN32)
  set(ONNXRUNTIME_GENAI_LIB "onnxruntime-genai.dll")
elseif(APPLE)
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.dylib")
elseif(CMAKE_SYSTEM_NAME MATCHES "AIX")
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.a")
else()
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.so")
endif()

if (ORT_GENAI_LIB_DIR)
  file (GLOB ort_genai_libs "${ORT_GENAI_LIB_DIR}/*")
else()
  file(GLOB ort_genai_libs "${CMAKE_SOURCE_DIR}/lib/*")
endif()

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
endif()

if(MODEL_QA)
  add_executable(model_qa ${EXAMPLES_SOURCE_DIR}/model_qa.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(model_qa)
endif()

if(MODEL_VISION)
  add_executable(model_vision ${EXAMPLES_SOURCE_DIR}/model_vision.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(model_vision)
endif()

if(PHI4-MM)
  add_executable(phi4-mm ${CMAKE_SOURCE_DIR}/src/phi4-mm.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(phi4-mm)
endif()

if(WHISPER)
  add_executable(whisper ${CMAKE_SOURCE_DIR}/src/whisper.cpp ${EXAMPLES_SOURCE_DIR}/common.cpp)
  prepare_executable(whisper)
endif()


include(cmake/nuget.cmake)
install_nuget_package(Microsoft.Windows.AI.MachineLearning 0.3.134-beta NUGET_MICROSOFT_WINDOWS_COPILOTRUNTIME_ROOT)
message(STATUS "NUGET_MICROSOFT_WINDOWS_COPILOTRUNTIME_ROOT: ${NUGET_MICROSOFT_WINDOWS_COPILOTRUNTIME_ROOT}")

include(cmake/winrt.cmake)
enable_cppwinrt(phi3
    REFERENCES    
    "${NUGET_MICROSOFT_WINDOWS_COPILOTRUNTIME_ROOT}/lib/uap10.0.22000/Microsoft.Windows.AI.MachineLearning.winmd"
)

set(MicrosoftWindowsCopilotRuntimeRootPath "${NUGET_MICROSOFT_WINDOWS_COPILOTRUNTIME_ROOT}")
set(WcrRuntimeFolder "win-arm64")
set(WcrRuntimeArchitecture "arm64")

set_target_properties(phi3 PROPERTIES VS_GLOBAL_WinMLBootstrapAutoInitializeDisabled true)

set(CMAKE_VERBOSE_MAKEFILE ON)
add_compile_options(/VERBOSE)
set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} /VERBOSE")

set_property(
  TARGET phi3
  APPEND
  PROPERTY VS_PROJECT_IMPORT ${MicrosoftWindowsCopilotRuntimeRootPath}/build/native/Microsoft.Windows.AI.MachineLearning.props
)

target_link_libraries(phi3
  PRIVATE
  ${MicrosoftWindowsCopilotRuntimeRootPath}/build/native/Microsoft.Windows.AI.MachineLearning.targets
)

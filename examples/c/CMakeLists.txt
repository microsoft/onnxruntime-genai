cmake_minimum_required(VERSION 3.18.1)

project(ortgenaiapp)
set(CMAKE_CXX_STANDARD 20)

option(USE_CUDA "Build with CUDA support" OFF)
option(USE_CXX "Invoke the C++ example" ON)

if(USE_CXX)
  add_compile_definitions(USE_CXX)
endif()

if(USE_CUDA)
  include(CheckLanguage)
  enable_language(CUDA)
endif()

set(ORT_GENAI_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)

if(WIN32)
  set(ONNXRUNTIME_GENAI_LIB "onnxruntime-genai.dll")
  set(ONNXRUNTIME_GENAI_DEPENDENCY "*.dll")
elseif(APPLE)
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.dylib")
  set(ONNXRUNTIME_GENAI_DEPENDENCY "*.dylib")
else()
  set(ONNXRUNTIME_GENAI_LIB "libonnxruntime-genai.so")
  set(ONNXRUNTIME_GENAI_DEPENDENCY "*.so")
endif()

add_executable(phi3 ${CMAKE_SOURCE_DIR}/src/main.cpp)
add_executable(phi3v ${CMAKE_SOURCE_DIR}/src/phi3v.cpp)


target_link_directories(phi3 PRIVATE ${ORT_GENAI_LIB_DIR})
target_link_libraries(phi3 PRIVATE ${ONNXRUNTIME_GENAI_LIB})
target_include_directories(phi3 PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_directories(phi3v PRIVATE ${ORT_GENAI_LIB_DIR})
target_link_libraries(phi3v PRIVATE ${ONNXRUNTIME_GENAI_LIB})
target_include_directories(phi3v PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(USE_CUDA)
  set_target_properties(phi3 PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(phi3v PROPERTIES LINKER_LANGUAGE CUDA)
endif()

target_link_libraries(
        phi3
        PUBLIC
        onnxruntime-genai)
target_link_libraries(
        phi3v
        PUBLIC
        onnxruntime-genai)

if(USE_CUDA)
  target_link_libraries(
        phi3
        PUBLIC
        cublas curand cudart)
  target_link_libraries(
        phi3v
        PUBLIC
        cublas curand cudart)
endif()

file(GLOB ort_genai_libs "${CMAKE_SOURCE_DIR}/lib/${ONNXRUNTIME_GENAI_DEPENDENCY}")

foreach(DLL_FILE ${ort_genai_libs})
  add_custom_command(
    TARGET phi3 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL_FILE} $<TARGET_FILE_DIR:phi3>
  )
  add_custom_command(
    TARGET phi3v POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL_FILE} $<TARGET_FILE_DIR:phi3v>
  )
endforeach()
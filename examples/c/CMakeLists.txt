cmake_minimum_required(VERSION 3.18.1)

project(phi2)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")

add_executable(phi2 ${CMAKE_SOURCE_DIR}/src/main.cpp)

add_library(onnxruntime-genai SHARED IMPORTED)
set_target_properties(onnxruntime-genai PROPERTIES
    IMPORTED_LOCATION_RELEASE ${CMAKE_SOURCE_DIR}/lib/onnxruntime-genai.dll
    IMPORTED_IMPLIB_RELEASE ${CMAKE_SOURCE_DIR}/lib/onnxruntime-genai.lib
)
target_include_directories(phi2 PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(
        phi2
        PUBLIC
        onnxruntime-genai)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/phi-2" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/Release")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/lib/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/Release")

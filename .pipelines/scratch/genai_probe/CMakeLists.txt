# Minimal consumer that validates the exported find_package(onnxruntime-genai)
# CMake package produced by the core packaging build. Used by the packaging
# scratch pipeline to prove the incremental SDK hand-off links on real agents.
cmake_minimum_required(VERSION 3.26)
project(genai_probe LANGUAGES CXX)

find_package(onnxruntime-genai CONFIG REQUIRED)

add_executable(genai_probe main.cpp)
target_link_libraries(genai_probe PRIVATE onnxruntime-genai::onnxruntime-genai)

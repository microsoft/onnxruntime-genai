set(TOKENIZER_ROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 10)
  add_compile_definitions(USE_CXX17=1)
  message("TOKENIZER IS USE C++17")
  set(CMAKE_CXX_STANDARD 17)
elseif(USE_CUDA AND CMAKE_CUDA_COMPILER AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12)
  add_compile_definitions(USE_CXX17=1)
  message(" TOKENIZER IS USE C++17 Because of CUDA Version is less than 12")
  set(CMAKE_CXX_STANDARD 17)
else()
  message("TOKENIZER IS USE C++20")
  set(CMAKE_CXX_STANDARD 20)
endif()

file(GLOB tokenizer_srcs CONFIGURE_DEPENDS
   "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
   "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cc"
   "${CMAKE_CURRENT_SOURCE_DIR}/c_api/*.cc"
)

FetchContent_Declare(GSL URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip)
FetchContent_MakeAvailable(GSL)

FetchContent_Declare(simdjson URL https://github.com/simdjson/simdjson/archive/refs/tags/v3.6.3.zip URL_HASH SHA1=2b063a2e81f74a5d1cb937fadf3d2fca0f1edb09)
FetchContent_MakeAvailable(simdjson)

add_library(tokenizer STATIC ${tokenizer_srcs})

if(NOT WIN32)
   target_link_libraries(tokenizer PRIVATE stdc++fs)
endif()

message(STATUS "GSL_SOURCE_DIR: ${GSL_SOURCE_DIR}")
target_include_directories(tokenizer PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GSL_SOURCE_DIR}/include
   ${simdjson_SOURCE_DIR}/singleheader/
)
target_compile_definitions(tokenizer PRIVATE _SILENCE_ALL_CXX20_DEPRECATION_WARNINGS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(simdjson PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_link_libraries(tokenizer PUBLIC simdjson)
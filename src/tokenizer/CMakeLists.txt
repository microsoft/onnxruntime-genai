set(TOKENIZER_ROOT ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
file(GLOB tokenizer_srcs CONFIGURE_DEPENDS
   "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
   "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cc"
   "${CMAKE_CURRENT_SOURCE_DIR}/c_api/*.cc"
)
FetchContent_Declare(spm URL https://github.com/google/sentencepiece/archive/refs/tags/v0.1.99.zip)
set(SPM_USE_EXTERNAL_ABSL OFF CACHE BOOL "Use external absl")
set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "Use built-in protobuf")
set(SPM_ENABLE_SHARED OFF)

FetchContent_GetProperties(spm)
if(NOT spm_POPULATED)
  FetchContent_Populate(spm)
  add_subdirectory(${spm_SOURCE_DIR} ${spm_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

FetchContent_Declare(GSL URL https://github.com/microsoft/GSL/archive/refs/tags/v4.0.0.zip)
FetchContent_MakeAvailable(GSL)

FetchContent_Declare(simdjson URL https://github.com/simdjson/simdjson/archive/refs/tags/v3.6.3.zip URL_HASH SHA1=2b063a2e81f74a5d1cb937fadf3d2fca0f1edb09)
FetchContent_MakeAvailable(simdjson)

add_library(tokenizer ${tokenizer_srcs})
message(STATUS "GSL_SOURCE_DIR: ${GSL_SOURCE_DIR}")
target_include_directories(tokenizer PUBLIC
   ${CMAKE_CURRENT_SOURCE_DIR}
   ${GSL_SOURCE_DIR}/include
   ${simdjson_SOURCE_DIR}/singleheader/
   ${spm_SOURCE_DIR}/src
   ${spm_SOURCE_DIR}/src/builtin_pb
   ${spm_SOURCE_DIR}/third_party/absl
   ${spm_SOURCE_DIR}/third_party/protobuf-lite
)
target_compile_definitions(tokenizer PRIVATE _SILENCE_ALL_CXX20_DEPRECATION_WARNINGS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(tokenizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_dependencies(tokenizer sentencepiece-static)
target_link_libraries(tokenizer PUBLIC sentencepiece-static simdjson)
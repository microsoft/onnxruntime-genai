# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB genai_flatbuffers_srcs CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/flatbuffers/*.h"
    "${CMAKE_SOURCE_DIR}/src/flatbuffers/*.cc"
    )

add_library(genai_flatbuffers STATIC ${genai_flatbuffers_srcs})
target_link_libraries(genai_flatbuffers PRIVATE FlatBuffers::FlatBuffers)
#target_include_directories(genai_flatbuffers PRIVATE FlatBuffers::FlatBuffers)

target_include_directories(genai_flatbuffers PRIVATE ${ORT_HEADER_DIR})
target_link_directories(genai_flatbuffers PRIVATE ${ORT_LIB_DIR})

# Add dependency so the flatbuffers compiler is built if enabled
if (FLATBUFFERS_BUILD_FLATC)
  add_dependencies(genai_flatbuffers flatc)
endif()


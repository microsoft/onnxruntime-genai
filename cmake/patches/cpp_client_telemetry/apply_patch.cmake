# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required.")
endif()

function(ortgenai_replace_required file_path old_text new_text)
  if(NOT EXISTS "${file_path}")
    message(FATAL_ERROR "1DS source file not found: ${file_path}")
  endif()

  file(READ "${file_path}" contents)
  string(FIND "${contents}" "${new_text}" new_position)
  if(NOT new_position EQUAL -1)
    return()
  endif()

  string(FIND "${contents}" "${old_text}" old_position)
  if(old_position EQUAL -1)
    message(FATAL_ERROR "Expected 1DS source text was not found in ${file_path}: ${old_text}")
  endif()

  string(REPLACE "${old_text}" "${new_text}" contents "${contents}")
  file(WRITE "${file_path}" "${contents}")
endfunction()

set(root_cmake "${SOURCE_DIR}/CMakeLists.txt")
set(lib_cmake "${SOURCE_DIR}/lib/CMakeLists.txt")

ortgenai_replace_required(
  "${root_cmake}"
  [=[include_directories(${CMAKE_SOURCE_DIR})]=]
  [=[include_directories(${CMAKE_CURRENT_SOURCE_DIR})]=])

ortgenai_replace_required(
  "${lib_cmake}"
  [=[if(NOT MATSDK_USE_VCPKG_DEPS)]=]
  [=[if(NOT MATSDK_USE_VCPKG_DEPS AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")]=])

ortgenai_replace_required(
  "${lib_cmake}"
  [=[else()
  # Legacy mode: use vendored or system-installed deps
  if(CMAKE_SYSTEM_NAME STREQUAL "Android")]=]
  [=[else()
  # Legacy mode: use vendored or system-installed deps
  if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR MATSDK_BUNDLE_VENDORED_DEPS)]=])

ortgenai_replace_required(
  "${lib_cmake}"
  [=[target_compile_options(sqlite3_bundled PRIVATE -fno-finite-math-only -Wno-unused-function)]=]
  [=[target_compile_options(sqlite3_bundled PRIVATE -fno-finite-math-only -Wno-unused-function)
    target_compile_definitions(sqlite3_bundled PRIVATE HAVE_GETHOSTUUID=0)]=])

ortgenai_replace_required(
  "${lib_cmake}"
  [=[  elseif(PAL_IMPLEMENTATION STREQUAL "WIN32")]=]
  [=[  elseif(APPLE)
    target_link_libraries(mat PRIVATE sqlite3 z ${LIBS})
  elseif(PAL_IMPLEMENTATION STREQUAL "WIN32")]=])

ortgenai_replace_required(
  "${SOURCE_DIR}/lib/system/EventProperties.cpp"
  [=[calloc(sizeof(evt_prop), size)]=]
  [=[calloc(size, sizeof(evt_prop))]=])

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

function(ortgenai_configure_shared_library_exports target)
  if(APPLE)
    set(exports_list "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/exported-symbols.lst")
    target_link_options(${target} PRIVATE "LINKER:-exported_symbols_list,${exports_list}")
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${exports_list}")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR ANDROID)
    set(exports_map "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/exports.map")
    target_link_options(${target} PRIVATE
      "LINKER:--version-script,${exports_map}"
      "LINKER:--exclude-libs,ALL")
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${exports_map}")
  endif()
endfunction()

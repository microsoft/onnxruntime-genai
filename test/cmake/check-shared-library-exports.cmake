# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(APPLE_EXPORTS)
  set(nm_args -g -U -j)
else()
  # GCC toolset compatibility objects can leave local symbols in .dynsym.
  set(nm_args --dynamic --defined-only --extern-only --format=posix)
endif()
execute_process(
  COMMAND "${NM}" ${nm_args} "${LIBRARY}"
  RESULT_VARIABLE nm_result
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error)
if(NOT nm_result STREQUAL "0")
  message(FATAL_ERROR "Could not inspect shared-library exports: ${nm_error}")
endif()

file(READ "${API_HEADER}" api_header)
string(REGEX MATCHALL "OGA_EXPORT[^;\n]+OGA_API_CALL[ \t]+Oga[A-Za-z0-9_]+[ \t]*\\(" declarations "${api_header}")
if(NOT declarations)
  message(FATAL_ERROR "No public API declarations found in ${API_HEADER}")
endif()
set(expected_exports)
foreach(declaration IN LISTS declarations)
  string(REGEX REPLACE ".*OGA_API_CALL[ \t]+(Oga[A-Za-z0-9_]+)[ \t]*\\(" "\\1" symbol "${declaration}")
  list(APPEND expected_exports "${symbol}")
endforeach()
list(REMOVE_DUPLICATES expected_exports)

string(REPLACE "\n" ";" nm_lines "${nm_output}")
set(actual_exports)
foreach(line IN LISTS nm_lines)
  if(APPLE_EXPORTS AND line MATCHES "^_([^ \t\r]+)[\r]?$")
    list(APPEND actual_exports "${CMAKE_MATCH_1}")
  elseif(NOT APPLE_EXPORTS AND line MATCHES "^([^ \t]+)[ \t]+")
    list(APPEND actual_exports "${CMAKE_MATCH_1}")
  endif()
endforeach()

set(missing_exports ${expected_exports})
set(unexpected_exports ${actual_exports})
if(actual_exports)
  list(REMOVE_ITEM missing_exports ${actual_exports})
endif()
list(REMOVE_ITEM unexpected_exports ${expected_exports})
if(missing_exports OR unexpected_exports)
  message(FATAL_ERROR
    "Shared-library export mismatch.\nMissing public APIs: ${missing_exports}\nUnexpected exports: ${unexpected_exports}")
endif()

list(LENGTH actual_exports export_count)
message(STATUS "Shared library exports only the ${export_count} public Oga APIs")

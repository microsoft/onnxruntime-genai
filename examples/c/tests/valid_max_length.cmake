# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

foreach(limit 16 32 512)
  set(limit_args)
  if(limit EQUAL 16)
    set(limit_args -l 16)
  elseif(limit EQUAL 32)
    set(limit_args --max_length 32)
  endif()
  execute_process(
    COMMAND "${EXAMPLE}" -m "${MODEL_DIR}" -e cpu --system_prompt a --user_prompt b
      --non_interactive -s false -v ${limit_args}
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 60
  )
  if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "Limit ${limit}: generation failed (${result})\n${output}\n${error}")
  endif()
  if(NOT output MATCHES "Prompt length: ([0-9]+), New tokens: ([0-9]+)")
    message(FATAL_ERROR "Limit ${limit}: missing generation token counts\n${output}")
  endif()
  math(EXPR total "${CMAKE_MATCH_1} + ${CMAKE_MATCH_2}")
  if(NOT total EQUAL limit)
    message(FATAL_ERROR "Limit ${limit}: generated ${total} total tokens\n${output}")
  endif()
endforeach()

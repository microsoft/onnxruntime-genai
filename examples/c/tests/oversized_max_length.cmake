# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

foreach(option -l --max_length)
  foreach(value 513 2147483647)
    execute_process(
      COMMAND "${EXAMPLE}" -m "${MODEL_DIR}" -e cpu --non_interactive -v "${option}" "${value}"
      RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 15
    )
    if("${result}" STREQUAL "0" OR NOT error MATCHES "cannot be greater than model context_length")
      message(FATAL_ERROR "${option} ${value}: expected context limit rejection, got ${result}\n${output}\n${error}")
    endif()
    if(output MATCHES "Creating model")
      message(FATAL_ERROR "${option} ${value}: validation happened after model construction started")
    endif()
  endforeach()
endforeach()

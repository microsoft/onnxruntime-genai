# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

foreach(option -l --max_length)
  foreach(value 0 -1 -2147483648)
    # A nonexistent model ensures validation happens before configuration loading,
    # without risking a large allocation when testing an unfixed executable.
    execute_process(
      COMMAND "${EXAMPLE}" -m "${CMAKE_CURRENT_BINARY_DIR}/missing-model-for-argument-validation"
        -e cpu --non_interactive -v "${option}" "${value}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE output
      ERROR_VARIABLE error
      TIMEOUT 15
    )
    if("${result}" STREQUAL "0" OR NOT error MATCHES "max_length must be greater than 0")
      message(FATAL_ERROR "${option} ${value}: expected argument rejection, got ${result}\n${output}\n${error}")
    endif()
    if(output MATCHES "Creating model")
      message(FATAL_ERROR "${option} ${value}: validation happened after model construction started")
    endif()
  endforeach()
endforeach()

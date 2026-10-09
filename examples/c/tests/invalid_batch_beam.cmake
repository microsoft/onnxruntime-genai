# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

foreach(option -b --batch_size -n --num_beams)
  if(option STREQUAL "-b" OR option STREQUAL "--batch_size")
    set(name batch_size)
    set(other_args -n 2)
  else()
    set(name num_beams)
    set(other_args -b 2)
  endif()
  foreach(value 0 -1 -2147483648 33 2147483647 1 32)
    # A missing model makes an unfixed binary fail safely, before provider setup.
    # The other dimension is 2 to cover the reported INT_MAX * 2 overflow.
    execute_process(
      COMMAND "${EXAMPLE}" -m "${CMAKE_CURRENT_BINARY_DIR}/missing-model-for-argument-validation"
        -e cpu --non_interactive -v "${option}" "${value}" ${other_args}
      RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 15
    )
    if(value EQUAL 1 OR value EQUAL 32)
      # Both valid endpoints must pass argument validation and reach config loading.
      set(expected "Error opening")
    else()
      set(expected "${name} (${value}) must be in [1, 32]")
    endif()
    string(FIND "${error}" "${expected}" error_position)
    if("${result}" STREQUAL "0" OR error_position EQUAL -1)
      message(FATAL_ERROR "${option} ${value}: expected '${expected}', got ${result}\n${output}\n${error}")
    endif()
    if(output MATCHES "Creating model")
      message(FATAL_ERROR "${option} ${value}: validation happened after model construction started")
    endif()
  endforeach()
endforeach()

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT DEFINED BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "BINARY must name an existing Linux shared library.")
endif()

file(GET_RUNTIME_DEPENDENCIES
  LIBRARIES "${BINARY}"
  RESOLVED_DEPENDENCIES_VAR resolved_dependencies
  UNRESOLVED_DEPENDENCIES_VAR unresolved_dependencies)

foreach(dependency IN LISTS resolved_dependencies unresolved_dependencies)
  get_filename_component(dependency_name "${dependency}" NAME)
  if(dependency_name MATCHES
      "^(libcurl|libmbedtls|libmbedx509|libmbedcrypto|libssl|libcrypto|libpsl|libidn|libbrotli|libzstd|libssh|libnghttp)")
    message(FATAL_ERROR
      "Linux telemetry must be self-contained, but ${BINARY} depends on ${dependency_name}.")
  endif()
endforeach()

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include_guard()

# Define the Version Info
file(READ "VERSION_INFO" ver)
set(VERSION_INFO ${ver})

# Example:
# VERSION_INFO: 0.4.0-dev
# VERSION_STR: 0.4.0
# VERSION_SUFFIX: dev
# VERSION_MAJOR: 0
# VERSION_MINOR: 4
# VERSION_PATCH: 0
string(REPLACE "-" ";" VERSION_LIST ${VERSION_INFO})
list(GET VERSION_LIST 0 VERSION_STR)
# Check if it is a stable or dev version
list(LENGTH VERSION_LIST VERSION_LIST_LENGTH)
if(VERSION_LIST_LENGTH GREATER 1)
    list(GET VERSION_LIST 1 VERSION_SUFFIX)
else()
    set(VERSION_SUFFIX "")  # Set VERSION_SUFFIX to empty if stable version
endif()
string(REPLACE "." ";" VERSION_LIST ${VERSION_STR})
list(GET VERSION_LIST 0 VERSION_MAJOR)
list(GET VERSION_LIST 1 VERSION_MINOR)
list(GET VERSION_LIST 2 VERSION_PATCH)

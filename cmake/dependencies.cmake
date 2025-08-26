#[[====================================================================================================================

====================================================================================================================]]#
include_guard()

include(${CMAKE_SOURCE_DIR}/cmake/nuget.cmake)

install_nuget_package(Microsoft.Windows.ImplementationLibrary 1.0.240803.1 NUGET_MICROSOFT_WINDOWS_IMPLEMENTATIONLIBRARY)

add_library(wil INTERFACE)

target_include_directories(wil
    INTERFACE
        ${NUGET_MICROSOFT_WINDOWS_IMPLEMENTATIONLIBRARY}/include
)

#[[====================================================================================================================
    az_download_dependency
    ----------------------

    Downloads a dependency from Azure DevOps Artifacts using the Azure CLI.

        az_download_dependency(
            NAME <package name>
            VERSION <package version>
            ORGANIZATION <organization name>
            PROJECT <project name>
            FEED <feed name>
            SCOPE <scope name>
            PATH <path to download to>
            OUTPUT_VAR <output variable name>
        )

    Note:
        - The package will be downloaded to the path: ${CMAKE_SOURCE_DIR}/${ARG_PATH}/${ARG_NAME}.
        - If 'OUTPUT_VAR' is already set and points to a different path, the function will return without downloading
            the package, assuming that the specified content is a private build.
        - The function will check if the package already exists at the specified path and if it has the correct version.
        - If the package exists and is up to date, it will not download again.
        - If the package exists but is outdated or empty, it will be removed and re-downloaded.
        - The Azure CLI must be installed and available in your PATH.
        - The Azure DevOps extension for Azure CLI must be installed.
====================================================================================================================]]#
function(az_download_dependency)
    # Parse required arguments
    set(options)
    set(oneValueArgs NAME VERSION ORGANIZATION PROJECT FEED SCOPE PATH OUTPUT_VAR)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    foreach(arg NAME VERSION ORGANIZATION PROJECT FEED SCOPE PATH OUTPUT_VAR)
        if(NOT DEFINED ARG_${arg})
            message(FATAL_ERROR "Missing required argument for az_download_dependency: ${arg}")
        endif()
    endforeach()

    # Construct full path
    set(DEPENDENCY_PATH "${CMAKE_SOURCE_DIR}/${ARG_PATH}/${ARG_NAME}")

    # If the output variable is already set _and_ points to a different path, then the caller is using a private build.
    if( (NOT("${${ARG_OUTPUT_VAR}}" STREQUAL "")) AND
        (NOT("${${ARG_OUTPUT_VAR}}" STREQUAL "${DEPENDENCY_PATH}")))
        message(STATUS "Using private dependency for '${ARG_NAME}' - '${ARG_OUTPUT_VAR}' is set to '${${ARG_OUTPUT_VAR}}'")
        return()
    endif()

    # Always set the output variable in parent scope and as a CACHE variable
    set(${ARG_OUTPUT_VAR} "${DEPENDENCY_PATH}" PARENT_SCOPE)
    set(${ARG_OUTPUT_VAR} "${DEPENDENCY_PATH}" CACHE PATH "Path to downloaded dependency: ${ARG_NAME}" FORCE)

    # Create version stamp file path
    set(VERSION_STAMP_FILE "${DEPENDENCY_PATH}/.version_stamp")

    # Check if dependency exists and has correct version
    set(SHOULD_DOWNLOAD TRUE)
    if(EXISTS "${DEPENDENCY_PATH}")
        if(EXISTS "${VERSION_STAMP_FILE}")
            file(READ "${VERSION_STAMP_FILE}" EXISTING_VERSION)
            string(STRIP "${EXISTING_VERSION}" EXISTING_VERSION)
            if("${EXISTING_VERSION}" STREQUAL "${ARG_VERSION}")
                # Version matches, check if the directory actually has content
                file(GLOB DIR_CONTENTS "${DEPENDENCY_PATH}/*")
                list(LENGTH DIR_CONTENTS CONTENT_COUNT)
                if(CONTENT_COUNT GREATER 1) # At least one file besides version stamp
                    message(STATUS "Dependency ${ARG_NAME} version ${ARG_VERSION} is up to date at ${DEPENDENCY_PATH}")
                    set(SHOULD_DOWNLOAD FALSE)
                else()
                    message(STATUS "Dependency ${ARG_NAME} directory exists but appears empty, will re-download")
                endif()
            else()
                message(STATUS "Dependency ${ARG_NAME} has version ${EXISTING_VERSION} but ${ARG_VERSION} is required, will re-download")
            endif()
        else()
            message(STATUS "Dependency ${ARG_NAME} exists but has no version information, will re-download")
        endif()

        # Remove directory if download is needed
        if(SHOULD_DOWNLOAD)
            message(STATUS "Removing outdated or incomplete dependency at ${DEPENDENCY_PATH}")
            file(REMOVE_RECURSE "${DEPENDENCY_PATH}")
        endif()
    endif()

    # Return if download not needed
    if(NOT SHOULD_DOWNLOAD)
        return()
    endif()

    # Set up authentication
    set(CMAKE_FIND_DEBUG_MODE TRUE)
    find_program(AZ_CLI "az.cmd")
    if(NOT AZ_CLI)
        message(NOTICE "Azure CLI not found. Install with: winget install -e --id Microsoft.AzureCLI")
        message(NOTICE "Use: set(CMAKE_FIND_DEBUG_MODE TRUE) to debug.")
        message(FATAL_ERROR "Please install Azure CLI and ensure it's in your PATH.")
    endif()

    # NOTE: There's a known issue with Azure CLI and CMake's execute_process:
    # https://github.com/Azure/azure-cli/issues/15910
    # As a workaround, we wrap all az calls with "cmd /c" to ensure they execute properly

    # Check if Azure DevOps extension is installed
    message(VERBOSE "Checking for Azure DevOps extension...")
    execute_process(
        COMMAND cmd /c ${AZ_CLI} extension show --name azure-devops
        RESULT_VARIABLE AZ_EXT_RESULT
    )

    if(NOT "${AZ_EXT_RESULT}" STREQUAL "0")
        message(STATUS "Installing Azure DevOps extension...")
        execute_process(
            COMMAND cmd /c ${AZ_CLI} extension add --name azure-devops --yes
            RESULT_VARIABLE AZ_EXT_ADD_RESULT
            OUTPUT_VARIABLE AZ_EXT_ADD_OUTPUT
            ERROR_VARIABLE AZ_EXT_ADD_ERROR
        )

        if(NOT "${AZ_EXT_ADD_RESULT}" STREQUAL "0")
            message(FATAL_ERROR "Failed to install Azure DevOps extension: ${AZ_EXT_ADD_ERROR}")
        endif()
    endif()

    # Check login status only if AZURE_DEVOPS_EXT_PAT is not set
    if(NOT DEFINED ENV{AZURE_DEVOPS_EXT_PAT} OR "$ENV{AZURE_DEVOPS_EXT_PAT}" STREQUAL "")
        message(VERBOSE "Checking Azure CLI login status...")
        execute_process(
            COMMAND cmd /c ${AZ_CLI} account show
            RESULT_VARIABLE AZ_LOGIN_RESULT
        )

        if(NOT "${AZ_LOGIN_RESULT}" STREQUAL "0")
            message(NOTICE "Azure CLI is not logged in. Attempting automatic login...")
            execute_process(
                COMMAND cmd /c ${AZ_CLI} login --only-show-errors
                RESULT_VARIABLE AZ_AUTO_LOGIN_RESULT
                ERROR_VARIABLE AZ_AUTO_LOGIN_ERROR
            )

            if(NOT "${AZ_AUTO_LOGIN_RESULT}" STREQUAL "0")
                message(FATAL_ERROR "Authentication failed. Please run 'az login' manually then retry. Error: ${AZ_AUTO_LOGIN_ERROR}")
            else()
                message(NOTICE "Successfully logged in to Azure CLI")
            endif()
        endif()
    else()
        message(STATUS "AZURE_DEVOPS_EXT_PAT is set; skipping az login.")
    endif()

    # Create the destination directory
    file(MAKE_DIRECTORY "${DEPENDENCY_PATH}")

    # Download the package
    message(STATUS "Downloading ${ARG_NAME} version ${ARG_VERSION}...")
    execute_process(
        COMMAND cmd /c ${AZ_CLI} artifacts universal download
            --organization "${ARG_ORGANIZATION}"
            --project "${ARG_PROJECT}"
            --scope "${ARG_SCOPE}"
            --feed "${ARG_FEED}"
            --name "${ARG_NAME}"
            --version "${ARG_VERSION}"
            --path "${DEPENDENCY_PATH}"
        RESULT_VARIABLE AZ_DOWNLOAD_RESULT
        ERROR_VARIABLE AZ_DOWNLOAD_ERROR
    )

    if(NOT "${AZ_DOWNLOAD_RESULT}" STREQUAL "0")
        message(FATAL_ERROR "Download failed: ${AZ_DOWNLOAD_ERROR}")
    endif()

    # Verify download was successful
    file(GLOB DIR_CONTENTS "${DEPENDENCY_PATH}/*")
    list(LENGTH DIR_CONTENTS CONTENT_COUNT)
    if(CONTENT_COUNT EQUAL 0)
        message(FATAL_ERROR "Download of ${ARG_NAME} appeared successful but directory is empty")
    endif()

    # Create version stamp file
    file(WRITE "${VERSION_STAMP_FILE}" "${ARG_VERSION}")

    message(STATUS "Successfully downloaded ${ARG_NAME} version ${ARG_VERSION}")
endfunction()

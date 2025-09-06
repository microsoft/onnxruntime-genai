include_guard()

#[[====================================================================================================================
    install_nuget_package
    ---------------------
    Downloads a NuGet package and returns the path to it.

        install_nuget_package(
            <package name>
            <package version>
            <variable name>
        )
====================================================================================================================]]#
function(install_nuget_package NUGET_PACKAGE_NAME NUGET_PACKAGE_VERSION NUGET_PACKAGE_PATH_PROPERTY)
    if(NOT NUGET_PACKAGE_ROOT_PATH)
        set(NUGET_PACKAGE_ROOT_PATH ${CMAKE_BINARY_DIR}/__nuget)
    endif()

    set(NUGET_PACKAGE_PATH "${NUGET_PACKAGE_ROOT_PATH}/${NUGET_PACKAGE_NAME}.${NUGET_PACKAGE_VERSION}")

    if(NOT EXISTS "${NUGET_PACKAGE_PATH}")
        find_program(NUGET_PATH
            NAMES nuget nuget.exe
        )

        if(NUGET_PATH STREQUAL "NUGET_PATH-NOTFOUND")
            message(FATAL_ERROR "nuget.exe cannot be found.")
        endif()

        set(NUGET_COMMAND ${NUGET_PATH} install ${NUGET_PACKAGE_NAME})
        list(APPEND NUGET_COMMAND -OutputDirectory ${NUGET_PACKAGE_ROOT_PATH})
        list(APPEND NUGET_COMMAND -Version ${NUGET_PACKAGE_VERSION})
        list(APPEND NUGET_COMMAND -PackageSaveMode nuspec)

        message(STATUS "Downloading ${NUGET_PACKAGE_NAME} ${NUGET_PACKAGE_VERSION}")
        message(VERBOSE "install_nuget_package: NUGET_COMMAND = ${NUGET_COMMAND}")

        execute_process(
            COMMAND ${NUGET_COMMAND}
            OUTPUT_VARIABLE NUGET_OUTPUT
            ERROR_VARIABLE NUGET_ERROR
            RESULT_VARIABLE NUGET_RESULT
        )

        message(VERBOSE "install_nuget_package: NUGET_OUTPUT = ${NUGET_OUTPUT}")
        if(NOT (NUGET_RESULT STREQUAL 0))
            message(FATAL_ERROR "install_nuget_package: Install failed with: ${NUGET_ERROR}")
        endif()
    endif()

    set(${NUGET_PACKAGE_PATH_PROPERTY} "${NUGET_PACKAGE_PATH}" PARENT_SCOPE)
endfunction()

#[[====================================================================================================================
    add_nuget_package
    -----------------
    Creates a target to package files into a NuGet package.

        add_nuget_package(<target>
            <nuspec file>
            VERSION <version>
            [PROPERTIES
                <<PROPERTY_NAME> <PROPERTY_VALUE>>+
            ]
            [FILES
                <<SOURCE> <TARGET>>+
            ]
        )

====================================================================================================================]]#
function(add_nuget_package TARGET NUSPEC_FILE)
    find_program(NUGET_PATH
        NAMES nuget nuget.exe
    )

    if(NUGET_PATH STREQUAL "NUGET_PATH-NOTFOUND")
        message(FATAL_ERROR "nuget.exe cannot be found.")
    endif()

    set(OPTIONS)
    set(ONE_VALUE_KEYWORDS VERSION)
    set(MULTI_VALUE_KEYWORDS FILES PROPERTIES)

    cmake_parse_arguments(PARSE_ARGV 2 NUGET_PACK "${OPTIONS}" "${ONE_VALUE_KEYWORDS}" "${MULTI_VALUE_KEYWORDS}")

    if(NOT NUGET_PACK_VERSION)
        message(FATAL_ERROR "add_nuget_package: 'VERSION' must be specified.")
    endif()

    # Set NUGET_BASE_NAME to NUSPEC_FILE without '.nuspec'.
    string(REPLACE ".nuspec" "" NUGET_BASE_NAME ${NUSPEC_FILE})

    # Walk the 'FILES' and:
    #    1. Build the 'NUGET_DEPENDENCIES' property that will be used for dependency checking the custom build step.
    #    2. Build the 'NUGET_FILES' property that will be added to the '<files\>' section of the configured file.
    set(NUGET_FILES)
    set(NUGET_DEPENDENCIES ${NUSPEC_FILE})
    while(NUGET_PACK_FILES)
        list(POP_FRONT NUGET_PACK_FILES FILE_SOURCE)
        list(POP_FRONT NUGET_PACK_FILES FILE_TARGET)

        list(APPEND NUGET_DEPENDENCIES ${FILE_SOURCE})

        # To allow '$<CONFIG>' usage in FILE_SOURCE, replace '$<CONFIG>'-->'$configuration$' and use nuget.exe support
        # to replace $configuration$ in the target.
        string(REPLACE "\$<CONFIG>" "$configuration$" FILE_SOURCE ${FILE_SOURCE})
        set(NUGET_FILES "${NUGET_FILES}\n        <file src=\"${FILE_SOURCE}\" target=\"${FILE_TARGET}\" />")
    endwhile()

    configure_file(${NUSPEC_FILE} ${CMAKE_CURRENT_BINARY_DIR}/${NUSPEC_FILE})

    set(NUGET_COMMAND ${NUGET_PATH} pack ${CMAKE_CURRENT_BINARY_DIR}/${NUSPEC_FILE})
    list(APPEND NUGET_COMMAND -OutputDirectory ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>)
    list(APPEND NUGET_COMMAND -Version ${NUGET_PACK_VERSION})

    while(NUGET_PACK_PROPERTIES)
        list(POP_FRONT NUGET_PACK_PROPERTIES PROPERTY_NAME)
        list(POP_FRONT NUGET_PACK_PROPERTIES PROPERTY_VALUE)

        list(APPEND NUGET_COMMAND -Properties "\"${PROPERTY_NAME}=${PROPERTY_VALUE}\"")
    endwhile()
    list(APPEND NUGET_COMMAND -Properties "\"configuration=$<CONFIG>\"")

    message(VERBOSE "add_nuget_package: NUGET_COMMAND = ${NUGET_COMMAND}")

    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${NUGET_BASE_NAME}.${NUGET_PACK_VERSION}.nupkg
        COMMAND ${NUGET_COMMAND}
        DEPENDS ${NUGET_DEPENDENCIES}
        COMMENT "NuGet: ${NUSPEC_FILE}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    add_custom_target(${TARGET} ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${NUGET_BASE_NAME}.${NUGET_PACK_VERSION}.nupkg
    )
endfunction()

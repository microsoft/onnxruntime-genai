include_guard()

include(${CMAKE_SOURCE_DIR}/cmake/nuget.cmake)

#[[====================================================================================================================
    enable_cppwinrt
    ---------------
    Enables CppWinRT support on the specified target.

        enable_cppwinrt(<target>
            [ROOT_NAMESPACE <root namespace>]
            [REFERENCES <paths>+]
            <COMPILE_MODULE>
            [GENERATE_METADATA]
        )

    `ROOT_NAMESPACE` specifies the root namespace of the module, influencing the name of the generated .winmd file.
    `REFERENCES` specified additional paths to .winmd files to reference. The only supported generator expression is
    '$<CONFIG>'.
    `COMPILE_MODULE` controls whether the generated 'module.g.cpp' is added to the target as source.
====================================================================================================================]]#
function(enable_cppwinrt TARGET)

    if(NOT (CMAKE_GENERATOR MATCHES "^Visual Studio"))
        message(FATAL_ERROR "'enable_cppwinrt' is only supported with the Visual Studio Generator.")
    endif()

    set(OPTIONS COMPILE_MODULE GENERATE_METADATA)
    set(ONE_VALUE_KEYWORDS ROOT_NAMESPACE)
    set(MULTI_VALUE_KEYWORDS REFERENCES)

    cmake_parse_arguments(PARSE_ARGV 1 CPPWINRT "${OPTIONS}" "${ONE_VALUE_KEYWORDS}" "${MULTI_VALUE_KEYWORDS}")

    # Add WinRT processing using - with Visual Studio generators - the Microsoft.Windows.CppWinRT NuGet.
    # The NuGet is downloaded during CMake's configuration phase, and the NuGet's .props and .targets
    # files are added to the CMake-generated .vcxproj.
    #
    install_nuget_package(Microsoft.Windows.CppWinRT 2.0.240405.15 NUGET_MICROSOFT_WINDOWS_CPPWINRT)

    target_link_libraries(${TARGET}
        PRIVATE
            ${NUGET_MICROSOFT_WINDOWS_CPPWINRT}/build/native/Microsoft.Windows.CppWinRT.targets
    )

    set_property(
        TARGET ${TARGET}
        APPEND
        PROPERTY VS_PROJECT_IMPORT
            ${NUGET_MICROSOFT_WINDOWS_CPPWINRT}/build/native/Microsoft.Windows.CppWinRT.props
    )

    # Set VS_GLOBAL_* properties to configure CppWinRT:
    #
    #   * 'VS_GLOBAL_CppWinRT*' configures the invocation of the cppwinrt tooling.
    #   * 'VS_GLOBAL_RootNamespace' to enable CppWinRT to derive the name of the generated .winmd file.
    #
    set_target_properties(${TARGET}
        PROPERTIES
            VS_GLOBAL_CppWinRTOptimized true
            VS_GLOBAL_CppWinRTRootNamespaceAutoMerge true
            VS_GLOBAL_CppWinRTUsePrefixes true
    )

    if(CPPWINRT_GENERATE_METADATA)
        set_target_properties(${TARGET}
            PROPERTIES
                VS_GLOBAL_CppWinRTGenerateWindowsMetadata true
        )
    endif()

    if(CPPWINRT_ROOT_NAMESPACE)
        set_target_properties(${TARGET}
            PROPERTIES
                VS_GLOBAL_RootNamespace ${CPPWINRT_ROOT_NAMESPACE}
        )
    endif()

    if(CPPWINRT_COMPILE_MODULE)
        target_sources(${TARGET}
            PRIVATE
                "$(IntDir)\\Generated Files\\module.g.cpp"
        )

        set_source_files_properties("$(IntDir)\\Generated Files\\module.g.cpp"
            PROPERTIES GENERATED true
        )
    endif()

    # CppWinRT expects the generated .winmd file to be written to '%(Midl.OutputDirectory)%(Midl.MetadataFileName)' -
    # where:
    #   %(Midl.OutputDirectory) is passed as midl.exe's /out parameter
    #   %(Midl.MetadataFileName) is passed as midl.exe's /winmd parameter.
    #
    # But midl.exe's '/winmd' parameter doesn't appear to be '/out'-relative, it is relative to the working directory, so
    # if %(Midl.OutputDirectory) differs from the working directory, CppWinRT can't find the midl-written .winmd file.
    #
    # The CMake Visual Studio generator unconditionally sets %(Midl.OutputDirectory) to '$(ProjectDir)/$(IntDir)', and
    # midl.exe is invoked with a working directory of '$(ProjectDir)', so CppWinRT fails to find the midl-written .winmd
    # file. As a result, .idl files that are consumed by CppWinRT need to have their metadata overridden, setting
    # %(Midl.OutputDirectory) back to $(ProjectDir), and updating the path for the written files to prefix them with
    # the portion that was removed from %(Midl.OutputDirectory).
    get_target_property(CANDIDATE_SOURCES ${TARGET} SOURCES)

    foreach(CANDIDATE_SOURCE IN ITEMS ${CANDIDATE_SOURCES})
        if(NOT (CANDIDATE_SOURCE MATCHES "\\.idl$"))
            continue()
        endif()

        message(VERBOSE "enable_cppwinrt: Configuring ${CANDIDATE_SOURCE}")
        set_property(
            SOURCE ${CANDIDATE_SOURCE}
            PROPERTY VS_SETTINGS
                "OutputDirectory=$(ProjectDir)"
                "HeaderFileName=$(IntDir)\\%(Filename).h"
                "TypeLibraryName=$(IntDir)\\%(Filename).tlb"
                "InterfaceIdentifierFileName=$(IntDir)\\%(Filename)_i.c"
                "ProxyFileName=$(IntDir)\\%(Filename)_p.c"
                "MetadataFileName=$(CppWinRTUnmergedDir)\\%(Filename).winmd"
        )
    endforeach()

    # Process the REFERENCES, adding them to the target
    foreach(CPPWINRT_REFERENCE IN ITEMS ${CPPWINRT_REFERENCES})
        string(REPLACE "\$<CONFIG>" "$(Configuration)" CPPWINRT_REFERENCE ${CPPWINRT_REFERENCE})
        set_property(
            TARGET ${TARGET}
            APPEND
            PROPERTY
                VS_WINRT_REFERENCES ${CPPWINRT_REFERENCE}
        )
    endforeach()
endfunction()

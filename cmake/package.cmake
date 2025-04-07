set(ONNXRUNTIME_GENAI_PUBLIC_HEADERS
  "${PROJECT_SOURCE_DIR}/src/ort_genai_c.h;${PROJECT_SOURCE_DIR}/src/ort_genai.h"
)

set_target_properties(
  onnxruntime-genai PROPERTIES
  PUBLIC_HEADER "${ONNXRUNTIME_GENAI_PUBLIC_HEADERS}"
)
install(TARGETS
  onnxruntime-genai
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
  FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(USE_CUDA)
  install(TARGETS
    onnxruntime-genai-cuda
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif()

if (WIN32)
  install(FILES $<TARGET_PDB_FILE:onnxruntime-genai> DESTINATION lib CONFIGURATIONS RelWithDebInfo Debug)
endif()
set(CPACK_PACKAGE_VENDOR "Microsoft")
set(CPACK_PACKAGE_NAME "onnxruntime-genai")
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${PROJECT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/microsoft/onnxruntime-genai")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/package")
if (WIN32)
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x64")
    if (USE_CUDA)
      set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-win-x64-cuda")
    elseif (USE_DML)
      set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-win-x64-dml")
    else ()
      set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-win-x64")
    endif ()
  else ()
    if(USE_DML)
      set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-win-arm64-dml")
    else()
      set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-win-arm64")
    endif()
  endif ()
elseif (LINUX)
  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x64")
    if (USE_CUDA)
      set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-linux-x64-cuda")
    elseif (USE_ROCM)
      set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-linux-x64-rocm")
    else ()
      set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-linux-x64")
    endif ()
  else ()
    set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-linux-arm64")
  endif ()
elseif (APPLE)
  if (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-osx-x64")
  else ()
    set(CPACK_PACKAGE_FILE_NAME "onnxruntime-genai-${VERSION_INFO}-osx-arm64")
  endif()
endif ()

if (WIN32)
  set(CPACK_GENERATOR "ZIP")
else ()
  set(CPACK_GENERATOR "TGZ")
endif ()

set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY TRUE)
install(FILES
  "${PROJECT_SOURCE_DIR}/README.md"
  "${PROJECT_SOURCE_DIR}/ThirdPartyNotices.txt"
  "${PROJECT_SOURCE_DIR}/SECURITY.md"
  "${PROJECT_SOURCE_DIR}/LICENSE"
  DESTINATION .)

include(CPack)


# Assemble the Apple static framework (iOS and macOS)
if(BUILD_APPLE_FRAMEWORK)
  # create Info.plist for the framework and podspec for CocoaPods (optional)
  set(MACOSX_FRAMEWORK_NAME "onnxruntime-genai")
  set(MACOSX_FRAMEWORK_IDENTIFIER "com.microsoft.onnxruntime-genai")
  # Need to include CoreML as a weaklink for CocoaPods package if the EP is enabled
  if(USE_COREML)
    set(APPLE_WEAK_FRAMEWORK "\\\"CoreML\\\"")
  endif()
  set(INFO_PLIST_PATH "${CMAKE_CURRENT_BINARY_DIR}/Info.plist")
  configure_file(${REPO_ROOT}/cmake/Info.plist.in ${INFO_PLIST_PATH})
  configure_file(
    ${REPO_ROOT}/tools/ci_build/github/apple/framework_info.json.template
    ${CMAKE_CURRENT_BINARY_DIR}/framework_info.json)

  set_target_properties(onnxruntime-genai PROPERTIES
    FRAMEWORK TRUE
    FRAMEWORK_VERSION A
    MACOSX_FRAMEWORK_INFO_PLIST ${INFO_PLIST_PATH}
  )

  if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    set_target_properties(onnxruntime-genai PROPERTIES
      MACOSX_RPATH TRUE
    )
  else()
    set_target_properties(onnxruntime-genai PROPERTIES INSTALL_RPATH "@loader_path")
  endif()

  # when building for mac catalyst, the CMAKE_OSX_SYSROOT is set to MacOSX as well, to avoid duplication,
  # we specify as `-macabi` in the name of the output static apple framework directory.
  if (PLATFORM_NAME STREQUAL "macabi")
    set(STATIC_FRAMEWORK_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-macabi)
  else()
    set(STATIC_FRAMEWORK_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-${CMAKE_OSX_SYSROOT})
  endif()

  # Setup the various directories required. Remove any existing ones so we start with a clean directory.
  set(STATIC_FRAMEWORK_DIR ${STATIC_FRAMEWORK_OUTPUT_DIR}/static_framework/onnxruntime-genai.framework)

  # Copy the onnxruntime-genai shared library to the framework directory
  add_custom_command(TARGET onnxruntime-genai POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E
                     copy $<TARGET_FILE:onnxruntime-genai> ${STATIC_FRAMEWORK_DIR}/onnxruntime-genai)

    # Assemble the other pieces of the static framework
  add_custom_command(TARGET onnxruntime-genai POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E
                     copy_if_different ${INFO_PLIST_PATH} ${STATIC_FRAMEWORK_DIR}/Info.plist)

  # add the framework header files
  set(STATIC_FRAMEWORK_HEADER_DIR ${STATIC_FRAMEWORK_DIR}/Headers)
  file(MAKE_DIRECTORY ${STATIC_FRAMEWORK_HEADER_DIR})

  foreach(h_ ${ONNXRUNTIME_GENAI_PUBLIC_HEADERS})
  get_filename_component(HEADER_NAME_ ${h_} NAME)
  add_custom_command(TARGET onnxruntime-genai POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E
                     copy_if_different ${h_} ${STATIC_FRAMEWORK_HEADER_DIR}/${HEADER_NAME_})
  endforeach()

endif()
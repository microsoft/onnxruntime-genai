set_target_properties(
  onnxruntime-genai PROPERTIES PUBLIC_HEADER
  "${PROJECT_SOURCE_DIR}/src/ort_genai_c.h;${PROJECT_SOURCE_DIR}/src/ort_genai.h"
)
install(TARGETS
  onnxruntime-genai
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  ARCHIVE DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
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

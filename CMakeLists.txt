# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.28)
include(FetchContent)
include(CMakeDependentOption)
project(Generators LANGUAGES C CXX)

# All Options should be defined in cmake/options.cmake This must be included before any other cmake file is included
include(cmake/options.cmake)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR  "GCC version must be greater than or equal to 11")
endif()

# Avoid warning of Calling FetchContent_Populate(Lib) is deprecated temporarily
# TODO: find a better way to handle the header-only 3rd party deps
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
  cmake_policy(SET CMP0169 OLD)
endif()

if(MSVC)
  # DLL initialization errors due to old conda msvcp140.dll dll are a result of the new MSVC compiler
  # See https://developercommunity.visualstudio.com/t/Access-violation-with-std::mutex::lock-a/10664660#T-N10668856
  # Remove this definition once the conda msvcp140.dll dll is updated.
  add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

include(cmake/external/onnxruntime_external_deps.cmake)
# All Global variables, including GLOB, for the top level CMakeLists.txt should be defined here
include(cmake/global_variables.cmake)
# Checking if CUDA is supported
include(cmake/check_cuda.cmake)
# Checking if ROCm is supported
include(cmake/check_rocm.cmake)
# Checking if DML is supported
include(cmake/check_dml.cmake)

include(cmake/cxx_standard.cmake)

add_compile_definitions(BUILDING_ORT_GENAI_C)
if(MSVC)
  # set updated value for __cplusplus macro instead of 199711L
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)

  add_compile_options(
      "$<$<COMPILE_LANGUAGE:C,CXX>:/w15038>"
      "$<$<COMPILE_LANGUAGE:C,CXX>:/wd4100>"
      "$<$<COMPILE_LANGUAGE:C,CXX>:/W4>"
      "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>"
  )
endif()

if(ENABLE_TESTS)
  # call enable_testing so we can add tests from subdirectories (e.g. test and src/java)
  # it applies recursively to all subdirectories
  enable_testing()
  if (TEST_PHI2)
    add_compile_definitions(TEST_PHI2=1)
  else()
    add_compile_definitions(TEST_PHI2=0)
  endif()
endif()


if(WIN32)
  add_library(onnxruntime-genai SHARED ${generator_srcs} "${GENERATORS_ROOT}/dll/onnxruntime-genai.rc")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_INFO=\"${VERSION_INFO}\")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MAJOR=${VERSION_MAJOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MINOR=${VERSION_MINOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_PATCH=${VERSION_PATCH})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_SUFFIX=${VERSION_SUFFIX})
  target_compile_definitions(onnxruntime-genai PRIVATE FILE_NAME=\"onnxruntime-genai.dll\")

  add_library(onnxruntime-genai-static STATIC ${generator_srcs})
else()
  add_library(onnxruntime-genai SHARED ${generator_srcs})
  add_library(onnxruntime-genai-static STATIC ${generator_srcs})
endif()

target_include_directories(onnxruntime-genai PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai-static PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/include)
target_include_directories(onnxruntime-genai PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/shared/api/)
target_include_directories(onnxruntime-genai-static PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/include)
target_include_directories(onnxruntime-genai-static PUBLIC ${onnxruntime_extensions_SOURCE_DIR}/shared/api/)
target_link_libraries(onnxruntime-genai PRIVATE onnxruntime_extensions)
target_link_libraries(onnxruntime-genai-static PUBLIC onnxruntime_extensions)
target_link_directories(onnxruntime-genai PRIVATE ${ORT_LIB_DIR})

# we keep the shared libraries disconnected on Android as they will come from separate AARs and we don't want to force
# the ORT version to match in both.
if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
target_link_libraries(onnxruntime-genai PRIVATE ${ONNXRUNTIME_LIB})
endif()

if(APPLE)
target_link_libraries(onnxruntime-genai PRIVATE "-framework Foundation" "-framework CoreML")
endif()

set_target_properties(onnxruntime-genai PROPERTIES FOLDER "Sources")
set_target_properties(onnxruntime-genai-static PROPERTIES FOLDER "Sources")
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${generator_srcs})

set(ONNXRUNTIME_GENAI_PUBLIC_HEADERS
  ${SRC_ROOT}/ort_genai.h
  ${SRC_ROOT}/ort_genai_c.h
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  set_target_properties(onnxruntime-genai PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(onnxruntime-genai PRIVATE cublasLt cublas curand cufft cudart)
  #  onnxruntime-genai-static is statically linked under Windows
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(onnxruntime-genai-static PROPERTIES LINKER_LANGUAGE CUDA)
    target_link_libraries(onnxruntime-genai-static PRIVATE cublasLt cublas curand cufft cudart)
  endif()
endif()

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(onnxruntime-genai PRIVATE "/CETCOMPAT")
  target_compile_options(onnxruntime-genai PRIVATE "/sdl")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(onnxruntime-genai-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()


if(USE_DML)
  list(APPEND onnxruntime_libs "${ORT_LIB_DIR}/DirectML.dll")
  list(APPEND onnxruntime_libs "${ORT_LIB_DIR}/D3D12Core.dll")
  list(APPEND ortgenai_embed_libs "${ORT_LIB_DIR}/D3D12Core.dll")
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${WIL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>/directx)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai-static PUBLIC $<TARGET_PROPERTY:${WIL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai-static PUBLIC $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>/directx)
  target_include_directories(onnxruntime-genai-static PUBLIC $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_link_libraries(onnxruntime-genai PRIVATE d3d12.lib dxcore.lib dxguid.lib dxgi.lib)
  target_link_libraries(onnxruntime-genai-static PUBLIC d3d12.lib dxcore.lib dxguid.lib dxgi.lib)

  get_filename_component(PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps ABSOLUTE)
  set(DXC_PACKAGE_DIR ${PACKAGES_DIR}/Microsoft.Direct3D.DXC.1.7.2308.12)
  set(NUGET_CONFIG ${PROJECT_SOURCE_DIR}/nuget.config)
  set(PACKAGES_CONFIG ${PROJECT_SOURCE_DIR}/packages.config)

  add_custom_command(
    OUTPUT
    ${DXC_PACKAGE_DIR}/build/native/bin/x64/dxc.exe
    DEPENDS
    ${PACKAGES_CONFIG}
    ${NUGET_CONFIG}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nuget/src/nuget restore ${PACKAGES_CONFIG} -PackagesDirectory ${PACKAGES_DIR} -ConfigFile ${NUGET_CONFIG}
    VERBATIM
  )

  add_custom_target(
    RESTORE_PACKAGES ALL
    DEPENDS
    ${DXC_PACKAGE_DIR}/build/native/bin/x64/dxc.exe
  )

  add_dependencies(RESTORE_PACKAGES nuget)
  add_dependencies(onnxruntime-genai RESTORE_PACKAGES)
  add_dependencies(onnxruntime-genai-static RESTORE_PACKAGES)
endif()

if (BUILD_APPLE_FRAMEWORK)
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
    PUBLIC_HEADER "${ONNXRUNTIME_GENAI_PUBLIC_HEADERS}"
  )

  if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    set_target_properties(onnxruntime-genai PROPERTIES
      MACOSX_RPATH TRUE
    )
  else()
    set_target_properties(onnxruntime-genai PROPERTIES INSTALL_RPATH "@loader_path")
  endif()
endif()

if(ANDROID)
  # strip the binary if it's not a build with debug info
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_RELEASE -s)
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_MINSIZEREL -s)
endif()

if(ENABLE_TESTS)
  message("------------------Enabling tests------------------")
  add_subdirectory("${REPO_ROOT}/test")
endif()

if(ENABLE_PYTHON)
  message("------------------Enabling Python Wheel------------------")
  add_subdirectory("${SRC_ROOT}/python")
endif()

if (ENABLE_JAVA)
  message("------------------Enabling Java Jar------------------")
  add_subdirectory("${SRC_ROOT}/java")
endif()


if(ENABLE_MODEL_BENCHMARK)
  message("------------------Enabling model benchmark------------------")
  add_subdirectory("${REPO_ROOT}/benchmark/c")
endif()

# Copy the onnxruntime binaries into the build folder so it's found on launch
foreach(DLL_FILE ${onnxruntime_libs})
  add_custom_command(
    TARGET onnxruntime-genai POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DLL_FILE} $<TARGET_FILE_DIR:onnxruntime-genai>
  )
endforeach()

# Assemble the Apple static framework (iOS and macOS)
if(BUILD_APPLE_FRAMEWORK)
  # when building for mac catalyst, the CMAKE_OSX_SYSROOT is set to MacOSX as well, to avoid duplication,
  # we specify as `-macabi` in the name of the output static apple framework directory.
  if (PLATFORM_NAME STREQUAL "macabi")
    set(STATIC_FRAMEWORK_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-macabi)
  else()
    set(STATIC_FRAMEWORK_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-${CMAKE_OSX_SYSROOT})
  endif()

  # Setup the various directories required. Remove any existing ones so we start with a clean directory.
  set(STATIC_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/static_libraries)
  set(STATIC_LIB_TEMP_DIR ${STATIC_LIB_DIR}/temp)
  add_custom_command(TARGET onnxruntime-genai PRE_BUILD COMMAND ${CMAKE_COMMAND} -E rm -rf ${STATIC_LIB_DIR})
  add_custom_command(TARGET onnxruntime-genai PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${STATIC_LIB_DIR})
  add_custom_command(TARGET onnxruntime-genai PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${STATIC_LIB_TEMP_DIR})

  set(STATIC_FRAMEWORK_DIR ${STATIC_FRAMEWORK_OUTPUT_DIR}/static_framework/onnxruntime-genai.framework)
  add_custom_command(TARGET onnxruntime-genai PRE_BUILD COMMAND ${CMAKE_COMMAND} -E rm -rf ${STATIC_FRAMEWORK_DIR})
  add_custom_command(TARGET onnxruntime-genai PRE_BUILD COMMAND ${CMAKE_COMMAND} -E make_directory ${STATIC_FRAMEWORK_DIR})

  set(INTERNAL_LIBRARIES)
  list(APPEND INTERNAL_LIBRARIES onnxruntime-genai-static)

  # If it's an onnxruntime library, extract .o files from the original cmake build path to a separate directory for
  # each library to avoid any clashes with filenames (e.g. utils.o)
  foreach(_LIB ${INTERNAL_LIBRARIES} )
    GET_TARGET_PROPERTY(_LIB_TYPE ${_LIB} TYPE)
    if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
      set(CUR_STATIC_LIB_OBJ_DIR ${STATIC_LIB_TEMP_DIR}/$<TARGET_LINKER_FILE_BASE_NAME:${_LIB}>)
      add_custom_command(TARGET onnxruntime-genai POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E make_directory ${CUR_STATIC_LIB_OBJ_DIR})

      add_custom_command(TARGET onnxruntime-genai POST_BUILD
      COMMAND ar ARGS -x $<TARGET_FILE:${_LIB}>
      WORKING_DIRECTORY ${CUR_STATIC_LIB_OBJ_DIR})
    endif()
  endforeach()

  # for external libraries we create a symlink to the .a file
  foreach(_LIB ${EXTERNAL_LIBRARIES})
    GET_TARGET_PROPERTY(_LIB_TYPE ${_LIB} TYPE)
    if(_LIB_TYPE STREQUAL "STATIC_LIBRARY")
      add_custom_command(TARGET onnxruntime-genai POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E create_symlink
                           $<TARGET_FILE:${_LIB}> ${STATIC_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:${_LIB}>)
    endif()
  endforeach()

  # do the pre-link with `ld -r` to create a single relocatable object with correct symbol visibility
  add_custom_command(TARGET onnxruntime-genai POST_BUILD
                     COMMAND ld ARGS -r -o ${STATIC_LIB_DIR}/prelinked_objects.o */*.o ../*.a
                     WORKING_DIRECTORY ${STATIC_LIB_TEMP_DIR})

  # create the static library
  add_custom_command(TARGET onnxruntime-genai POST_BUILD
                     COMMAND libtool -static -o ${STATIC_FRAMEWORK_DIR}/onnxruntime-genai prelinked_objects.o
                     WORKING_DIRECTORY ${STATIC_LIB_DIR})

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

# Have visual studio put all files into one single folder vs the default split of header files into a separate folder
source_group(TREE ${GENERATORS_ROOT} FILES ${generator_srcs})

include(cmake/package.cmake)

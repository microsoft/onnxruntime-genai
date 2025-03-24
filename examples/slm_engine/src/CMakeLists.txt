cmake_minimum_required(VERSION 3.28)
project(SLMEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TOPLEVEL_DIR ${CMAKE_CURRENT_LIST_DIR})
set(SYS_TARGET_ID ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})
if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin") 
    set(SYS_TARGET_ID MacOS-${CMAKE_SYSTEM_PROCESSOR})
endif()

set(ARTIFACTS ${ARTIFACTS_DIR})

get_filename_component(ARTIFACTS ${ARTIFACTS} ABSOLUTE)
message(STATUS "ARTIFACTS: ${ARTIFACTS}")

# Read the VERSION number 
file (
    STRINGS 
    ${CMAKE_CURRENT_LIST_DIR}/VERSION.txt 
    VERSION_NUMBER
    LIMIT_COUNT 1)
message(STATUS "SLM Engine VERSION: ${VERSION_NUMBER}")
add_compile_definitions(SW_VERSION_NUMBER="${VERSION_NUMBER}")
add_compile_definitions(BUILDING_SLM_ENGINE)

# Read the version number of the Gen AI library
set(REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../..)
if (EXISTS ${REPO_ROOT}/VERSION_INFO)
    file(READ "${REPO_ROOT}/VERSION_INFO" ver)
    set(ORT_GENAI_VERSION ${ver})
else()
    message(WARNING "VERSION_INFO file not found. Setting to UNKNOWN")
    set(ORT_GENAI_VERSION "UNKNOWN")  
endif()
message(STATUS "ORT-GENAI-VERSION: ${ORT_GENAI_VERSION}")
add_compile_definitions(ORT_GENAI_VERSION="${ORT_GENAI_VERSION}")

if (${CMAKE_SYSTEM_NAME} STREQUAL "Darwin") 
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(LIB_EXT "dylib")
elseif( (${CMAKE_SYSTEM_NAME} STREQUAL "Android") OR (${CMAKE_SYSTEM_NAME} STREQUAL "Linux"))
    set(LIB_EXT "so")
    if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
        set(CMAKE_INSTALL_RPATH "\$ORIGIN")
    endif()
elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(LIB_EXT "lib")
else()
    message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}")
endif()

# Setup Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if (NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    set(LIBNAME_PREFIX lib)
endif()

# Import the onnxruntime
add_library(ort SHARED IMPORTED)
set_target_properties(
    ort 
    PROPERTIES 
    IMPORTED_IMPLIB 
    ${ARTIFACTS}/${SYS_TARGET_ID}/lib/${LIBNAME_PREFIX}onnxruntime${LIB_VERSION_ORT}.${LIB_EXT})

file(GLOB ORT_LIB_FILES 
    "${ARTIFACTS}/${SYS_TARGET_ID}/lib/${LIBNAME_PREFIX}onnxruntime*")

file(GLOB ORT_LIB_DIRS "${ARTIFACTS}/${SYS_TARGET_ID}/lib"
    "${ARTIFACTS}/${SYS_TARGET_ID}/lib/${LIBNAME_PREFIX}onnxruntime*/")

list(APPEND IMPORTED_LIB_LIST ${ORT_LIB_FILES})
list(APPEND IMPORTED_LIB_LIST ${ORT_LIB_DIRS})

# Import ONNXRT-GenAI stuff
add_library(ort-genai SHARED IMPORTED)
set_target_properties(
    ort-genai 
    PROPERTIES 
    IMPORTED_IMPLIB 
    ${ARTIFACTS}/${SYS_TARGET_ID}/lib/${LIBNAME_PREFIX}onnxruntime-genai.${LIB_EXT})

list(APPEND IMPORTED_LIB_LIST 
${ARTIFACTS}/${SYS_TARGET_ID}/lib/${LIBNAME_PREFIX}onnxruntime-genai.${LIB_EXT})

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install/")

# In a loop copy all the files and directories in the list
# We cannot use "install(FILES" because the list contains both files and directories"
foreach (lib ${IMPORTED_LIB_LIST})
    get_filename_component(lib_name ${lib} NAME)
    message(STATUS "Copying ${lib_name} to install directory")
    file(COPY ${lib} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endforeach()

# Include directories
message(STATUS "Adding include directories: ${ARTIFACTS}/${SYS_TARGET_ID}/include")
include_directories(${ARTIFACTS}/${SYS_TARGET_ID}/include)
include_directories(${ARTIFACTS}/common/include)
message(STATUS "Adding include directories: ${CMAKE_CURRENT_LIST_DIR}/include")

add_subdirectory(cpp)

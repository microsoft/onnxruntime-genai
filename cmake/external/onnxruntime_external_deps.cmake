message("Loading Dependencies URLs ...")
include(FetchContent)
include(cmake/external/helper_functions.cmake)

file(STRINGS cmake/deps.txt ONNXRUNTIME_DEPS_LIST)
foreach(ONNXRUNTIME_DEP IN LISTS ONNXRUNTIME_DEPS_LIST)
  # Lines start with "#" are comments
  if(NOT ONNXRUNTIME_DEP MATCHES "^#")
    # The first column is name
    list(POP_FRONT ONNXRUNTIME_DEP ONNXRUNTIME_DEP_NAME)
    # The second column is URL
    # The URL below may be a local file path or an HTTPS URL
    list(POP_FRONT ONNXRUNTIME_DEP ONNXRUNTIME_DEP_URL)
    set(DEP_URL_${ONNXRUNTIME_DEP_NAME} ${ONNXRUNTIME_DEP_URL})
    # The third column is SHA1 hash value
    set(DEP_SHA1_${ONNXRUNTIME_DEP_NAME} ${ONNXRUNTIME_DEP})
  endif()
endforeach()

message("Loading Dependencies ...")

if(ENABLE_PYTHON)
  FetchContent_Declare(
    pybind11_project
    URL ${DEP_URL_pybind11}
    URL_HASH SHA1=${DEP_SHA1_pybind11}
    FIND_PACKAGE_ARGS 2.6 NAMES pybind11
  )
  onnxruntime_fetchcontent_makeavailable(pybind11_project)

  if(TARGET pybind11::module)
    set(pybind11_lib pybind11::module)
  else()
    set(pybind11_dep pybind11::pybind11)
  endif()
endif()

FetchContent_Declare(
  googletest
  URL ${DEP_URL_googletest}
  URL_HASH SHA1=${DEP_SHA1_googletest}
  FIND_PACKAGE_ARGS 1.14.0...<2.0.0 NAMES GTest
)

onnxruntime_fetchcontent_makeavailable(googletest)

if(USE_DML)
  set(WIL_BUILD_PACKAGING OFF CACHE BOOL "" FORCE)
  set(WIL_BUILD_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    microsoft_wil
    URL ${DEP_URL_microsoft_wil}
    URL_HASH SHA1=${DEP_SHA1_microsoft_wil}
    FIND_PACKAGE_ARGS NAMES wil
  )

  onnxruntime_fetchcontent_makeavailable(microsoft_wil)
  set(WIL_TARGET "WIL::WIL")

  FetchContent_Declare(
    directx_headers
    URL ${DEP_URL_directx_headers}
    URL_HASH SHA1=${DEP_SHA1_directx_headers}
  )

  onnxruntime_fetchcontent_makeavailable(directx_headers)
  set(DIRECTX_HEADERS_TARGET "DirectX-Headers")

  include(ExternalProject)
  ExternalProject_Add(nuget
    PREFIX nuget
    URL "https://dist.nuget.org/win-x86-commandline/v5.3.0/nuget.exe"
    DOWNLOAD_NO_EXTRACT 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND ""
  )
endif()


FetchContent_Declare(
  onnxruntime_extensions
  GIT_REPOSITORY ${DEP_URL_onnxruntime_extensions}
  GIT_TAG ${DEP_SHA1_onnxruntime_extensions}
)
set(OCOS_BUILD_PRESET ort_genai)
onnxruntime_fetchcontent_makeavailable(onnxruntime_extensions)

list(APPEND EXTERNAL_LIBRARIES
  onnxruntime_extensions
  ocos_operators
  noexcep_operators
)

if(USE_GUIDANCE)
  FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY ${DEP_URL_corrosion}
    GIT_TAG ${DEP_SHA1_corrosion}
    )
  onnxruntime_fetchcontent_makeavailable(Corrosion)
  FetchContent_Declare(
    llguidance
    GIT_REPOSITORY ${DEP_URL_llguidance}
    GIT_TAG ${DEP_SHA1_llguidance}
  )
  onnxruntime_fetchcontent_makeavailable(llguidance)
  corrosion_import_crate(MANIFEST_PATH ${llguidance_SOURCE_DIR}/parser/Cargo.toml)
endif()
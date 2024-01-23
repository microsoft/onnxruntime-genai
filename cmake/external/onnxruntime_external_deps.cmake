message("Loading Dependencies URLs ...")

include(cmake/external/helper_functions.cmake)

file(STRINGS deps.txt ONNXRUNTIME_DEPS_LIST)
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
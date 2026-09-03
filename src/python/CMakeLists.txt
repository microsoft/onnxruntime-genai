if(NOT DEFINED REPO_ROOT)
  # ---------------------------------------------------------------------------
  # Standalone (incremental) SDK build.
  #
  # This project is configured directly (cmake -S src/python) and links against a
  # prebuilt core installed via find_package(onnxruntime-genai), instead of being
  # added as a subdirectory of the top-level core build. build.py drives this via
  # `--sdk python --prebuilt_genai_home <prefix> --ort_home <ort>`.
  # ---------------------------------------------------------------------------
  cmake_minimum_required(VERSION 3.26)
  project(onnxruntime_genai_python LANGUAGES C CXX)

  get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
  set(SRC_ROOT ${REPO_ROOT}/src)

  # Read the project version the same way global_variables.cmake does in-tree so
  # the wheel (setup.py.in / __init__.py.in @VERSION_INFO@) is versioned correctly.
  file(READ "${REPO_ROOT}/VERSION_INFO" VERSION_INFO)
  string(STRIP "${VERSION_INFO}" VERSION_INFO)

  find_package(onnxruntime-genai CONFIG REQUIRED)

  # Mirror the EP flavor of the prebuilt core so the wheel is named correctly.
  set(USE_CUDA ${onnxruntime-genai_USE_CUDA})
  set(USE_TRT_RTX ${onnxruntime-genai_USE_TRT_RTX})
  set(USE_DML ${onnxruntime-genai_USE_DML})
  set(USE_WINML ${onnxruntime-genai_USE_WINML})

  if(NOT DEFINED ORT_HOME)
    message(FATAL_ERROR "ORT_HOME must be set (-DORT_HOME=...) for a standalone python SDK build.")
  endif()
  # Desktop ORT layout: headers under <ORT_HOME>/include and libs under
  # <ORT_HOME>/lib. Android/iOS use different ORT package layouts (see
  # cmake/ortlib.cmake) and are not yet supported by the standalone SDK build.
  set(ORT_HEADER_DIR ${ORT_HOME}/include)
  set(ORT_LIB_DIR ${ORT_HOME}/lib)

  include(${REPO_ROOT}/cmake/target_platform.cmake)

  # Apply the same MSVC security hardening as the top-level CMakeLists.txt so
  # standalone SDK builds match the security posture of an in-tree build.
  include(${REPO_ROOT}/cmake/msvc_security.cmake)

  if(WIN32)
    set(ONNXRUNTIME_LIB "onnxruntime.dll")
  elseif(APPLE)
    set(ONNXRUNTIME_LIB "libonnxruntime.dylib")
  else()
    set(ONNXRUNTIME_LIB "libonnxruntime.so")
  endif()

  # pybind11 (installed via pip). Locate its CMake package through the interpreter.
  # Only the Development.Module component is needed to build the extension (like the
  # in-tree pybind11_add_module path); requesting the full Development component would
  # demand an embeddable libpython, which manylinux CPython builds do not ship.
  # build.py passes -DPython_EXECUTABLE so CMake uses the exact interpreter that will
  # build the wheel (e.g. /opt/python/cp311-cp311/... in the manylinux container),
  # which is otherwise not on CMake's default search paths.
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  # The wheel-packaging custom target uses ${PYTHON_EXECUTABLE}; in-tree this is
  # provided by the parent build, so mirror it from the modern Python module here.
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
      OUTPUT_VARIABLE _pybind11_cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _pybind11_result)
    if(NOT _pybind11_result EQUAL 0)
      message(FATAL_ERROR "pybind11 not found. From the repository root, install build dependencies with `pip install -r requirements-dev.txt`.")
    endif()
    find_package(pybind11 CONFIG REQUIRED PATHS "${_pybind11_cmakedir}" NO_DEFAULT_PATH)
  endif()

  # Provide the plain `onnxruntime-genai` target name the rest of this file links against.
  if(NOT TARGET onnxruntime-genai)
    add_library(onnxruntime-genai ALIAS onnxruntime-genai::onnxruntime-genai)
  endif()

  # Gather the prebuilt core shared libraries (main + optional cuda) to embed in the wheel.
  # They were installed next to the exported CMake package (see cmake/package.cmake).
  get_filename_component(_genai_install_lib_dir "${onnxruntime-genai_DIR}/../.." ABSOLUTE)
  if(WIN32)
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.dll")
  elseif(APPLE)
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.dylib")
  else()
    file(GLOB ortgenai_embed_libs "${_genai_install_lib_dir}/*genai*.so")
  endif()

  # WinML wheels bundle the WinML SDK runtime DLLs directly — unlike CPU/CUDA
  # wheels there is no `onnxruntime` PyPI dependency to deliver them. The pipeline
  # copies the SDK's runtimes/win-<arch>/native/* into ORT_HOME/lib. On ARM64
  # Windows WinML is an OS inbox component, so only x64 wheels bundle them
  # (matches the published onnxruntime-genai-winml wheel).
  if(USE_WINML AND WIN32 AND genai_target_platform STREQUAL "x64")
    file(GLOB _winml_sdk_dlls "${ORT_LIB_DIR}/*.dll")
    list(APPEND ortgenai_embed_libs ${_winml_sdk_dlls})
  endif()
endif()

include(${REPO_ROOT}/cmake/cxx_standard.cmake)

file(GLOB python_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
find_package(Python COMPONENTS Interpreter Development)
pybind11_add_module(python ${python_srcs})
target_include_directories(python PRIVATE ${ORT_HEADER_DIR})
target_link_directories(python PRIVATE ${ORT_LIB_DIR})
target_link_libraries(python PRIVATE onnxruntime-genai)

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
  target_link_libraries(python PRIVATE ${ONNXRUNTIME_LIB})
endif()

set_target_properties(python PROPERTIES OUTPUT_NAME "onnxruntime_genai")

if(MSVC)
  # /sdl: enables additional SDL security checks (e.g. buffer overread detection).
  # The global hardening flags (/GS /guard:cf /DYNAMICBASE /CETCOMPAT /Qspectre)
  # are already applied project-wide by cmake/msvc_security.cmake above.
  target_compile_options(python PRIVATE "/sdl")
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${python_srcs})

if(BUILD_WHEEL)
  set(WHEEL_FILES_DIR "${CMAKE_BINARY_DIR}/wheel")
  message("Setting up wheel files in : ${WHEEL_FILES_DIR}")
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${REPO_ROOT}"
    RESULT_VARIABLE GENAI_COMMIT_RESULT
    OUTPUT_VARIABLE GENAI_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(NOT GENAI_COMMIT_RESULT EQUAL 0)
    set(GENAI_COMMIT "")
  endif()
  if(USE_TRT_RTX)
    set(TARGET_NAME "onnxruntime-genai-trt-rtx")
  elseif(USE_WINML)
    set(TARGET_NAME "onnxruntime-genai-winml")
  elseif(USE_CUDA)
    set(TARGET_NAME "onnxruntime-genai-cuda")
  elseif(USE_DML)
    set(TARGET_NAME "onnxruntime-genai-directml")
  else()
    set(TARGET_NAME "onnxruntime-genai")
  endif()
  set(PACKAGE_DIR_NAME "onnxruntime_genai")
  set(WHEEL_TARGET_NAME "${WHEEL_FILES_DIR}/${PACKAGE_DIR_NAME}")
  set(GENERATE_TYPE_DIR_NAME "stubgen")
  set(GENERATE_TYPE_DIR "${WHEEL_FILES_DIR}/${GENERATE_TYPE_DIR_NAME}")
  file(MAKE_DIRECTORY "${GENERATE_TYPE_DIR}")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${WHEEL_FILES_DIR}/setup.py @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in ${WHEEL_TARGET_NAME}/__init__.py @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.pyi.in ${WHEEL_TARGET_NAME}/__init__.pyi @ONLY)

  # Copy over any additional python files
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/py/" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/package_description.md" DESTINATION ${WHEEL_FILES_DIR}/)
  file(COPY "${REPO_ROOT}/ThirdPartyNotices.txt" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${REPO_ROOT}/LICENSE" DESTINATION ${WHEEL_TARGET_NAME}/)

  add_custom_command(TARGET python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${ortgenai_embed_libs} $<TARGET_FILE:python>
    ${WHEEL_TARGET_NAME}
    COMMENT "Copying files to wheel directory: ${WHEEL_TARGET_NAME}"
  )
  # pybind11-stubgen introspects the extension by importing it, so the freshly
  # built module must be runnable on the build host. That is only true when the
  # target architecture matches the host; cross-compiled wheels (e.g. arm64 built
  # on an x64 host, or a macOS x86_64 wheel built on Apple Silicon) produce a
  # module the host cannot load, which fails the packaging pipeline. Detect this
  # and skip stub generation for cross builds.
  string(TOLOWER "${CMAKE_HOST_SYSTEM_PROCESSOR}" _genai_host_processor)
  if(_genai_host_processor MATCHES "arm64|aarch64")
    set(_genai_host_platform "arm64")
  elseif(_genai_host_processor MATCHES "x86_64|amd64|x64")
    set(_genai_host_platform "x64")
  else()
    set(_genai_host_platform "${_genai_host_processor}")
  endif()

  # ORT runtime copied into the isolated stub-generation dir so the freshly built
  # extension can be imported. For CPU/CUDA this is the ORT the core was built
  # against (ORT_LIB_DIR), which matches the required API. For WinML the bundled
  # SDK ORT can be older than the API the GenAI core requires, so importing the
  # extension fails with an ORT API-version mismatch. Use the pip-installed
  # onnxruntime runtime instead (the WinML pipeline installs a matching onnxruntime
  # before the build — see sdk-python-winml-win-step.yml); copying it over the
  # isolated dir replaces the stale bundled ORT there. The shipped wheel is
  # unaffected (it still bundles the SDK ORT for end users).
  set(STUBGEN_ORT_LIB_DIR "${ORT_LIB_DIR}")
  if(USE_WINML)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import onnxruntime, os, sys; sys.stdout.write(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))"
      OUTPUT_VARIABLE _ort_python_capi_dir
      RESULT_VARIABLE _ort_python_capi_result
      ERROR_QUIET)
    if(_ort_python_capi_result EQUAL 0 AND EXISTS "${_ort_python_capi_dir}")
      set(STUBGEN_ORT_LIB_DIR "${_ort_python_capi_dir}")
      message(STATUS "WinML stub generation will use the pip-installed onnxruntime runtime at: ${_ort_python_capi_dir}")
    else()
      message(WARNING "WinML stub generation: pip-installed onnxruntime not found; falling back to the bundled SDK ORT (${ORT_LIB_DIR}). The stubgen import may fail with an ORT API-version mismatch. Install a matching onnxruntime before building.")
    endif()
  endif()

  if(genai_target_platform STREQUAL _genai_host_platform)
    add_custom_command(TARGET python POST_BUILD
      # Use a new directory to fix the onnxruntime_genai runtime dependency and avoid contamination
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${WHEEL_TARGET_NAME}" "${GENERATE_TYPE_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${STUBGEN_ORT_LIB_DIR}" "${GENERATE_TYPE_DIR}"
      # pybind11-stubgen and numpy must already be installed in the Python environment used for the
      # build (see test/python/requirements.txt).
      # --numpy-array-use-type-var makes stubgen emit valid `numpy.ndarray[Any, numpy.dtype[...]]`
      # annotations (equivalent to numpy.typing.NDArray[...]) instead of the invalid
      # `numpy.ndarray[numpy.<dtype>]` form that type checkers reject.
      COMMAND ${PYTHON_EXECUTABLE} -m pybind11_stubgen $<TARGET_FILE_BASE_NAME:python> --output-dir "${WHEEL_TARGET_NAME}" --numpy-array-use-type-var
      COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/fix_pyi.py" -i "${WHEEL_TARGET_NAME}/$<TARGET_FILE_BASE_NAME:python>.pyi"
      WORKING_DIRECTORY "${GENERATE_TYPE_DIR}"
      COMMENT "Generate type stubs for onnxruntime_genai"
    )
  else()
    message(STATUS "Skipping onnxruntime_genai type stub generation: cross-compiling for '${genai_target_platform}' on host '${_genai_host_platform}' (the built module cannot be imported on the build host).")
  endif()
  set(auditwheel_exclude_list
    "libcuda.so.1"
    "libcublas.so.11"
    "libcublas.so.12"
    "libcublas.so.13"
    "libcublasLt.so.11"
    "libcublasLt.so.12"
    "libcublasLt.so.13"
    "libcudart.so.11.0"
    "libcudart.so.12"
    "libcudart.so.13"
    "libcudnn.so.8"
    "libcudnn.so.9"
    "libcufft.so.10"
    "libcufft.so.11"
    "libcufft.so.12"
    "libcurand.so.10"
    "libnvJitLink.so.12"
    "libnvJitLink.so.13"
    "libnvrtc.so.11.2"
    "libnvrtc.so.12"
    "libnvrtc.so.13"
    "libnvrtc-builtins.so.11"
    "libnvrtc-builtins.so.12"
    "libnvrtc-builtins.so.13"
    "libnvinfer.so.8"
    "libnvinfer.so.10"
    "libnvinfer_plugin.so.8"
    "libnvinfer_plugin.so.10"
    "libnvonnxparser.so.8"
    "libnvonnxparser.so.10"
  )
  set(modified_exclude_list)
  foreach(item IN LISTS auditwheel_exclude_list)
    list(APPEND modified_exclude_list "--exclude" ${item})
  endforeach()

  # On Linux we need this so the python bindings can load our other shared libraries in the same folder
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(python PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")  # For macOS
    set_target_properties(python PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
  endif()

  if(MANYLINUX)
    # auditwheel produces a wheel tagged with the platform (e.g. linux_x86_64
    # or linux_aarch64) — use a glob that matches all Linux platform tags so
    # this target works on every supported manylinux architecture.
    add_custom_target(PyPackageBuild
      COMMAND ${PYTHON_EXECUTABLE} -m pip wheel --no-deps .
      COMMAND ${CMAKE_COMMAND} -E remove ${WHEEL_TARGET_NAME}/onnxruntime_genai.cpython-*
      COMMAND auditwheel repair onnxruntime_genai*linux_*.whl -w ${WHEEL_FILES_DIR} ${modified_exclude_list}
      WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
      COMMENT "Building wheel with MANYLINUX on ${WHEEL_FILES_DIR}"
      EXCLUDE_FROM_ALL
    )
  else()
    add_custom_target(PyPackageBuild
      COMMAND ${PYTHON_EXECUTABLE} -m pip wheel --no-deps .
      WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
      COMMENT "Building wheel on ${WHEEL_FILES_DIR}"
      EXCLUDE_FROM_ALL
    )
  endif()
endif()
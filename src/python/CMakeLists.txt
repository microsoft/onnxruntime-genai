include(${REPO_ROOT}/cmake/cxx_standard.cmake)

file(GLOB python_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

# The Python bindings are built with nanobind. In addition to producing the
# extension module, nanobind lets us emit a typed `.pyi` stub so that editors
# such as VS Code (Pylance) can resolve the generated classes and methods for
# go-to-definition, find-references, and autocompletion.
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# Locate the pip-installed nanobind (declared in test/python/requirements.txt).
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT
  RESULT_VARIABLE _nanobind_result)
if(NOT _nanobind_result EQUAL 0)
  message(FATAL_ERROR
    "nanobind was not found for '${Python_EXECUTABLE}'. Install the Python build "
    "requirements first, e.g. `pip install -r test/python/requirements.txt` "
    "(or `pip install nanobind`).")
endif()
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(python NB_STATIC ${python_srcs})
target_include_directories(python PRIVATE ${ORT_HEADER_DIR})
target_link_directories(python PRIVATE ${ORT_LIB_DIR})
target_link_libraries(python PRIVATE onnxruntime-genai)

# This project compiles with warnings-as-errors (`/WX`), but nanobind's bundled
# runtime sources (and generated binding glue) are third-party code that isn't
# held to that policy. Relax warnings-as-errors for the binding target and the
# nanobind runtime library so their warnings don't fail the build.
if(MSVC)
  target_compile_options(python PRIVATE /WX-)
  if(TARGET nanobind-static)
    target_compile_options(nanobind-static PRIVATE /WX-)
  endif()
endif()

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
  target_link_libraries(python PRIVATE ${ONNXRUNTIME_LIB})
endif()

set_target_properties(python PROPERTIES OUTPUT_NAME "onnxruntime_genai")

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(python PRIVATE "/CETCOMPAT")
  target_compile_options(python PRIVATE "/sdl" PRIVATE "/Qspectre")
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${python_srcs})

if(BUILD_WHEEL)
  set(WHEEL_FILES_DIR "${CMAKE_BINARY_DIR}/wheel")
  message("Setting up wheel files in : ${WHEEL_FILES_DIR}")
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
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${WHEEL_FILES_DIR}/setup.py @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in ${WHEEL_TARGET_NAME}/__init__.py @ONLY)

  # Copy over any additional python files
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/py/" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/package_description.md" DESTINATION ${WHEEL_FILES_DIR}/)
  file(COPY "${CMAKE_SOURCE_DIR}/ThirdPartyNotices.txt" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION ${WHEEL_TARGET_NAME}/)

  add_custom_command(TARGET python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${ortgenai_embed_libs} $<TARGET_FILE:python>
    ${WHEEL_TARGET_NAME}
    COMMENT "Copying files to wheel directory: ${WHEEL_TARGET_NAME}"
  )

  # Generate a typed stub (onnxruntime_genai.pyi) plus a py.typed marker next to
  # the compiled extension so IDEs can resolve the nanobind-generated API. The
  # stub is produced for the extension submodule and imported via the package so
  # that its __init__ runs `add_onnxruntime_dependency` first, ensuring the
  # matching onnxruntime shared library is on the DLL search path.
  nanobind_add_stub(
    onnxruntime_genai_stub
    MODULE onnxruntime_genai.onnxruntime_genai
    OUTPUT "${WHEEL_TARGET_NAME}/onnxruntime_genai.pyi"
    MARKER_FILE "${WHEEL_TARGET_NAME}/py.typed"
    PYTHON_PATH "${WHEEL_FILES_DIR}"
    DEPENDS python
  )

  set(auditwheel_exclude_list
    "libcublas.so.11"
    "libcublas.so.12"
    "libcublasLt.so.11"
    "libcublasLt.so.12"
    "libcudart.so.11.0"
    "libcudart.so.12"
    "libcudnn.so.8"
    "libcudnn.so.9"
    "libcufft.so.10"
    "libcufft.so.11"
    "libcurand.so.10"
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

  # Ensure the typed stub is generated before the wheel is packaged.
  add_dependencies(PyPackageBuild onnxruntime_genai_stub)
endif()
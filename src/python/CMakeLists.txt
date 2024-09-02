include(${REPO_ROOT}/cmake/cxx_standard.cmake)

file(GLOB python_srcs CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
pybind11_add_module(python ${python_srcs})
target_include_directories(python PRIVATE ${ORT_HEADER_DIR})
target_link_directories(python PRIVATE ${ORT_LIB_DIR})
target_link_libraries(python PRIVATE onnxruntime-genai-static)

if(NOT (CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
target_link_libraries(python PRIVATE ${ONNXRUNTIME_LIB})
endif()

set_target_properties(python PROPERTIES OUTPUT_NAME "onnxruntime_genai")

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(python PRIVATE "/CETCOMPAT")
  target_compile_options(python PRIVATE "/sdl" PRIVATE "/Qspectre")
endif()

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  cmake_policy(SET CMP0104 OLD)
  enable_language(CUDA)
  set_target_properties(python PROPERTIES LINKER_LANGUAGE CUDA)
  target_link_libraries(python PRIVATE cublas curand cudart)
endif()

# Avoid warning of Calling FetchContent_Populate(Lib) is deprecated temporarily
# TODO: find a better way to handle the header-only 3rd party deps
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
  cmake_policy(SET CMP0169 OLD)
endif()

set_target_properties(python PROPERTIES FOLDER "Sources")
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${python_srcs})

if(BUILD_WHEEL)
  set(WHEEL_FILES_DIR "${CMAKE_BINARY_DIR}/wheel")
  message("Setting up wheel files in : ${WHEEL_FILES_DIR}")
  if(USE_CUDA)
    set(TARGET_NAME "onnxruntime-genai-cuda")
  elseif(USE_ROCM)
    set(TARGET_NAME "onnxruntime-genai-rocm")
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

    "libamdhip64.so.5"
    "libamdhip64.so.6"
    "libhipblas.so.0"
    "libhipblas.so.2"
    "libhipfft.so"
    "libhipfft.so.0"
    "libhiprtc.so.5"
    "libhsa-runtime64.so.1"
    "librccl.so.1"
    "librocblas.so.3"
    "librocfft.so.0"
    "libroctracer64.so.4"
    "libMIOpen.so.1"
  )
  set(modified_exclude_list)
  foreach(item IN LISTS auditwheel_exclude_list)
    list(APPEND modified_exclude_list "--exclude" ${item})
  endforeach()

  if(MANYLINUX)
    add_custom_target(PyPackageBuild ALL
      COMMAND ${PYTHON_EXECUTABLE} -m pip wheel --no-deps .
      COMMAND ${CMAKE_COMMAND} -E remove ${WHEEL_TARGET_NAME}/onnxruntime_genai.cpython-*
      COMMAND auditwheel repair onnxruntime_genai*linux_x86_64.whl -w ${WHEEL_FILES_DIR} ${modified_exclude_list}
      WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
      COMMENT "Building wheel with MANYLINUX on ${WHEEL_FILES_DIR}"
    )
  else()
    add_custom_target(PyPackageBuild ALL
      COMMAND ${PYTHON_EXECUTABLE} -m pip wheel --no-deps .
      WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
      COMMENT "Building wheel on ${WHEEL_FILES_DIR}"
    )
  endif()
  add_dependencies(PyPackageBuild python)
endif()
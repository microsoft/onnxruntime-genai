include(${REPO_ROOT}/cmake/cxx_standard.cmake)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# nanobind is fetched via FetchContent in cmake/external/onnxruntime_external_deps.cmake

# Source files for the Python module
set(python_srcs
  python.cpp
  py_tensor.cpp
  py_config.cpp
  py_model.cpp
  py_tokenizer.cpp
  py_generator_params.cpp
  py_generator.cpp
  py_named_tensors.cpp
  py_adapters.cpp
  py_multimodal.cpp
)

# Create the Python module using nanobind
nanobind_add_module(python ${python_srcs})

# Include directories
target_include_directories(python PRIVATE 
  ${ORT_HEADER_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${onnxruntime_extensions_SOURCE_DIR}/include
  ${onnxruntime_extensions_SOURCE_DIR}/shared/api
  ${onnxruntime_extensions_SOURCE_DIR}/base
  ${CMAKE_BINARY_DIR}/_deps/nlohmann_json-src/include
)

# Link directories
target_link_directories(python PRIVATE ${ORT_LIB_DIR})

# Link against the object files directly instead of the shared library
# This gives us direct access to internal C++ APIs without going through the C API
target_link_libraries(python PRIVATE 
  onnxruntime-genai-objects
  onnxruntime_extensions
  ${ONNXRUNTIME_LIB}
)

# Link CUDA if enabled
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(python PRIVATE CUDA::cudart)
endif()

# Set output name
set_target_properties(python PROPERTIES OUTPUT_NAME "onnxruntime_genai")

# Visual Studio specific options
if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(python PRIVATE "/CETCOMPAT")
  target_compile_options(python PRIVATE "/sdl" PRIVATE "/Qspectre")
endif()

# Source grouping for IDE
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${python_srcs})

# Wheel building
if(BUILD_WHEEL)
  set(WHEEL_FILES_DIR "${CMAKE_BINARY_DIR}/wheel")
  set(PACKAGE_DIR_NAME "onnxruntime_genai")
  set(WHEEL_TARGET_NAME "${WHEEL_FILES_DIR}/${PACKAGE_DIR_NAME}")
  
  # Determine package name based on build configuration
  if(USE_CUDA)
    set(TARGET_NAME "onnxruntime-genai-cuda")
  elseif(USE_DML)
    set(TARGET_NAME "onnxruntime-genai-directml")
  elseif(USE_ROCM)
    set(TARGET_NAME "onnxruntime-genai-rocm")
  else()
    set(TARGET_NAME "onnxruntime-genai")
  endif()
  
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in ${WHEEL_FILES_DIR}/setup.py @ONLY)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in ${WHEEL_TARGET_NAME}/__init__.py @ONLY)

  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/py/" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/package_description.md" DESTINATION ${WHEEL_FILES_DIR}/)
  file(COPY "${CMAKE_SOURCE_DIR}/ThirdPartyNotices.txt" DESTINATION ${WHEEL_TARGET_NAME}/)
  file(COPY "${CMAKE_SOURCE_DIR}/LICENSE" DESTINATION ${WHEEL_TARGET_NAME}/)

  add_custom_command(TARGET python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${ortgenai_embed_libs} $<TARGET_FILE:python> ${WHEEL_TARGET_NAME}
    COMMENT "Copying files to wheel directory: ${WHEEL_TARGET_NAME}"
  )

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(python PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "$ORIGIN")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set_target_properties(python PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH "@loader_path")
  endif()

  add_custom_target(PyPackageBuild
    COMMAND ${Python_EXECUTABLE} -m pip wheel --no-deps .
    WORKING_DIRECTORY "${WHEEL_FILES_DIR}"
    COMMENT "Building wheel on ${WHEEL_FILES_DIR}"
    EXCLUDE_FROM_ALL
  )
endif()

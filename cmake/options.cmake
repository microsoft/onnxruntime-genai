include(CMakeDependentOption)

# features
option(USE_CUDA "Build with CUDA support" ON)
option(USE_TRT_RTX "Build with TensorRT-RTX support" OFF)
option(USE_DML "Build with DML support" OFF)
option(USE_WINML "Build with WinML support" OFF)
option(USE_GUIDANCE "Build with guidance support" OFF)

# When ON, genai registers the bundled CUDA plugin EP library (shipped in the
# same directory as libonnxruntime, e.g. the onnxruntime-genai-cuda package) by
# its default file name, so no caller-side registration is needed. When OFF
# (default), the caller registers the plugin library out-of-band via
# register_execution_provider_library, allowing the plugin to live in a separate
# directory and be installed/upgraded independently (mirrors the NvTensorRtRtx EP).
cmake_dependent_option(REGISTER_BUNDLED_CUDA_PLUGIN_EP
  "Auto-register the bundled CUDA plugin EP library" OFF "USE_CUDA" OFF)

# bindings
option(ENABLE_JAVA "Build the Java API." OFF)
cmake_dependent_option(PUBLISH_JAVA_MAVEN_LOCAL "Publish Java artifacts to local Maven repo" OFF "ENABLE_JAVA" ON)
option(ENABLE_PYTHON "Build the Python API." ON)
cmake_dependent_option(BUILD_WHEEL "Build the python wheel" ON "ENABLE_PYTHON" OFF)

# testing
option(ENABLE_TESTS "Enable tests" ON)
option(TEST_PHI2 "Enable tests for Phi-2" OFF)
option(TEST_QWEN_2_5 "Enable tests for Qwen-2.5 0.5B" OFF)

# performance
option(ENABLE_MODEL_BENCHMARK "Build model benchmark program" ON)

# diagnostics
option(ENABLE_TRACING "Enable recording of tracing data" OFF)

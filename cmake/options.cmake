include(CMakeDependentOption)

# features
option(USE_CUDA "Build with CUDA support" ON)
option(USE_ROCM "Build with ROCm support" ON)
option(USE_DML "Build with DML support" OFF)
option(USE_WEBGPU "Build with WEBGPU support" ON)
option(USE_GUIDANCE "Build with guidance support" ON)

# bindings
option(ENABLE_JAVA "Build the Java API." OFF)
option(ENABLE_PYTHON "Build the Python API." ON)
cmake_dependent_option(BUILD_WHEEL "Build the python wheel" ON "ENABLE_PYTHON" OFF)

# testing
option(ENABLE_TESTS "Enable tests" ON)
option(TEST_PHI2 "Enable tests for Phi2" OFF)

# performance
option(ENABLE_MODEL_BENCHMARK "Build model benchmark program" ON)

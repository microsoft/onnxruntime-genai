include(CMakeDependentOption)

option(USE_CUDA "Build with CUDA support" ON)
option(USE_DML "Build with DML support" OFF)
option(NO_TOKENIZER "Don't include the Tokenizer" OFF)
option(ENABLE_PYTHON "Build the Python API." ON)
option(ENABLE_TESTS "Enable tests" ON)
option(TEST_PHI2 "Enable tests for Phi2" OFF)
option(ENABLE_MODEL_BENCHMARK "Build model benchmark program" ON)

cmake_dependent_option(BUILD_WHEEL "Build the python wheel" ON "ENABLE_PYTHON" OFF)
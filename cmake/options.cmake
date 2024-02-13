include(CMakeDependentOption)

option(USE_CUDA "Build with CUDA support" ON)
option(USE_TOKENIZER "Build with Tokenizer support" ON)
option(ENABLE_PYTHON "Enable python buildings" ON)
option(ENABLE_TESTS "Enable tests" ON)
option(TEST_PHI2 "Enable tests for Phi2" ON)

cmake_dependent_option(BUILD_WHEEL "Build the python wheel" ON "ENABLE_PYTHON" OFF)
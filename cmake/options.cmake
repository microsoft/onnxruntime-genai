include(CMakeDependentOption)

option(USE_CUDA "Build with CUDA support" CUDAToolkit_FOUND)
option(NO_TOKENIZER "Don't include the Tokenizer" OFF)
option(ENABLE_PYTHON "Build the Python API." ON)
option(ENABLE_TESTS "Enable tests" ON)
option(TEST_PHI2 "Enable tests for Phi2" ON)

cmake_dependent_option(BUILD_WHEEL "Build the python wheel" ON "ENABLE_PYTHON" OFF)

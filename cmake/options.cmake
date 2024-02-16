include(CMakeDependentOption)

option(USE_CUDA "Build with CUDA support" ON)
option(NO_TOKENIZER "Don't include the Tokenizer" OFF)
option(ENABLE_PYTHON "Build the Python API." ON)
option(ENABLE_TESTS "Enable tests" ON)

cmake_dependent_option(BUILD_WHEEL "Build the python wheel" ON "ENABLE_PYTHON" OFF)
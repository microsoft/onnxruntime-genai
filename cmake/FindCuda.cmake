message("hello world")


# Checking if CUDA is supported
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    set (CUDA_FOUND TRUE)
endif()
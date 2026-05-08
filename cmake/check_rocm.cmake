# ROCm support has been removed.
if(USE_ROCM)
  message(FATAL_ERROR "ROCm is no longer supported. Please remove -DUSE_ROCM=ON from your CMake configuration.")
endif()
add_compile_definitions(USE_ROCM=0)
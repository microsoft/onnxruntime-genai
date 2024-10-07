
if(USE_WEBGPU)
  add_compile_definitions(USE_WEBGPU=1)
else()
  add_compile_definitions(USE_WEBGPU=0)
endif()
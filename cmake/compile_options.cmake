if(MSVC)
  # set updated value for __cplusplus macro instead of 199711L
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
endif()
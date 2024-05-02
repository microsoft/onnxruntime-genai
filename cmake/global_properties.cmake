

# support 64-bit platforms. normalize the target platform
if (MSVC)
  if (CMAKE_VS_PLATFORM_NAME)
    # Multi-platform generator
    set(genai_target_platform ${CMAKE_VS_PLATFORM_NAME})
  else()
    set(genai_target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()

  if (genai_target_platform STREQUAL "ARM64")
    # pass
  elseif (genai_target_platform STREQUAL "x64" OR 
          genai_target_platform STREQUAL "x86_64" OR 
          genai_target_platform STREQUAL "AMD64" OR 
          CMAKE_GENERATOR MATCHES "Win64")
    # normalize
    set(genai_target_platform "x64")
  else()
    message(FATAL_ERROR "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()
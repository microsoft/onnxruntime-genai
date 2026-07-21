# Normalize the target platform to x64, arm64, or powerpc. Additional architectures can be added as needed.
#
# This logic is intentionally kept in a small standalone module (rather than inline in
# global_variables.cmake) so that standalone SDK projects (e.g. src/python, src/java) built
# incrementally against a prebuilt core can reuse it without pulling in the full core build.
#
# Sets: genai_target_platform (one of: x64, arm64, powerpc)

if (MSVC)
  if (CMAKE_VS_PLATFORM_NAME)
    # cross-platform generator
    set(genai_target_platform ${CMAKE_VS_PLATFORM_NAME})
  else()
    set(genai_target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()

  if (genai_target_platform STREQUAL "arm64")
    # pass
  elseif (genai_target_platform STREQUAL "ARM64" OR
          genai_target_platform STREQUAL "ARM64EC")
    set(genai_target_platform "arm64")
  elseif (genai_target_platform STREQUAL "x64" OR
          genai_target_platform STREQUAL "x86_64" OR
          genai_target_platform STREQUAL "AMD64" OR
          CMAKE_GENERATOR MATCHES "Win64")
    set(genai_target_platform "x64")
  else()
    message(FATAL_ERROR "Unsupported architecture. CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
elseif(APPLE)
  # TODO: do we need to support CMAKE_OSX_ARCHITECTURES having multiple values?
  set(_apple_target_arch ${CMAKE_OSX_ARCHITECTURES})
  if (NOT _apple_target_arch)
    set(_apple_target_arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
  endif()

  if (_apple_target_arch STREQUAL "arm64")
    set(genai_target_platform "arm64")
  elseif (_apple_target_arch STREQUAL "x86_64")
    set(genai_target_platform "x64")
  else()
    message(FATAL_ERROR "Unsupported architecture. ${_apple_target_arch}")
  endif()
elseif(ANDROID)
  if (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    set(genai_target_platform "arm64")
  elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
    set(genai_target_platform "x64")
  else()
    message(FATAL_ERROR "Unsupported architecture. CMAKE_ANDROID_ARCH_ABI: ${CMAKE_ANDROID_ARCH_ABI}")
  endif()
else()
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
    set(genai_target_platform "arm64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
    set(genai_target_platform "arm64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
    set(genai_target_platform "x64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "powerpc")
    set(genai_target_platform "powerpc")
  else()
    message(FATAL_ERROR "Unsupported architecture. CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
endif()

# cmake/msvc_security.cmake — MSVC compiler security and warning policy.
#
# Applies the BinSkim-required security flags and project warning policy to all
# targets in the current CMake directory scope. Designed to be included at the
# top of a project() so the flags are inherited by every subsequent target.
#
# Included by:
#   - The top-level CMakeLists.txt (replacing the inline security block).
#   - Standalone SDK sub-projects (src/python, src/java) so incremental SDK
#     builds compile with the same security posture as an in-tree build.
#
# No-ops harmlessly when the generator is not MSVC.

if(NOT MSVC)
  return()
endif()

# --- /GS  Stack buffer overrun protection (stack canaries) ---
add_compile_options(
  "$<$<COMPILE_LANGUAGE:C>:/GS>"
  "$<$<COMPILE_LANGUAGE:CXX>:/GS>"
)

# --- /W4 /WX  Project warning policy ---
add_compile_options(
  "$<$<COMPILE_LANGUAGE:C>:/w15038>"
  "$<$<COMPILE_LANGUAGE:CXX>:/w15038>"
  "$<$<COMPILE_LANGUAGE:C>:/wd4100>"
  "$<$<COMPILE_LANGUAGE:CXX>:/wd4100>"
  "$<$<COMPILE_LANGUAGE:C>:/wd4819>"
  "$<$<COMPILE_LANGUAGE:CXX>:/wd4819>"
  "$<$<COMPILE_LANGUAGE:C>:/wd4996>"
  "$<$<COMPILE_LANGUAGE:CXX>:/wd4996>"
  "$<$<COMPILE_LANGUAGE:C>:/W4>"
  "$<$<COMPILE_LANGUAGE:CXX>:/W4>"
  "$<$<COMPILE_LANGUAGE:C>:/WX>"
  "$<$<COMPILE_LANGUAGE:CXX>:/WX>"
)

# MSVC 19.50+ emits C4875 for a GSL construct used by third-party code. Also
# silence the STL assertion for the deprecated coroutine header it includes.
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19.50")
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C>:/wd4875>"
    "$<$<COMPILE_LANGUAGE:CXX>:/wd4875>"
  )
  add_compile_definitions(_SILENCE_EXPERIMENTAL_COROUTINE_DEPRECATION_WARNINGS)
endif()

# --- /guard:cf  Control Flow Guard (compiler instrumentation + linker enforcement) ---
add_compile_options(
  "$<$<COMPILE_LANGUAGE:C>:/guard:cf>"
  "$<$<COMPILE_LANGUAGE:CXX>:/guard:cf>"
)
add_link_options(/guard:cf /DYNAMICBASE)

# --- /CETCOMPAT  Intel CET shadow stack ---
# Not supported on ARM64 (hardware feature absent); skip on that architecture.
# Prefer genai_target_platform (set by cmake/target_platform.cmake) when
# available, so cross-compilation scenarios are handled correctly.
# Fall back to raw CMake platform variables when the module is included before
# target_platform.cmake (e.g. early in the top-level CMakeLists.txt).
if(DEFINED genai_target_platform)
  set(_msec_plat "${genai_target_platform}")
elseif(CMAKE_VS_PLATFORM_NAME)
  set(_msec_plat "${CMAKE_VS_PLATFORM_NAME}")
else()
  set(_msec_plat "${CMAKE_SYSTEM_PROCESSOR}")
endif()
string(TOLOWER "${_msec_plat}" _msec_plat_lc)
if(NOT _msec_plat_lc MATCHES "arm64")
  add_link_options("/CETCOMPAT")
  message(STATUS "msvc_security: /CETCOMPAT enabled (${_msec_plat}).")
else()
  message(STATUS "msvc_security: /CETCOMPAT skipped (ARM64 does not support CET).")
endif()
unset(_msec_plat)
unset(_msec_plat_lc)

# --- /Qspectre  Spectre variant 1 mitigations ---
# Generator expressions restrict this to C/C++ only — CUDA nvcc does not
# accept /Qspectre and will error if it is passed to it.
add_compile_options(
  "$<$<COMPILE_LANGUAGE:C>:/Qspectre>"
  "$<$<COMPILE_LANGUAGE:CXX>:/Qspectre>"
)

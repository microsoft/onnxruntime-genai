# cmake/msvc_security.cmake — MSVC compiler and linker security hardening.
#
# Applies the BinSkim-required security flags to *all targets* in the current
# CMake directory scope. Designed to be included at the top of a project() so
# the flags are inherited by every subsequent target.
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

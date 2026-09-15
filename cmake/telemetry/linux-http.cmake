# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The embedded curl transport is supported only on Linux.")
endif()

if(TARGET CURL::libcurl)
  message(FATAL_ERROR
    "A CURL::libcurl target already exists. Linux telemetry requires its pinned static curl target "
    "so packaged binaries never depend on a system libcurl.")
endif()

# Keep dependency options local to this block. mbedTLS uses an older CMake policy level, so explicitly
# make option() honor these normal variables rather than creating or modifying parent cache entries.
block(SCOPE_FOR VARIABLES POLICIES)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)

  foreach(option IN ITEMS
      BUILD_SHARED_LIBS
      BUILD_TESTING
      ENABLE_PROGRAMS
      ENABLE_TESTING
      GEN_FILES
      UNSAFE_BUILD
      INSTALL_MBEDTLS_HEADERS
      MBEDTLS_FATAL_WARNINGS
      USE_SHARED_MBEDTLS_LIBRARY
      LINK_WITH_PTHREAD
      BUILD_CURL_EXE
      BUILD_EXAMPLES
      BUILD_LIBCURL_DOCS
      BUILD_MISC_DOCS
      ENABLE_CURL_MANUAL
      CURL_ENABLE_EXPORT_TARGET
      CURL_USE_OPENSSL
      CURL_USE_PKGCONFIG
      CURL_USE_CMAKECONFIG
      CURL_ZLIB
      CURL_BROTLI
      CURL_ZSTD
      USE_LIBIDN2
      CURL_USE_LIBPSL
      CURL_USE_LIBSSH2
      CURL_USE_LIBSSH
      CURL_USE_GSSAPI
      CURL_USE_GSASL
      USE_NGHTTP2
      USE_NGTCP2
      USE_QUICHE
      ENABLE_ARES
      ENABLE_UNIX_SOCKETS)
    set(${option} OFF)
  endforeach()

  foreach(option IN ITEMS
      BUILD_STATIC_LIBS
      USE_STATIC_MBEDTLS_LIBRARY
      DISABLE_PACKAGE_CONFIG_AND_INSTALL
      CURL_DISABLE_INSTALL
      CURL_USE_MBEDTLS
      HTTP_ONLY
      HAVE_MBEDTLS_DES_CRYPT_ECB
      CURL_DISABLE_ALTSVC
      CURL_DISABLE_HSTS
      CURL_DISABLE_COOKIES
      CURL_DISABLE_NETRC
      CURL_DISABLE_MIME
      CURL_DISABLE_DOH
      CURL_DISABLE_AWS
      CURL_DISABLE_BEARER_AUTH
      CURL_DISABLE_DIGEST_AUTH
      CURL_DISABLE_KERBEROS_AUTH
      CURL_DISABLE_NEGOTIATE_AUTH)
    set(${option} ON)
  endforeach()

  set(MBEDTLS_CONFIG_FILE "")
  set(MBEDTLS_USER_CONFIG_FILE "")
  set(CURL_CA_BUNDLE none)
  set(CURL_CA_PATH none)

  FetchContent_Declare(
    ortgenai_mbedtls
    URL ${DEP_URL_mbedtls}
    URL_HASH SHA1=${DEP_SHA1_mbedtls}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(ortgenai_mbedtls)

  foreach(target mbedtls mbedx509 mbedcrypto)
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Embedded telemetry dependency target not found: ${target}")
    endif()
  endforeach()

  # curl's FindMbedTLS module accepts target names through these variables, avoiding host discovery.
  set(MBEDTLS_INCLUDE_DIR "${ortgenai_mbedtls_SOURCE_DIR}/include")
  set(MBEDTLS_LIBRARY MbedTLS::mbedtls)
  set(MBEDX509_LIBRARY MbedTLS::mbedx509)
  set(MBEDCRYPTO_LIBRARY MbedTLS::mbedcrypto)
  set(MBEDTLS_USE_STATIC_LIBS ON)

  FetchContent_Declare(
    ortgenai_curl
    URL ${DEP_URL_curl}
    URL_HASH SHA1=${DEP_SHA1_curl}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(ortgenai_curl)

  if(NOT TARGET CURL::libcurl OR NOT TARGET libcurl_static)
    message(FATAL_ERROR "The pinned static CURL::libcurl target was not created.")
  endif()

  foreach(target mbedtls mbedx509 mbedcrypto libcurl_static)
    set_target_properties(${target} PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      C_VISIBILITY_PRESET hidden)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-ffunction-sections;-fdata-sections>)
  endforeach()
endblock()

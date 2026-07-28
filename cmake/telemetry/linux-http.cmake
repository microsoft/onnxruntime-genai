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

function(ortgenai_save_cache_variable name)
  get_property(is_set CACHE "${name}" PROPERTY TYPE SET)
  set("_ortgenai_cache_${name}_is_set" "${is_set}" PARENT_SCOPE)
  if(is_set)
    get_property(type CACHE "${name}" PROPERTY TYPE)
    get_property(help CACHE "${name}" PROPERTY HELPSTRING)
    get_property(value CACHE "${name}" PROPERTY VALUE)
    set("_ortgenai_cache_${name}_type" "${type}" PARENT_SCOPE)
    set("_ortgenai_cache_${name}_help" "${help}" PARENT_SCOPE)
    set("_ortgenai_cache_${name}_value" "${value}" PARENT_SCOPE)
  endif()
endfunction()

function(ortgenai_restore_cache_variable name)
  if(_ortgenai_cache_${name}_is_set)
    set(${name}
      "${_ortgenai_cache_${name}_value}"
      CACHE "${_ortgenai_cache_${name}_type}"
      "${_ortgenai_cache_${name}_help}"
      FORCE)
  else()
    unset(${name} CACHE)
  endif()
endfunction()

set(_ortgenai_http_cache_variables
  BUILD_SHARED_LIBS
  BUILD_STATIC_LIBS
  BUILD_TESTING
  ENABLE_PROGRAMS
  ENABLE_TESTING
  GEN_FILES
  UNSAFE_BUILD
  INSTALL_MBEDTLS_HEADERS
  MBEDTLS_FATAL_WARNINGS
  MBEDTLS_CONFIG_FILE
  MBEDTLS_USER_CONFIG_FILE
  USE_STATIC_MBEDTLS_LIBRARY
  USE_SHARED_MBEDTLS_LIBRARY
  LINK_WITH_PTHREAD
  DISABLE_PACKAGE_CONFIG_AND_INSTALL
  BUILD_CURL_EXE
  BUILD_EXAMPLES
  BUILD_LIBCURL_DOCS
  BUILD_MISC_DOCS
  ENABLE_CURL_MANUAL
  CURL_DISABLE_INSTALL
  CURL_ENABLE_EXPORT_TARGET
  CURL_USE_MBEDTLS
  CURL_USE_OPENSSL
  CURL_USE_PKGCONFIG
  CURL_USE_CMAKECONFIG
  CURL_CA_BUNDLE
  CURL_CA_PATH
  HTTP_ONLY
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
  HAVE_MBEDTLS_DES_CRYPT_ECB
  ENABLE_ARES
  ENABLE_UNIX_SOCKETS
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

foreach(_ortgenai_cache_variable IN LISTS _ortgenai_http_cache_variables)
  ortgenai_save_cache_variable(${_ortgenai_cache_variable})
endforeach()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build embedded telemetry dependencies statically" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static libcurl" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable dependency tests" FORCE)

set(ENABLE_PROGRAMS OFF CACHE BOOL "Disable mbedTLS programs" FORCE)
set(ENABLE_TESTING OFF CACHE BOOL "Disable mbedTLS tests" FORCE)
set(GEN_FILES OFF CACHE BOOL "Use generated files from the mbedTLS release archive" FORCE)
set(UNSAFE_BUILD OFF CACHE BOOL "Require secure mbedTLS configuration" FORCE)
set(INSTALL_MBEDTLS_HEADERS OFF CACHE BOOL "Keep mbedTLS headers internal" FORCE)
set(MBEDTLS_FATAL_WARNINGS OFF CACHE BOOL "Do not inherit dependency warnings as errors" FORCE)
set(MBEDTLS_CONFIG_FILE "" CACHE FILEPATH "Use the pinned mbedTLS configuration" FORCE)
set(MBEDTLS_USER_CONFIG_FILE "" CACHE FILEPATH "Do not append a host mbedTLS configuration" FORCE)
set(USE_STATIC_MBEDTLS_LIBRARY ON CACHE BOOL "Build static mbedTLS libraries" FORCE)
set(USE_SHARED_MBEDTLS_LIBRARY OFF CACHE BOOL "Disable shared mbedTLS libraries" FORCE)
set(LINK_WITH_PTHREAD OFF CACHE BOOL "Do not add an mbedTLS pthread dependency" FORCE)
set(DISABLE_PACKAGE_CONFIG_AND_INSTALL ON CACHE BOOL "Keep mbedTLS internal" FORCE)

FetchContent_Declare(
  ortgenai_mbedtls
  URL ${DEP_URL_mbedtls}
  URL_HASH SHA1=${DEP_SHA1_mbedtls}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(ortgenai_mbedtls)

foreach(_ortgenai_mbedtls_target mbedtls mbedx509 mbedcrypto)
  if(NOT TARGET ${_ortgenai_mbedtls_target})
    message(FATAL_ERROR "Embedded telemetry dependency target not found: ${_ortgenai_mbedtls_target}")
  endif()
endforeach()

# curl's FindMbedTLS module accepts target names through these variables. Supplying them directly
# prevents discovery of a host installation and preserves the static target dependency graph.
set(MBEDTLS_INCLUDE_DIR "${ortgenai_mbedtls_SOURCE_DIR}/include")
set(MBEDTLS_LIBRARY MbedTLS::mbedtls)
set(MBEDX509_LIBRARY MbedTLS::mbedx509)
set(MBEDCRYPTO_LIBRARY MbedTLS::mbedcrypto)
set(MBEDTLS_USE_STATIC_LIBS ON)

set(BUILD_CURL_EXE OFF CACHE BOOL "Disable the curl executable" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable curl examples" FORCE)
set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "Disable libcurl documentation" FORCE)
set(BUILD_MISC_DOCS OFF CACHE BOOL "Disable curl documentation" FORCE)
set(ENABLE_CURL_MANUAL OFF CACHE BOOL "Disable the curl manual" FORCE)
set(CURL_DISABLE_INSTALL ON CACHE BOOL "Keep curl internal" FORCE)
set(CURL_ENABLE_EXPORT_TARGET OFF CACHE BOOL "Do not export the internal curl target" FORCE)

set(CURL_USE_MBEDTLS ON CACHE BOOL "Use mbedTLS for HTTPS" FORCE)
set(CURL_USE_OPENSSL OFF CACHE BOOL "Do not use OpenSSL" FORCE)
set(CURL_USE_PKGCONFIG OFF CACHE BOOL "Do not discover host dependencies with pkg-config" FORCE)
set(CURL_USE_CMAKECONFIG OFF CACHE BOOL "Do not discover host dependency packages" FORCE)
set(CURL_CA_BUNDLE none CACHE STRING "Select the Linux CA bundle at runtime" FORCE)
set(CURL_CA_PATH none CACHE STRING "Select the Linux CA bundle at runtime" FORCE)

set(HTTP_ONLY ON CACHE BOOL "Build only HTTP and HTTPS support" FORCE)
set(CURL_ZLIB OFF CACHE STRING "Disable zlib" FORCE)
set(CURL_BROTLI OFF CACHE STRING "Disable brotli" FORCE)
set(CURL_ZSTD OFF CACHE STRING "Disable zstd" FORCE)
set(USE_LIBIDN2 OFF CACHE BOOL "Disable libidn2" FORCE)
set(CURL_USE_LIBPSL OFF CACHE BOOL "Disable libpsl" FORCE)
set(CURL_USE_LIBSSH2 OFF CACHE BOOL "Disable libssh2" FORCE)
set(CURL_USE_LIBSSH OFF CACHE BOOL "Disable libssh" FORCE)
set(CURL_USE_GSSAPI OFF CACHE BOOL "Disable GSSAPI" FORCE)
set(CURL_USE_GSASL OFF CACHE BOOL "Disable GSASL" FORCE)
set(USE_NGHTTP2 OFF CACHE BOOL "Disable HTTP/2" FORCE)
set(USE_NGTCP2 OFF CACHE BOOL "Disable ngtcp2" FORCE)
set(USE_QUICHE OFF CACHE BOOL "Disable quiche" FORCE)
# curl otherwise checks this with try_compile(), whose isolated project cannot see in-tree targets.
set(HAVE_MBEDTLS_DES_CRYPT_ECB ON CACHE BOOL "mbedTLS 3.6.7 provides mbedtls_des_crypt_ecb" FORCE)
set(ENABLE_ARES OFF CACHE BOOL "Use the threaded resolver instead of c-ares" FORCE)
set(ENABLE_UNIX_SOCKETS OFF CACHE BOOL "Disable Unix domain sockets" FORCE)

set(CURL_DISABLE_ALTSVC ON CACHE BOOL "Disable alt-svc" FORCE)
set(CURL_DISABLE_HSTS ON CACHE BOOL "Disable HSTS caching" FORCE)
set(CURL_DISABLE_COOKIES ON CACHE BOOL "Disable cookies" FORCE)
set(CURL_DISABLE_NETRC ON CACHE BOOL "Disable netrc" FORCE)
set(CURL_DISABLE_MIME ON CACHE BOOL "Disable MIME" FORCE)
set(CURL_DISABLE_DOH ON CACHE BOOL "Disable DNS-over-HTTPS" FORCE)
set(CURL_DISABLE_AWS ON CACHE BOOL "Disable AWS request signing" FORCE)
set(CURL_DISABLE_BEARER_AUTH ON CACHE BOOL "Disable bearer authentication" FORCE)
set(CURL_DISABLE_DIGEST_AUTH ON CACHE BOOL "Disable digest authentication" FORCE)
set(CURL_DISABLE_KERBEROS_AUTH ON CACHE BOOL "Disable Kerberos authentication" FORCE)
set(CURL_DISABLE_NEGOTIATE_AUTH ON CACHE BOOL "Disable negotiate authentication" FORCE)

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

foreach(_ortgenai_http_target mbedtls mbedx509 mbedcrypto libcurl_static)
  set_target_properties(${_ortgenai_http_target} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden)
  target_compile_options(${_ortgenai_http_target} PRIVATE
    $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-ffunction-sections;-fdata-sections>)
endforeach()

foreach(_ortgenai_cache_variable IN LISTS _ortgenai_http_cache_variables)
  ortgenai_restore_cache_variable(${_ortgenai_cache_variable})
endforeach()

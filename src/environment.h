// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <optional>
#include <string>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace Generators {

#ifdef _WIN32
namespace EnvironmentInternal {

template <typename Reader>
std::optional<std::string> ReadWindowsEnvironmentVariable(const char* name, size_t max_bytes, Reader&& read) {
  // Use the process environment rather than the CRT's potentially stale copy.
  ::SetLastError(ERROR_SUCCESS);
  DWORD required_size = read(name, nullptr, 0);
  if (required_size == 0) {
    const DWORD error = ::GetLastError();
    if (error != ERROR_SUCCESS && error != ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
    return std::string{};
  }
  if (required_size == 1) return std::string{};
  for (int attempt = 0; attempt < 3; ++attempt) {
    if (required_size - 1 > max_bytes) return std::nullopt;
    std::string value(required_size, '\0');
    ::SetLastError(ERROR_SUCCESS);
    const DWORD written = read(name, value.data(), required_size);
    // A previously populated value disappearing or becoming empty is an unstable read.
    if (written == 0) return std::nullopt;
    if (written < required_size) {
      value.resize(written);
      return value;
    }
    required_size = written;
  }
  return std::nullopt;
}

}  // namespace EnvironmentInternal
#endif

// Unset/empty values succeed; oversized, unreadable, or unstable values return nullopt.
inline std::optional<std::string> ReadEnvironmentVariable(const char* name, size_t max_bytes) {
#ifdef _WIN32
  return EnvironmentInternal::ReadWindowsEnvironmentVariable(name, max_bytes, ::GetEnvironmentVariableA);
#else
  const char* value = std::getenv(name);
  if (value == nullptr) return std::string{};
  size_t length = 0;
  while (length < max_bytes && value[length] != '\0') ++length;
  if (value[length] != '\0') return std::nullopt;
  return std::string{value, length};
#endif
}

}  // namespace Generators

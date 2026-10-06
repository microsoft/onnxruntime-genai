// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace Generators {

inline constexpr size_t kMaxEnvironmentVariableBytes = 32766;

#ifdef _WIN32
namespace EnvInternal {

template <typename Reader>
std::optional<std::string> GetWindowsEnv(const char* name, size_t max_bytes, Reader&& read) {
  // The reader returns the native size, or nullopt on an API error.
  auto required_size = read(name, nullptr, 0);
  if (!required_size) return std::nullopt;
  if (*required_size <= 1) return std::string{};
  for (int attempt = 0; attempt < 3; ++attempt) {
    if (*required_size - 1 > max_bytes) return std::nullopt;
    std::string value(*required_size, '\0');
    const auto written = read(name, value.data(), *required_size);
    // A previously populated value disappearing or becoming empty is an unstable read.
    if (!written || *written == 0) return std::nullopt;
    if (*written < *required_size) {
      value.resize(*written);
      return value;
    }
    required_size = written;
  }
  return std::nullopt;
}

}  // namespace EnvInternal
#endif

// Unset/empty values succeed; oversized, unreadable, or unstable values return nullopt.
std::optional<std::string> GetEnv(const char* name, size_t max_bytes);

// Unset/empty values return empty; reads exceeding the default limit or failing throw.
std::string GetEnv(const char* var_name);

// This overload is used to get boolean environment variables.
// "1"/"true" set true and "0"/"false" set false (case-sensitive). Unset/empty values leave
// value unchanged; other values throw std::invalid_argument.
void GetEnv(const char* var_name, bool& value);

}  // namespace Generators

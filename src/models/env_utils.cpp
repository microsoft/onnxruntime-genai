// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "env_utils.h"

#include <cstdlib>
#include <stdexcept>
#include <utility>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace Generators {

std::optional<std::string> GetEnv(const char* name, size_t max_bytes) {
#ifdef _WIN32
  // Use the process environment rather than the CRT's potentially stale copy.
  return EnvInternal::GetWindowsEnv(
      name, max_bytes,
      [](const char* variable_name, char* buffer, DWORD size) -> std::optional<DWORD> {
        ::SetLastError(ERROR_SUCCESS);
        const DWORD written = ::GetEnvironmentVariableA(variable_name, buffer, size);
        if (written == 0) {
          const DWORD error = ::GetLastError();
          if (error != ERROR_SUCCESS && error != ERROR_ENVVAR_NOT_FOUND) return std::nullopt;
        }
        return written;
      });
#else
  const char* value = std::getenv(name);
  if (value == nullptr) return std::string{};
  size_t length = 0;
  while (length < max_bytes && value[length] != '\0') ++length;
  if (value[length] != '\0') return std::nullopt;
  return std::string{value, length};
#endif
}

std::string GetEnv(const char* var_name) {
  auto value = GetEnv(var_name, kMaxEnvironmentVariableBytes);
  if (!value) {
    throw std::runtime_error("Environment variable " + std::string(var_name) +
                             " exceeds the byte limit or could not be read consistently.");
  }
  return std::move(*value);
}

void GetEnv(const char* var_name, bool& value) {
  std::string str_value = GetEnv(var_name);
  if (str_value == "1" || str_value == "true") {
    value = true;
  } else if (str_value == "0" || str_value == "false") {
    value = false;
  } else if (!str_value.empty()) {
    throw std::invalid_argument("Invalid value for environment variable " + std::string(var_name) + ": " + str_value +
                                ". Expected '1' or 'true' for true, '0' or 'false' for false.");
  }

  // Otherwise, value will not be modified.
}

}  // namespace Generators

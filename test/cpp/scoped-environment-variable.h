// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <gtest/gtest.h>

#ifdef _WIN32
#include <Windows.h>
#endif

namespace Generators::test {

class ScopedEnvironmentVariable {
 public:
  explicit ScopedEnvironmentVariable(const char* name) : name_{name} {
#ifdef _WIN32
    for (int attempt = 0; attempt < 3; ++attempt) {
      ::SetLastError(ERROR_SUCCESS);
      const DWORD required_size = ::GetEnvironmentVariableA(name, nullptr, 0);
      if (required_size == 0) {
        const DWORD error = ::GetLastError();
        if (error == ERROR_SUCCESS)
          original_value_ = std::string{};
        else if (error != ERROR_ENVVAR_NOT_FOUND)
          break;
        return;
      }
      std::string value(required_size, '\0');
      ::SetLastError(ERROR_SUCCESS);
      const DWORD written = ::GetEnvironmentVariableA(name, value.data(), required_size);
      if (written == 0) {
        const DWORD error = ::GetLastError();
        if (error == ERROR_SUCCESS)
          original_value_ = std::string{};
        else if (error != ERROR_ENVVAR_NOT_FOUND)
          break;
        return;
      }
      if (written < required_size) {
        value.resize(written);
        original_value_ = std::move(value);
        return;
      }
    }
    throw std::runtime_error("Could not save environment variable " + name_);
#else
    if (const char* value = std::getenv(name)) original_value_ = value;
#endif
  }

  ScopedEnvironmentVariable(const char* name, std::optional<std::string> value)
      : ScopedEnvironmentVariable{name} {
    Set(value);
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) = delete;

  ~ScopedEnvironmentVariable() { Set(original_value_); }

  void Set(const std::optional<std::string>& value) const {
#ifdef _WIN32
    EXPECT_NE(::SetEnvironmentVariableA(name_.c_str(), value ? value->c_str() : nullptr), 0);
#else
    EXPECT_EQ(value ? setenv(name_.c_str(), value->c_str(), 1) : unsetenv(name_.c_str()), 0);
#endif
  }

 private:
  std::string name_;
  std::optional<std::string> original_value_;
};

}  // namespace Generators::test

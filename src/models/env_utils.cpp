// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "env_utils.h"

#include <stdexcept>

#if _MSC_VER
#include <Windows.h>
#endif

namespace Generators {

std::string GetEnv(const char* var_name) {
#if _MSC_VER
  // Why getenv() should be avoided on Windows:
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/getenv-wgetenv
  // Instead use the Win32 API: GetEnvironmentVariableA()

  // Max limit of an environment variable on Windows including the null-terminating character
  constexpr DWORD kBufferSize = 32767;

  // Create buffer to hold the result
  std::string buffer(kBufferSize, '\0');

  // The last argument is the size of the buffer pointed to by the lpBuffer parameter, including the null-terminating character, in characters.
  // If the function succeeds, the return value is the number of characters stored in the buffer pointed to by lpBuffer, not including the terminating null character.
  // Therefore, If the function succeeds, kBufferSize should be larger than char_count.
  auto char_count = ::GetEnvironmentVariableA(var_name, buffer.data(), kBufferSize);

  if (kBufferSize > char_count) {
    buffer.resize(char_count);
    return buffer;
  }

  return {};
#else
  const char* val = getenv(var_name);
  return val == nullptr ? "" : std::string(val);
#endif  // _MSC_VER
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

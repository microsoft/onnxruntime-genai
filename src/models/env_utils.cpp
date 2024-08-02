// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "env_utils.h"

namespace Generators {

std::string GetEnvironmentVariable(const char* var) {
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
  auto char_count = GetEnvironmentVariableA(var_name.c_str(), buffer.data(), kBufferSize);

  if (kBufferSize > char_count) {
    buffer.resize(char_count);
    return buffer;
  }

  return std::string();
#else
  const char* val = getenv(var);
  return val == nullptr ? "" : std::string(val);
#endif  // _MSC_VER
}

}  // namespace Generators
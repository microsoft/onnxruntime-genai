// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "env_utils.h"

#include <limits>
#include <stdexcept>
#include "../environment.h"

namespace Generators {

std::string GetEnv(const char* var_name) {
#ifdef _WIN32
  constexpr size_t max_bytes = 32766;
#else
  constexpr size_t max_bytes = (std::numeric_limits<size_t>::max)();
#endif
  return ReadEnvironmentVariable(var_name, max_bytes).value_or(std::string{});
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

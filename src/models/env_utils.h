// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

namespace Generators {

// Gets the environment variable value. If no environment variable is found, the result will be empty.
std::string GetEnv(const char* var_name);

// This overload is used to get boolean environment variables.
// "1"/"true" set true and "0"/"false" set false (case-sensitive). Unset/empty values leave
// value unchanged; other values throw std::invalid_argument.
void GetEnv(const char* var_name, bool& value);

}  // namespace Generators

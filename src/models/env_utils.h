// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>

namespace Generators {

std::string GetEnvironmentVariable(const char* var_name);

// This overload is used to get boolean environment variables.
// If the environment variable is set to "1" or "true" (case-sensitive), value will be set to true.
// Otherwise, value will not be modified.
void GetEnvironmentVariable(const char* var_name, bool& value);

}  // namespace Generators

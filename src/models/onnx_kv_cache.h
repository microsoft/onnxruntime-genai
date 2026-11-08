// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <filesystem>
#include <istream>
#include <string>
#include <utility>
#include <vector>

namespace Generators {

using SharedKvBindings = std::vector<std::pair<std::string, std::string>>;

// Only direct GQA past/present bindings with no other cache consumers qualify.
// Reads graph/node metadata while skipping initializer payloads. Unknown or
// malformed metadata returns no bindings and retains the existing dynamic policy.
SharedKvBindings ReadOnnxSharedKvBindings(std::istream& stream, uint64_t size);
SharedKvBindings ReadOnnxSharedKvBindings(const std::filesystem::path& path);

}  // namespace Generators

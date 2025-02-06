// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <sstream>

namespace Generators {

template <typename... Args>
inline std::string MakeString(Args&&... args) {
  std::ostringstream s;
  (s << ... << std::forward<Args>(args));
  return s.str();
}

}  // namespace Generators

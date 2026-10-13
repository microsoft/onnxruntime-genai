// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <istream>
#include "telemetry_string.h"

namespace Generators::TelemetryInternal {

inline std::string ReadTelemetryInput(std::istream& input) {
  std::string contents(kMaxTelemetryInputBytes, '\0');
  input.read(contents.data(), static_cast<std::streamsize>(contents.size()));
  contents.resize(static_cast<size_t>(input.gcount()));
  return contents;
}

inline std::string CpuModelFromTelemetryInput(std::string_view contents) {
  contents = contents.substr(0, kMaxTelemetryInputBytes);
  while (!contents.empty()) {
    const size_t end = contents.find('\n');
    const auto line = contents.substr(0, end);
    if (line.substr(0, 10) == "model name") {
      const size_t colon = line.find(':');
      if (colon != std::string_view::npos) {
        auto name = line.substr(colon + 1);
        const size_t start = name.find_first_not_of(" \t");
        if (start != std::string_view::npos) name.remove_prefix(start);
        return BoundTelemetryString(name);
      }
    }
    if (end == std::string_view::npos) break;
    contents.remove_prefix(end + 1);
  }
  return "unknown";
}

}  // namespace Generators::TelemetryInternal

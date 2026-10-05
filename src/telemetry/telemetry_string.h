// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>

namespace Generators {

inline constexpr size_t kMaxTelemetryStringLength = 1024;
inline constexpr size_t kMaxTelemetryInputBytes = 16 * 1024;

inline std::string_view BoundedTelemetryCString(const char* value,
                                                size_t limit = kMaxTelemetryInputBytes + 1) {
  if (value == nullptr) return {};
  size_t length = 0;
  while (length < limit && value[length] != '\0') ++length;
  return {value, length};
}

// Invalid UTF-8 bytes become '?'; a valid codepoint that cannot fit is omitted in full.
inline size_t AppendTelemetryString(std::string& output, std::string_view value,
                                    size_t limit = kMaxTelemetryStringLength) {
  size_t offset = 0;
  while (offset < value.size() && output.size() < limit) {
    const auto lead = static_cast<unsigned char>(value[offset]);
    const size_t width = lead < 0x80 ? 1 : lead >= 0xC2 && lead <= 0xDF ? 2
                                       : lead >= 0xE0 && lead <= 0xEF   ? 3
                                       : lead >= 0xF0 && lead <= 0xF4   ? 4
                                                                        : 0;
    bool valid = width != 0 && width <= value.size() - offset;
    for (size_t i = 1; valid && i < width; ++i) {
      const auto byte = static_cast<unsigned char>(value[offset + i]);
      valid = (byte & 0xC0) == 0x80;
      if (i == 1) {
        valid = valid && !(lead == 0xE0 && byte < 0xA0) &&
                !(lead == 0xED && byte >= 0xA0) &&
                !(lead == 0xF0 && byte < 0x90) &&
                !(lead == 0xF4 && byte >= 0x90);
      }
    }
    if (!valid) {
      output += '?';
      ++offset;
    } else {
      if (width > limit - output.size()) break;
      output.append(value.data() + offset, width);
      offset += width;
    }
  }
  return offset;
}

inline std::string BoundTelemetryString(std::string_view value,
                                        size_t limit = kMaxTelemetryStringLength) {
  std::string result;
  result.reserve((std::min)(value.size(), limit));
  AppendTelemetryString(result, value, limit);
  return result;
}

template <typename Range>
std::string JoinTelemetryStrings(const Range& values) {
  std::string result;
  size_t count = 0;
  for (const auto& value : values) {
    // Also bound iteration for a configuration containing arbitrarily many empty entries.
    if (++count > kMaxTelemetryStringLength || result.size() == kMaxTelemetryStringLength) break;
    if (!result.empty()) result += ',';
    if (AppendTelemetryString(result, value) != value.size()) break;
  }
  return result;
}

template <typename Event>
void SetTelemetryStringProperty(Event& event, const char* name, std::string_view value) {
  event.SetProperty(name, BoundTelemetryString(value));
}

}  // namespace Generators

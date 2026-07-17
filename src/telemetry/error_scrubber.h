// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cctype>
#include <cstddef>
#include <string>

namespace Generators::TelemetryInternal {

// Returns the index of the first filesystem-path anchor in s, or npos. Anchors:
// a drive prefix (C:\ or C:/), a UNC prefix (\\), a home prefix (~/ or ~\), a relative
// Windows path with >= 2 '\'-delimited segments (Users\jane\...), or a POSIX path with
// >= 2 '/'-delimited segments (/a/b...). Single-separator tokens such as "n/a",
// "read/write", or "domain\user" are not anchors.
inline size_t FindPathAnchor(const std::string& s) {
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '\\' && i + 1 < s.size() && s[i + 1] == '\\') return i;  // UNC
    if (c == '~' && i + 1 < s.size() && (s[i + 1] == '/' || s[i + 1] == '\\')) return i;
    if (std::isalpha(static_cast<unsigned char>(c)) && i + 2 < s.size() &&
        s[i + 1] == ':' && (s[i + 2] == '\\' || s[i + 2] == '/'))
      return i;  // drive prefix
    if (c == '\\') {
      // Relative Windows path with at least two '\' separators (e.g.
      // Users\First Last\model.onnx). Anchor at the beginning of the token
      // before the first separator so drive-less paths do not leave a prefix
      // such as "Users" unredacted. Spaces are allowed because Windows user
      // profile directories often contain them.
      size_t start = i;
      while (start > 0) {
        const unsigned char prev = static_cast<unsigned char>(s[start - 1]);
        if (std::isspace(prev) || s[start - 1] == '"' || s[start - 1] == '\'') break;
        --start;
      }
      size_t separators = 0;
      for (size_t j = i; j < s.size() && s[j] != '\r' && s[j] != '\n'; ++j) {
        if (s[j] == '\\' && ++separators >= 2) return start;
      }
    }
    if (c == '/') {
      // Absolute POSIX path with >= 2 '/'-delimited non-empty segments. Segments
      // may contain spaces (e.g. "/home/John Smith"), which is why the whole run
      // is treated as a path.
      size_t segments = 0;
      size_t j = i;
      while (j < s.size() && s[j] == '/') {
        size_t seg_start = ++j;
        // A path segment for detection is non-empty and space-free; this avoids
        // treating unrelated slashes (e.g. "n/a ... read/write") as a path while
        // still detecting real paths like "/home/jdoe" or "/Users/Jane Doe"
        // (whose spaced tail is removed by the to-end-of-message redaction).
        while (j < s.size() && s[j] != '/' && s[j] != '\r' && s[j] != '\n' &&
               s[j] != ' ' && s[j] != '\t')
          ++j;
        if (j > seg_start)
          ++segments;
        else
          break;
      }
      if (segments >= 2) return i;
    }
  }
  return std::string::npos;
}

// Redact filesystem paths from free-text error strings before transmission.
inline std::string ScrubErrorMessage(const std::string& msg) {
  size_t anchor = FindPathAnchor(msg);
  if (anchor == std::string::npos) return msg;
  return msg.substr(0, anchor) + "[path]";
}

inline std::string TruncateUtf8AtBoundary(std::string value, size_t max_length) {
  if (value.size() <= max_length) return value;

  size_t end = max_length;
  while (end > 0 && (static_cast<unsigned char>(value[end]) & 0xC0) == 0x80) {
    --end;
  }
  value.resize(end);
  return value;
}

}  // namespace Generators::TelemetryInternal

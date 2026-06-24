// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cctype>
#include <stdexcept>
#include <string>

namespace Generators {

// Validate that a config-specified filename/path does not escape the model directory
// via absolute paths, Windows drive/UNC roots, or path traversal components.
//
// `context` is an optional human-readable label (e.g. "vision config_filename")
// that is prepended to error messages to make it easier to identify which
// configuration value caused the failure.
//
// This is defined as an inline function in a standalone header so that both the
// main shared library (onnxruntime-genai) and the unit test binary can use it
// without requiring a DLL symbol export.
inline void ValidateConfigPath(const std::string& path, const std::string& context = {}) {
  if (path.empty()) return;

  auto make_error = [&](const std::string& msg) {
    return context.empty() ? msg : (context + ": " + msg);
  };

  // Reject absolute paths: Unix "/" or Windows drive letters "C:" / "C:\" or UNC "\\"
  if (path[0] == '/' || path[0] == '\\') {
    throw std::runtime_error(make_error("Config path must be a relative path under the model directory, got: " + path));
  }
#ifdef _WIN32
  if (path.size() >= 2 && std::isalpha(static_cast<unsigned char>(path[0])) && path[1] == ':') {
    throw std::runtime_error(make_error("Config path must be a relative path under the model directory, got: " + path));
  }
#endif

  // Reject path traversal ".." components
  // Split on '/' and '\\' and check each component
  std::string component;
  for (size_t i = 0; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/' || path[i] == '\\') {
      if (component == "..") {
        throw std::runtime_error(make_error("Config path must not contain path traversal (..): " + path));
      }
      component.clear();
    } else {
      component += path[i];
    }
  }
}

}  // namespace Generators

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Test-only, header-only helpers for discovering and registering ONNX Runtime execution-provider
// plugin libraries at test time -- from an explicit directory (--ep_dir) and/or WinML-installed EP
// packages. Shared by the unit_tests and reinit_tests binaries. This is NOT part of the shipping
// library; it only uses the public Oga* C API to register EPs.

#pragma once

#include <array>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "ort_genai.h"

namespace test_ep {

namespace fs = std::filesystem;

// Platform-specific file-name prefix/suffix of an ORT execution provider plugin library,
// e.g. "onnxruntime_providers_webgpu.dll" or "libonnxruntime_providers_webgpu.so".
#if defined(_WIN32)
inline constexpr const char* kProviderPrefix = "";
inline constexpr const char* kProviderSuffix = ".dll";
#elif defined(__APPLE__)
inline constexpr const char* kProviderPrefix = "lib";
inline constexpr const char* kProviderSuffix = ".dylib";
#else
inline constexpr const char* kProviderPrefix = "lib";
inline constexpr const char* kProviderSuffix = ".so";
#endif

// A plugin EP genai can load at test time.
//   registration_name : arbitrary handle passed to OgaRegisterExecutionProviderLibrary. It is NOT
//                       the EP name (the ".GenAI" suffix makes that obvious); it is also the key
//                       used to dedupe an EP that ships under multiple candidate library names.
//   library_stem      : platform-independent stem; the file is "<prefix><stem><suffix>".
//   provider_name     : genai config provider name (OgaConfig::AppendProvider), or empty if none.
struct EpLibrary {
  std::string_view registration_name;
  std::string_view library_stem;
  std::string_view provider_name;
};

// A single EP may ship under more than one library file name across ORT versions (e.g. CUDA); list
// each candidate under the same registration_name. The NvTensorRtRtx library name is confirmed from
// the WinML TRT-RTX EP package.
inline constexpr std::array<EpLibrary, 5> kPluginEpLibraries = {{
    {"WebGPU.GenAI", "onnxruntime_providers_webgpu", "WebGPU"},
    {"CUDA.GenAI", "onnxruntime_providers_cuda", "cuda"},
    {"CUDA.GenAI", "onnxruntime_providers_cuda_plugin", "cuda"},
    {"OpenVINO.GenAI", "onnxruntime_providers_openvino_plugin", "OpenVINO"},
    {"NvTensorRtRtx.GenAI", "onnxruntime_providers_nv_tensorrt_rtx", "NvTensorRtRtx"},
}};

inline std::string EpLibraryFileName(std::string_view stem) {
  return std::string(kProviderPrefix) + std::string(stem) + kProviderSuffix;
}

// Discovers plugin EP libraries (from directories and/or WinML packages) and registers them with
// genai. Discovery is separated from registration so callers that tear genai down and re-initialize
// (the reinit tests) can re-register the same discovered libraries on each fresh env.
class EpRegistrar {
 public:
  // Records every known EP plugin library found under `ep_dir` (recursive, so a flat --ep_dir and
  // nested WinML MSIX package layouts both work). Deduped by registration handle.
  void DiscoverFromDirectory(const fs::path& ep_dir) {
    std::error_code ec;
    if (ep_dir.empty()) return;
    if (!fs::is_directory(ep_dir, ec)) {
      std::cerr << "Warning: EP directory '" << ep_dir.string() << "' is not a directory." << std::endl;
      return;
    }
    for (fs::recursive_directory_iterator it{ep_dir, fs::directory_options::skip_permission_denied, ec}, end;
         it != end && !ec; it.increment(ec)) {
      if (!it->is_regular_file(ec)) continue;
      const std::string fname = it->path().filename().string();
      for (const auto& ep : kPluginEpLibraries)
        if (fname == EpLibraryFileName(ep.library_stem))
          AddFound(ep, it->path());
    }
  }

#if defined(_WIN32)
  // Discovers EPs from all WinML-installed EP packages. WinML packages follow the naming convention
  // "*.EP.*" and ship the provider DLL(s) under their MSIX InstallLocation.
  void DiscoverWinML() {
    for (const auto& location : GetWinMLEpInstallLocations()) {
      std::cout << "Scanning WinML EP package: " << location.string() << std::endl;
      DiscoverFromDirectory(location);
    }
  }
#endif

  // Registers all discovered EP libraries on genai's current env. Safe to call again after
  // OgaShutdown() (re-registers on the freshly created env). Recomputes and returns the genai
  // provider names (OgaConfig::AppendProvider) that registered successfully.
  const std::vector<std::string>& RegisterAll() {
    providers_.clear();
    registered_handles_.clear();
    for (const auto& f : found_) {
      try {
        std::cout << "Registering execution provider library '" << f.registration_name << "' -> "
                  << f.path.string() << std::endl;
        OgaRegisterExecutionProviderLibrary(f.registration_name.c_str(), f.path.string().c_str());
        registered_handles_.push_back(f.registration_name);
        if (!f.provider_name.empty())
          providers_.push_back(f.provider_name);
      } catch (const std::exception& e) {
        std::cerr << "Warning: failed to register execution provider library '" << f.registration_name
                  << "': " << e.what() << std::endl;
      }
    }
    return providers_;
  }

  // True if the given registration handle registered successfully in the most recent RegisterAll().
  bool IsRegistered(std::string_view registration_name) const {
    for (const auto& h : registered_handles_)
      if (h == registration_name) return true;
    return false;
  }

  // genai provider names available from the most recent RegisterAll().
  const std::vector<std::string>& Providers() const { return providers_; }

 private:
  struct Found {
    std::string registration_name;
    std::string provider_name;
    fs::path path;
  };

  void AddFound(const EpLibrary& ep, const fs::path& path) {
    for (const auto& f : found_)
      if (f.registration_name == ep.registration_name) return;  // dedupe by handle
    found_.push_back({std::string(ep.registration_name), std::string(ep.provider_name), path});
  }

#if defined(_WIN32)
  // Queries WinML EP package install locations. The command string is a fixed literal (no external
  // input interpolated), so there is no command-injection surface.
  static std::vector<fs::path> GetWinMLEpInstallLocations() {
    std::vector<fs::path> locations;
    FILE* pipe = _popen(
        "powershell -NoProfile -NonInteractive -Command "
        "\"Get-AppxPackage '*.EP.*' | Select-Object -ExpandProperty InstallLocation\"",
        "r");
    if (!pipe) {
      std::cerr << "Warning: failed to query WinML EP packages via Get-AppxPackage." << std::endl;
      return locations;
    }
    char buffer[1024];
    while (std::fgets(buffer, sizeof(buffer), pipe)) {
      std::string line{buffer};
      while (!line.empty() && (line.back() == '\n' || line.back() == '\r' || line.back() == ' '))
        line.pop_back();
      if (!line.empty()) locations.emplace_back(line);
    }
    _pclose(pipe);
    return locations;
  }
#endif

  std::vector<Found> found_;
  std::vector<std::string> providers_;
  std::vector<std::string> registered_handles_;
};

}  // namespace test_ep

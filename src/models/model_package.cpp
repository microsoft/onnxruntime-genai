// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "model_package.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace Generators {

namespace {

// fs::path -> std::filesystem::path. The wrapper's c_str() returns the platform-correct
// UTF-16/UTF-8 form.
std::filesystem::path AsStdPath(const fs::path& p) {
  return std::filesystem::path{p.c_str()};
}

#if ORT_GENAI_HAS_MODEL_PACKAGE

std::string ToLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// Maps a user-supplied EP name to its canonical ORT form for matching against the EP
// names ORT returns from GetVariantEpName. Accepts short aliases used in genai_config.json
// (e.g. "cuda") and the full ORT names. Unknown names pass through unchanged.
std::string NormalizeEpName(const std::string& ep_name) {
  static const std::pair<const char*, const char*> kAliases[] = {
      {"cpu", "CPUExecutionProvider"},
      {"cuda", "CUDAExecutionProvider"},
      {"dml", "DmlExecutionProvider"},
      {"directml", "DmlExecutionProvider"},
      {"openvino", "OpenVINOExecutionProvider"},
      {"qnn", "QNNExecutionProvider"},
      {"nvtensorrtrtx", "NvTensorRtRtxExecutionProvider"},
      {"tensorrt", "TensorrtExecutionProvider"},
      {"webgpu", "WebGpuExecutionProvider"},
      {"vitisai", "VitisAIExecutionProvider"},
      {"ryzenai", "RyzenAIExecutionProvider"},
      {"rocm", "ROCMExecutionProvider"},
  };
  const std::string lower = ToLower(ep_name);
  for (const auto& [alias, full] : kAliases) {
    if (lower == alias || lower == ToLower(full)) {
      return full;
    }
  }
  return ep_name;
}

std::set<std::string> CollectVariantEps(const OrtModelPackageContext& pkg_ctx,
                                        const std::string& component_name) {
  std::set<std::string> result;
  for (const auto& variant : pkg_ctx.GetVariantNames(component_name.c_str())) {
    std::string ep = pkg_ctx.GetVariantEpName(component_name.c_str(), variant.c_str());
    if (!ep.empty()) {
      result.insert(std::move(ep));
    }
  }
  return result;
}

// Builds a fresh OrtSessionOptions with `ep` appended, used only by ORT to pick a variant.
// CUDA needs the dedicated V2 entry point; every other EP goes through the generic API.
std::unique_ptr<OrtSessionOptions> BuildSelectionSessionOptions(const std::string& ep) {
  auto session_options = OrtSessionOptions::Create();
  if (ep.empty() || ep == "CPUExecutionProvider") {
    return session_options;
  }
  if (ep == "CUDAExecutionProvider") {
    auto cuda_opts = OrtCUDAProviderOptionsV2::Create();
    session_options->AppendExecutionProvider_CUDA_V2(*cuda_opts);
  } else {
    session_options->AppendExecutionProvider(ep.c_str(), nullptr, nullptr, 0);
  }
  return session_options;
}

std::string OrtPathToUtf8(const std::basic_string<ORTCHAR_T>& s) {
  if (s.empty()) return {};
  std::filesystem::path p(s);
  const auto u8 = p.u8string();
  return std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
}

#endif  // ORT_GENAI_HAS_MODEL_PACKAGE

}  // namespace

bool IsModelPackage(const fs::path& path) {
  constexpr const char* kManifestFilename = "manifest.json";
  std::error_code ec;
  const std::filesystem::path root = AsStdPath(path);
  if (!std::filesystem::is_directory(root, ec) || ec) {
    return false;
  }
  return std::filesystem::is_regular_file(root / kManifestFilename, ec) && !ec;
}

#if ORT_GENAI_HAS_MODEL_PACKAGE

PackageLoadResult OpenAndSelectVariant(OrtEnv& env,
                                       const fs::path& package_root,
                                       const std::string& explicit_ep) {
  auto pkg_ctx = OrtModelPackageContext::Create(package_root.c_str());

  // Single-component restriction. The package author selects "the" genai component by being
  // the only component in the package. A future change can replace this with an explicit
  // lookup once ORT exposes per-manifest additional_metadata to consumers.
  const auto components = pkg_ctx->GetComponentNames();
  if (components.size() != 1) {
    std::ostringstream oss;
    oss << "Model package at \"" << package_root.string() << "\" declares "
        << components.size() << " components; onnxruntime-genai requires exactly one.";
    if (!components.empty()) {
      oss << " Found:";
      for (const auto& name : components) oss << " \"" << name << "\"";
    }
    throw std::runtime_error(oss.str());
  }
  const std::string& component_name = components.front();

  std::string ep;
  if (!explicit_ep.empty()) {
    ep = NormalizeEpName(explicit_ep);
  } else {
    auto variant_eps = CollectVariantEps(*pkg_ctx, component_name);
    constexpr const char* kEpHint =
        " Specify an execution provider explicitly using OgaCreateConfigFromPackageEp "
        "(C) / OgaConfig::CreateFromPackageEp(path, ep) (C++) / "
        "og.Config.from_package_ep(path, ep) (Python), then load the model from the "
        "resulting OgaConfig.";
    if (variant_eps.empty()) {
      throw std::runtime_error(
          "Model package at \"" + package_root.string() +
          "\" does not declare an execution provider for any variant of component \"" +
          component_name + "\"." + kEpHint);
    }
    if (variant_eps.size() > 1) {
      std::ostringstream oss;
      oss << "Model package at \"" << package_root.string() << "\" declares multiple execution "
          << "providers across the variants of component \"" << component_name << "\":";
      for (const auto& v : variant_eps) oss << " \"" << v << "\"";
      oss << "." << kEpHint;
      throw std::runtime_error(oss.str());
    }
    ep = *variant_eps.begin();
  }

  auto session_options = BuildSelectionSessionOptions(ep);
  auto pkg_opts = OrtModelPackageOptions::Create(env, *session_options);
  auto component_ctx = pkg_ctx->SelectComponent(component_name.c_str(), *pkg_opts);

  PackageLoadResult result;
  result.package_root = package_root;
  result.variant_dir = fs::path{OrtPathToUtf8(component_ctx->GetSelectedVariantFolderPath())};
  return result;
}

#endif  // ORT_GENAI_HAS_MODEL_PACKAGE

}  // namespace Generators

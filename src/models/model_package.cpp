// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "model_package.h"

#include <filesystem>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#include "../generators.h"
#include "session_options.h"

namespace Generators {

namespace {

// fs::path -> std::filesystem::path. The wrapper's c_str() returns the platform-correct
// UTF-16/UTF-8 form.
std::filesystem::path AsStdPath(const fs::path& p) {
  return std::filesystem::path{p.c_str()};
}

#if ORT_GENAI_HAS_MODEL_PACKAGE

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

// Builds OrtSessionOptions with `ep` registered so ORT can score and select a variant.
// Reuses SetProviderSessionOptions (via a throwaway Config) so plugin EPs and providers
// with bespoke append handling are registered just as they are for a real session.
std::unique_ptr<OrtSessionOptions> BuildSelectionSessionOptions(const std::string& ep) {
  auto session_options = OrtSessionOptions::Create();
  if (ep.empty()) {
    return session_options;  // No EP: ORT defaults to CPU.
  }
  Config config;
  SetProviderOption(config, ep, /*option_name=*/{}, /*option_value=*/{});
  SetProviderSessionOptions(*session_options,
                            config.model.decoder.session_options.providers,
                            config.model.decoder.session_options.provider_options,
                            /*is_primary_session_options=*/true, config);
  return session_options;
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

  // Single-component restriction: the package author designates "the" genai component by
  // making it the only one.
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
    ep = explicit_ep;  // Normalized downstream by SetProviderOption.
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
  result.variant_dir = fs::path{component_ctx->GetSelectedVariantFolderPath()};
  return result;
}

#endif  // ORT_GENAI_HAS_MODEL_PACKAGE

}  // namespace Generators

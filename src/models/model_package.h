// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../filesystem.h"

struct OrtEnv;
struct OrtSessionOptions;
struct OrtModelPackageContext;
struct OrtModelPackageOptions;
struct OrtModelPackageComponentContext;

namespace Generators {

struct Config;

/// Detect whether a path is a model package (contains manifest.json) or a flat directory.
bool IsModelPackage(const fs::path& path);

/// Given a model package root, intersect per-component EP sets and return the default EP.
/// Throws if ambiguous (>1 EP in intersection) or empty intersection.
std::string DefaultEpFromPackage(const OrtModelPackageContext& pkg_ctx);

/// Holds the model package state for a single model load.
/// Owns the package context, options, and per-component contexts.
struct ModelPackageState {
  ModelPackageState(const fs::path& package_root, OrtEnv& env, const OrtSessionOptions& session_options,
                   const std::string& resolved_ep_name = "CPUExecutionProvider");
  ~ModelPackageState() = default;

  // Non-copyable, movable
  ModelPackageState(const ModelPackageState&) = delete;
  ModelPackageState& operator=(const ModelPackageState&) = delete;
  ModelPackageState(ModelPackageState&&) = default;
  ModelPackageState& operator=(ModelPackageState&&) = default;

  /// Select a component by name. The component context is cached and returned.
  OrtModelPackageComponentContext* SelectComponent(const std::string& component_name);

  /// Get a previously selected component context. Returns nullptr if not selected.
  OrtModelPackageComponentContext* GetComponent(const std::string& component_name) const;

  /// Get the configs directory path (<package>/configs/)
  const fs::path& GetConfigsPath() const { return configs_path_; }

  /// Get the package root path
  const fs::path& GetPackageRoot() const { return package_root_; }

  /// Get the resolved EP name used for variant selection
  const std::string& GetResolvedEpName() const { return resolved_ep_name_; }

  /// Get the consumer_metadata overlay JSON for a component.
  /// Returns the genai_config_overlay extracted from consumer_metadata, or empty string.
  std::string GetGenAIConfigOverlay(const std::string& component_name) const;

 private:
  fs::path package_root_;
  fs::path configs_path_;
  std::string resolved_ep_name_;
  std::unique_ptr<OrtModelPackageContext> pkg_ctx_;
  std::unique_ptr<OrtModelPackageOptions> pkg_opts_;
  std::unordered_map<std::string, std::unique_ptr<OrtModelPackageComponentContext>> component_contexts_;
};

/// Apply RFC 7386 JSON Merge Patch: merge patch_json into base_json.
/// Returns the merged JSON string.
std::string JsonMergePatch(std::string_view base_json, std::string_view patch_json);

/// Map an EP name string to a DeviceInterface*.
/// Used to determine p_device_ from the resolved EP name in the package path.
DeviceInterface* DeviceFromEpName(const std::string& ep_name);

}  // namespace Generators

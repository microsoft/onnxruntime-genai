// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../filesystem.h"
#include "onnxruntime_c_api.h"

#if !defined(ORT_HAS_MODEL_PACKAGE)
#if defined(ORT_API_VERSION) && ORT_API_VERSION >= 27
#define ORT_HAS_MODEL_PACKAGE 1
#else
#define ORT_HAS_MODEL_PACKAGE 0
#endif
#endif

struct OrtEnv;
struct OrtSessionOptions;
#if ORT_HAS_MODEL_PACKAGE
struct OrtModelPackageContext;
struct OrtModelPackageOptions;
struct OrtModelPackageComponentContext;
#endif

namespace Generators {

struct Config;

/// Detect whether a path is a model package (contains manifest.json) or a flat directory.
bool IsModelPackage(const fs::path& path);

#if ORT_HAS_MODEL_PACKAGE
/// Given a model package root, intersect per-component EP sets and return the default EP.
/// If component_names is empty, all components in the package are considered.
/// Throws if ambiguous (>1 EP in intersection) or empty intersection.
std::string DefaultEpFromPackage(const OrtModelPackageContext& pkg_ctx,
                                  const std::vector<std::string>& component_names = {});

/// Open a model package and prepare it for component selection.
/// Resolves the EP (from `explicit_ep` if non-empty, else auto-detected via
/// DefaultEpFromPackage scoped to the components referenced by `base_json`), builds an
/// OrtSessionOptions with that EP appended (encapsulating the CUDA V2 special case), and
/// constructs a ModelPackageState. Returns the state; the resolved EP is written to
/// `out_resolved_ep` and is also accessible via state->GetResolvedEpName().
std::shared_ptr<ModelPackageState> OpenAndPrepareModelPackage(
    OrtEnv& env,
    const fs::path& package_root,
    std::string_view base_json,
    const std::string& explicit_ep,
    std::string& out_resolved_ep);

/// Walk the Config's referenced components and materialize variant data into the Config so
/// downstream code can treat it as a flat-dir Config:
///   - role.filename = absolute path from GetSelectedVariantFilePath(0) (single-file role) or
///     per-pipeline-element from GetSelectedVariantFilePath(i) (pipeline, positional).
///   - role.session_options (and per-pipeline-element session_options) = variant per-file SO
///     overlaid with the genai_config role SO, with back-fill to keep variant provider_options
///     that genai_config didn't override.
///   - For run_on_cpu pipeline stages, no EP is injected (CPU is implicit).
/// Throws if a referenced component is not selected or if pipeline.size() doesn't match the
/// selected variant's file count.
void NormalizePackageIntoConfig(Config& config);

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

  /// Get the selected variant's directory path for a component.
  /// Returns empty path if the component hasn't been selected.
  fs::path GetVariantDir(const std::string& component_name) const;

  /// Build a map from filename (basename) to file index for the selected variant of a component.
  /// Throws if duplicate basenames are found.
  std::unordered_map<std::string, size_t> BuildFileIndexMap(const std::string& component_name) const;

  /// Resolve a filename to its file index within the selected variant.
  /// Tries exact basename match. Throws if not found.
  size_t ResolveFileIndex(const std::string& component_name, const std::string& filename) const;

 private:
  fs::path package_root_;
  fs::path configs_path_;
  std::string resolved_ep_name_;
  std::unique_ptr<OrtModelPackageContext> pkg_ctx_;
  std::unique_ptr<OrtModelPackageOptions> pkg_opts_;
  std::unordered_map<std::string, std::unique_ptr<OrtModelPackageComponentContext>> component_contexts_;
};
#endif

/// Apply RFC 7386 JSON Merge Patch: merge patch_json into base_json.
/// Returns the merged JSON string.
std::string JsonMergePatch(std::string_view base_json, std::string_view patch_json);

/// Normalize an EP name string to its canonical ORT form.
/// Accepts short aliases (e.g. "cuda") and full names (e.g. "CUDAExecutionProvider").
/// Returns the canonical name, or the input unchanged if no mapping exists.
std::string NormalizeEpName(const std::string& ep_name);

/// Map an EP name string to a DeviceInterface*.
/// Used to determine p_device_ from the resolved EP name in the package path.
DeviceInterface* DeviceFromEpName(const std::string& ep_name);

/// Map a full ORT EP name (e.g. "CUDAExecutionProvider") to the short name
/// used in genai_config.json provider_options (e.g. "cuda").
/// Returns the input unchanged if no mapping exists.
std::string EpNameToGenAIProviderName(const std::string& ep_name);

}  // namespace Generators

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
struct ModelPackageState;  // defined below

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

/// Walk the Config's referenced components and record the selected variant's directory on
/// each role's `asset_dir`. Path resolution at session-creation time (Model::CreateSession,
/// custom_ops_library lookup, LoRA adapter loading) then uses asset_dir as the primary search
/// root, falling back to config_path when empty (matching flat-dir behavior).
/// Throws if a referenced component is not selected (overlays introducing new component
/// references are rejected).
void NormalizePackageIntoConfig(Config& config, ModelPackageState& package_state);

/// Name of the GenAI overlay file that may sit alongside the per-variant ONNX artifacts.
/// When present at `<variant_dir>/genai_config_overlay.json`, its contents are JSON-merged
/// onto the package's pre-merged genai_config.json. ORT itself does not interpret this file;
/// it is a GenAI consumer convention recorded in the v4 model-package spec.
inline constexpr std::string_view kVariantOverlayFilename = "genai_config_overlay.json";

/// Holds the lifetime state for a single model-package load. Constructed by
/// OpenAndPrepareModelPackage; consumed by component selection, genai_config_overlay
/// extraction, and NormalizePackageIntoConfig. Not stored on Config — once normalization
/// completes the state's resources can be released.
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

  /// Get the resolved EP name used for variant selection.
  const std::string& GetResolvedEpName() const { return resolved_ep_name_; }

  /// Read the variant's GenAI overlay (`<variant_dir>/genai_config_overlay.json`) for the
  /// selected component. Returns an empty string when the file does not exist (no patch).
  /// Throws if the component has not been selected or the file is unreadable.
  std::string GetGenAIConfigOverlay(const std::string& component_name) const;

 private:
  fs::path package_root_;
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
/// Map a full ORT EP name (e.g. "CUDAExecutionProvider") to the short name
/// used in genai_config.json provider_options (e.g. "cuda").
/// Returns the input unchanged if no mapping exists.
std::string EpNameToGenAIProviderName(const std::string& ep_name);

}  // namespace Generators

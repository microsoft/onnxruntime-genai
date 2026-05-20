// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../filesystem.h"
#include "../json.h"
#include "../span.h"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Generators {

// GenAI-internal abstraction over ORT's v4 model-package API. Constructed at
// Config-time and consumed by the package-vs-flat-dir branch in
// `Config::Config(path, overlay)`.
//
// Detection is conservative: missing package markers return nullptr so the
// legacy flat-directory path can load. Once a marker is present, package mode
// must use ORT's model-package API; GenAI does not keep a private package
// parser.

// One entry from a variant's `metadata.json` `ep_compatibility` list.
struct EpCompatibilityEntry {
  std::string ep;                                   // e.g. "CUDAExecutionProvider"
  std::optional<std::string> device;                // optional discriminator (OpenVINO "GPU" / "NPU")
  std::optional<std::string> compatibility_string;  // EP-owned opaque constraint string (sm_80, soc_69, ...)
};

// One entry on the caller's prioritized EP list. Mirrors the captured-EP
// shape of `OrtModelPackageOptions`: ordered, with optional device
// discriminator. The ordering is the user's preference; selection treats
// earlier entries as preferred.
struct EpSelection {
  std::string ep_name;
  std::optional<std::string> device;
};

// Inputs to `SelectComponent`. Mirrors ORT's current selector: an empty
// `ep_priority` list has no captured EP, so no variant is selected.
struct ModelPackageSelectionOptions {
  std::vector<EpSelection> ep_priority;
};

struct ComponentInstance {
  virtual ~ComponentInstance() = default;

  // Path to the variant folder for the selected variant. Per-component ONNX
  // files, LoRA adapters, and custom-ops libraries are loaded from here.
  virtual fs::path VariantFolderPath() const = 0;

  // Number of files declared in the variant's `variant.json`. Single-file
  // consumers expect 1; pipeline runners handle >= 1.
  virtual std::size_t FileCount() const = 0;

  // Opaque consumer-metadata blob extracted verbatim from the selected
  // variant's `variant.json`. Returned as a serialized JSON string (the
  // package format does not interpret it). Empty string if absent. The
  // GenAI consumer parses this and pulls out `genai_config_overlay`.
  virtual std::string ConsumerMetadata() const = 0;

  // Canonical ORT EP name (e.g. "CUDAExecutionProvider") of the variant
  // selected for this component. The string mirrors ORT's `ep` field
  // in `ep_compatibility[]`. Used to plumb the package's chosen EP
  // through the session-creation pipeline. Empty string is reserved for
  // future "no EP captured" cases — `SelectComponent` populates this from
  // the priority entry that matched the chosen variant.
  virtual std::string SelectedEp() const = 0;

  // Resolve a shared-weight checksum (as referenced by a file's
  // `shared_files` map) to its absolute path on disk. Throws on missing,
  // path-unsafe, zero-blob, or multi-blob checksum directories — the
  // package is malformed in any of those cases.
  virtual fs::path ResolveSharedWeight(std::string_view checksum) const = 0;
};

struct ModelPackageContext {
  virtual ~ModelPackageContext() = default;

  // Detect and open. Returns `nullptr` if `path` does not point at a v4
  // model package. A nullptr return is the caller's signal to fall back
  // to flat-directory mode — it is NOT an error.
  //
  // Detection rule (intentionally conservative to keep flat-dir loading
  // reliable): treat as a v4 package iff `<path>/manifest.json` exists, or
  // `<path>/metadata.json` exists for ORT's single-component-root mode. The
  // bare presence of `<path>/models/` or `<path>/configs/` is NOT a positive
  // signal — a flat-dir model could plausibly have such directories.
  //
  // Once the package is recognized, malformed content is a hard error
  // (throws). The two cases are deliberately distinct: a missing package
  // marker is "not a v4 package, try flat-dir"; a present but malformed
  // package is "broken, surface the error to the user".
  static std::unique_ptr<ModelPackageContext> Open(const fs::path& path);

  // Component traversal. Order matches `manifest.json`'s `components`
  // array when present, otherwise the discovered `models/` directory order.
  virtual std::size_t NumComponents() const = 0;
  virtual std::string ComponentName(std::size_t cix) const = 0;

  // Variant traversal order follows ORT's model-package API.
  virtual std::size_t NumVariants(std::size_t cix) const = 0;
  virtual std::string VariantName(std::size_t cix, std::size_t vix) const = 0;
  virtual std::span<const EpCompatibilityEntry> VariantEpCompatibility(
      std::size_t cix, std::size_t vix) const = 0;

  // Per-component EP compatibility, computed as the union of EP names
  // declared in each variant's `ep_compatibility[]` (first-seen order).
  // Used by Model-level EP defaulting: intersect across all components
  // to find the EPs that can load the whole package.
  virtual std::vector<std::string> EpsCompatibleWith(std::size_t cix) const = 0;

  // Pick a variant for a component using ORT's model-package selection API.
  // Returns `nullptr` when no EP priority was provided. Otherwise ORT owns
  // compatibility validation and reports selection failures as exceptions.
  virtual std::unique_ptr<ComponentInstance> SelectComponent(
      std::size_t cix, const ModelPackageSelectionOptions& options) const = 0;

  // Path to the package's shared-assets bucket: `<package>/configs/`. Houses
  // the base `genai_config.json`, tokenizer files, processor configs, and
  // the chat template. See `Config::shared_assets_path`. The directory
  // is not required to exist; existence-checking is the caller's concern.
  virtual fs::path SharedAssetsPath() const = 0;
};

// One entry of a variant's `variant.json` `files[]` array. Used by the
// multi-file pipeline runner. The session_options / provider_options values
// are scalar-stringified to match ORT's current flat option parser.
struct VariantFile {
  std::string filename;
  std::map<std::string, std::string> session_options;
  std::map<std::string, std::string> provider_options;
  // Map from filename-as-referenced-by-the-onnx-graph to the checksum of a
  // shared-weight blob. Resolve via `ComponentInstance::ResolveSharedWeight`.
  std::map<std::string, std::string> shared_files;
};

struct VariantManifest {
  std::vector<VariantFile> files;
};

// Parse `<variant_folder>/variant.json`. Standalone helper used by GenAI's
// multi-file pipeline runner. Throws on missing file or malformed JSON.
VariantManifest ParseVariantManifest(const fs::path& variant_folder);

}  // namespace Generators

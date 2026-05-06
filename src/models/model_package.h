// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../filesystem.h"
#include "../json.h"

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Generators {

// GenAI-internal abstraction over the v4 model-package layout. Constructed
// at Config-time and consumed by the package-vs-flat-dir branch in
// `Config::Config(path, overlay)` (see `feature/v4-package-w3-...`).
//
// Today the only implementation is a stub directory walker (model_package.cpp)
// that interprets the on-disk layout described in the ORT v4 model-package
// proposal. When ORT v4 lands, an alternate implementation will delegate to
// `OrtModelPackageContext`. Callers see only the abstract surface, so the
// v4 plumbing change is contained to this file pair.
//
// The abstract surface deliberately mirrors the shape of the proposed C API:
// a context that traverses components and variants, a per-component
// `SelectComponent` that takes a captured EP-priority list, and a per-instance
// accessor for the variant folder, file count, opaque consumer metadata blob,
// and a checksum-keyed shared-weight resolver. Per-file detail (filename,
// session_options, provider_options, shared_files) is intentionally NOT on
// `ComponentInstance` — consumers parse `variant.json` directly via
// `ParseVariantManifest`, matching the v4 contract that ORT does not expose
// per-file accessors.

// One entry from a variant's `metadata.json` `ep_compatibility` list.
struct EpCompatibilityEntry {
  std::string ep;                          // e.g. "CUDAExecutionProvider"
  std::optional<std::string> device;       // optional discriminator (OpenVINO "GPU" / "NPU")
  std::vector<std::string> compatibility;  // free-form constraint strings (sm_80, soc_69, ...)
};

// One entry on the caller's prioritized EP list. Mirrors the captured-EP
// shape of `OrtModelPackageOptions`: ordered, with optional device
// discriminator. The ordering is the user's preference; selection treats
// earlier entries as preferred.
struct EpSelection {
  std::string ep_name;
  std::optional<std::string> device;
};

// Inputs to `SelectComponent`. An empty `ep_priority` list is treated as
// `[{CPUExecutionProvider}]` per spec — callers that didn't append any EP
// to their session-options template should still get CPU variants.
struct ModelPackageSelectionOptions {
  std::vector<EpSelection> ep_priority;
};

struct ComponentInstance {
  virtual ~ComponentInstance() = default;

  // Path to the variant folder for the selected variant. Per-component ONNX
  // files, LoRA adapters, custom-ops libraries are loaded from here.
  virtual fs::path VariantFolderPath() const = 0;

  // Number of files declared in the variant's `variant.json`. Single-file
  // consumers expect 1; pipeline runners (W6) handle >= 1.
  virtual std::size_t FileCount() const = 0;

  // Opaque consumer-metadata blob extracted verbatim from the selected
  // variant's `variant.json`. Returned as a serialized JSON string (the
  // package format does not interpret it). Empty string if absent. The
  // GenAI consumer parses this and pulls out `genai_config_overlay`.
  virtual std::string ConsumerMetadata() const = 0;

  // Canonical ORT EP name (e.g. "CUDAExecutionProvider") of the variant
  // selected for this component. The string mirrors the spec's `ep` field
  // in `ep_compatibility[]`. Used by W5b to plumb the package's chosen EP
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
  // Detection rule (intentionally conservative to keep flat-dir fallback
  // reliable): treat as a v4 package iff EITHER
  //   * `<path>/manifest.json` exists, OR
  //   * at least one immediate non-hidden child directory of `<path>`,
  //     other than `configs/`, contains a `metadata.json`.
  // The bare presence of `<path>/configs/` is NOT a positive signal — a
  // flat-dir model could plausibly have such a directory.
  //
  // Once the package is recognized, malformed content is a hard error
  // (throws). The two cases are deliberately distinct: a missing package
  // marker is "not a v4 package, try flat-dir"; a present but malformed
  // package is "broken, surface the error to the user".
  static std::unique_ptr<ModelPackageContext> Open(const fs::path& path);

  // Component traversal. Order matches `manifest.json`'s `components`
  // array when present, otherwise lexicographic order of subdirectories.
  virtual std::size_t NumComponents() const = 0;
  virtual std::string ComponentName(std::size_t cix) const = 0;

  // Variant traversal — order is `metadata.json` declaration order (the
  // spec's tie-break for selection when the EP preference ABI declines).
  virtual std::size_t NumVariants(std::size_t cix) const = 0;
  virtual std::string VariantName(std::size_t cix, std::size_t vix) const = 0;
  virtual std::span<const EpCompatibilityEntry> VariantEpCompatibility(
      std::size_t cix, std::size_t vix) const = 0;

  // Per-component EP compatibility, computed as the union of EP names
  // declared in each variant's `ep_compatibility[]` (first-seen order).
  // Used by Model-level EP defaulting: intersect across all components
  // to find the EPs that can load the whole package.
  virtual std::vector<std::string> EpsCompatibleWith(std::size_t cix) const = 0;

  // Pick a variant for a component using the captured EP priority list.
  //
  // Algorithm (mirrors the spec's selection algorithm subset that the stub
  // backend can implement without an EP preference ABI):
  //   1. Filter variants whose `ep_compatibility[]` has any entry matching
  //      one of `options.ep_priority` by `ep_name` (and `device` if the
  //      compat entry pins one).
  //   2. Score by ordinal position of the matched EP in `ep_priority`
  //      (lower = better).
  //   3. Tie-break by `metadata.json` insertion order.
  //
  // Returns `nullptr` if no variant of this component is compatible with
  // any of the requested EPs — this is a recoverable signal callers can
  // use for diagnostic / fallback purposes (W3 EP defaulting). Throws on
  // malformed package content.
  virtual std::unique_ptr<ComponentInstance> SelectComponent(
      std::size_t cix, const ModelPackageSelectionOptions& options) const = 0;

  // Path to the package's shared-assets bucket: `<package>/configs/`. Houses
  // the base `genai_config.json`, tokenizer files, processor configs, and
  // the chat template. See `Config::shared_assets_path`. The directory
  // is not required to exist; existence-checking is the caller's concern.
  virtual fs::path SharedAssetsPath() const = 0;
};

// One entry of a variant's `variant.json` `files[]` array. Used by the
// multi-file pipeline runner (W6). The session_options / provider_options
// values are kept as a `JSON::Object` (string-keyed map of typed
// `JSON::Document`) so callers can preserve number/bool typing when
// applying them through the SetProviderSessionOptions machinery.
struct VariantFile {
  std::string filename;
  JSON::Object session_options;
  JSON::Object provider_options;
  // Map from filename-as-referenced-by-the-onnx-graph to the checksum of a
  // shared-weight blob. Resolve via `ComponentInstance::ResolveSharedWeight`.
  std::map<std::string, std::string> shared_files;
};

struct VariantManifest {
  std::vector<VariantFile> files;
};

// Parse `<variant_folder>/variant.json`. Standalone helper because the v4
// C API does not expose per-file detail through the `cix` handle: consumers
// parse `variant.json` directly. Throws on missing file or malformed JSON.
VariantManifest ParseVariantManifest(const fs::path& variant_folder);

}  // namespace Generators

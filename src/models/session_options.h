// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "../generators.h"

namespace Generators {

// Filters a list of EP devices according to the device_filtering_options specified
// in provider_options (hardware_device_id, hardware_vendor_id, hardware_device_type).
// Returns the full list unchanged if no filtering criteria are set.
// Throws std::runtime_error if criteria are set but no devices match.
std::vector<const OrtEpDevice*> ApplyDeviceFiltering(const Config::ProviderOptions& provider_options,
                                                     const std::vector<const OrtEpDevice*>& devices);

// Returns all OrtEpDevice instances whose EP name matches |ep_name|.
// Returns an empty vector if the provider is not registered as a plugin.
std::vector<const OrtEpDevice*> FindRegisteredEpDevices(const std::string& ep_name);

// Attempts to append an execution provider via the V2 plugin API.
// Discovers registered EP devices for |ep_name|, applies device filtering, and
// calls AppendExecutionProvider_V2. Returns true on success, false if the
// provider is not registered (caller should fall back to V1).
bool AppendExecutionProviderV2(OrtSessionOptions& session_options,
                               const Config::ProviderOptions& provider_options,
                               DeviceType device_type,
                               const std::string& ep_name);

// Appends an execution provider using the legacy V1 API (key/value string pairs).
void AppendExecutionProviderV1(OrtSessionOptions& session_options,
                               const Config::ProviderOptions& provider_options);

// Iterates over the requested providers, dispatches to provider-specific
// AppendExecutionProvider implementations, and returns the DeviceInterface
// for the first provider that supplies one (or nullptr if none do).
DeviceInterface* SetProviderSessionOptions(OrtSessionOptions& session_options,
                                           const std::vector<std::string>& providers,
                                           const std::vector<Config::ProviderOptions>& provider_options_list,
                                           bool is_primary_session_options,
                                           const Config& config,
                                           bool disable_graph_capture = false);

// Maps a canonical ORT EP name (e.g. "CUDAExecutionProvider", as carried in
// a v4 model package's `ep_compatibility[].ep` field) to the GenAI internal
// provider tag used in `Config::SessionOptions::providers` and the dispatch
// table inside `SetProviderSessionOptions` (e.g. "cuda"). Returns an empty
// string for "CPUExecutionProvider" (CPU is the implicit fallback path and
// never appears in `providers`), an unrecognised EP, or empty input.
std::string EpNameToProviderTag(std::string_view canonical_ep_name);

// Implicit-provider-add for v4 model packages.
//
// In package mode, `genai_config.json`'s `model.<role>.session_options`
// block is reserved for layer-2 runtime overrides — the active EP is
// determined by package variant selection and lives on
// `Config::component_instances[role]->SelectedEp()`. This helper makes
// sure that EP is the first entry of `session_options.providers` (so it
// wins priority in `SetProviderSessionOptions`'s dispatch loop) and that
// `provider_options` carries a matching entry (otherwise the dispatch
// loop would throw "Provider options not found"). Empty
// `canonical_ep_name`, "CPUExecutionProvider", and unrecognised EP names
// are no-ops — flat-dir Configs and CPU-only packages flow through
// unchanged.
//
// The helper is idempotent: re-applying with the same EP is a no-op once
// the entry is already present at the front of the list.
void EnsurePackageProvider(Config::SessionOptions& session_options,
                           std::string_view canonical_ep_name);

// Forward declaration to avoid pulling model_package.h into every translation
// unit that includes session_options.h. Definition lives in model_package.h.
struct VariantFile;

// Apply a v4-package variant.json `files[i]` per-file options block onto
// `session_options` as layer-1 defaults.
//
// Semantics (layered config; existing values in `session_options` win):
//   * `vf.session_options` (string -> string): for typed keys
//     (intra_op_num_threads, enable_cpu_mem_arena, graph_optimization_level,
//     ...) the value is filled in only when the corresponding
//     `std::optional` field on `session_options` is empty. All other keys
//     are appended to `config_entries` only when no entry with the same
//     key already exists. Unrecognised graph_optimization_level values
//     throw.
//   * `vf.provider_options` (string -> string): merged into the
//     `Config::ProviderOptions` entry whose `name` matches the GenAI tag
//     derived from `canonical_ep_name` (see `EpNameToProviderTag`). If no
//     matching entry exists, the variant provider_options are dropped:
//     a typical caller invokes `EnsurePackageProvider` first to ensure
//     the entry is present. Empty / CPU / unknown `canonical_ep_name`
//     skips the provider_options merge entirely.
//
// Used by the v4-package-aware paths in `Model::CreateSessionOptions` to
// honor `variant.json` per-file options for the decoder's main session
// and each pipeline stage.
void ApplyVariantFileDefaults(Config::SessionOptions& session_options,
                              const VariantFile& vf,
                              std::string_view canonical_ep_name);

}  // namespace Generators

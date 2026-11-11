// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <optional>
#include <string>

#include "generator/generators.h"

namespace Generators::CUDAExecutionProvider {

// Converts one BFC arena provider option (max_mem, arena_extend_strategy, initial_chunk_size_bytes,
// max_dead_bytes_per_chunk, initial_growth_chunk_size_bytes) to the integer stored in OrtArenaCfg; -1 keeps the ORT default.
// Only arena_extend_strategy accepts the names kNextPowerOfTwo (0) and kSameAsRequested (1); every other key is a byte
// count and must be an integer with nothing after it ("1GB" is rejected, not read as 1). Throws std::invalid_argument for
// a non-numeric value or trailing text and std::out_of_range below -1.
size_t ParseArenaOptionValue(std::string_view key, const std::string& value);

// The provider options that decide which GPU the shared GenAI stream is bound to before the EP is registered.
// `selected_plugin_device_id` is std::nullopt when no plugin EP device is registered, so the provider-bridge EP serves the
// session; otherwise it is the CUDA id the selected plugin device reports, which may be empty. A session that names no
// device_id is bound to that plugin device. Without a plugin device it is bound to GPU 0, the device ONNX Runtime gives
// the provider-bridge EP, rather than to the GPU the stream or the calling thread happens to use: the stream is handed to
// the EP, and a session left on another GPU would receive a stream from the wrong device. An explicit device_id is never
// replaced, and an empty one counts as omitted.
Config::ProviderOptions ResolveStreamBindingOptions(const Config::ProviderOptions& provider_options,
                                                    const std::optional<std::string>& selected_plugin_device_id);

// Writes the CUDA compute stream pointer (as a stringified integer) into a
// session config entry keyed by |config_key|. This is used by both the CUDA
// and NvTensorRtRtx providers so that the EP can share the same stream.
void AddCudaStreamConfig(OrtSessionOptions& session_options, DeviceInterface* device,
                         const std::string& config_key = "user_compute_stream");

// Registers the CUDA execution provider on |session_options|. Tries the V2
// plugin path first; falls back to the provider-bridge (CUDA V2 options) path.
DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool disable_graph_capture = false);

}  // namespace Generators::CUDAExecutionProvider

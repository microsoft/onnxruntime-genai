// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "model.h"
#include "../../models/model.h"

#include <algorithm>
#include <string_view>

namespace Generators::WebGPU {

static void DisableGraphCapture(Config& config) {
  auto disable_options = [](Config::SessionOptions& options) {
    for (auto& provider : options.provider_options) {
      if (provider.name != "WebGPU") continue;
      for (auto& [key, value] : provider.options) {
        if (key == "enableGraphCapture") value = "0";
      }
    }
  };
  auto disable_decoder = [&](Config::Model::Decoder& decoder) {
    disable_options(decoder.session_options);
    for (auto& component : decoder.pipeline) {
      if (component.session_options) disable_options(*component.session_options);
    }
  };
  disable_decoder(config.model.decoder);
  if (config.model.draft) disable_decoder(*config.model.draft);
}

static bool HasSharedKeyValueCacheShape(const Model& model) {
  const auto& decoder = model.config_->model.decoder;
  const auto input_names = model.session_info_.GetInputNames();
  int64_t common_sequence_dim = 0;
  auto compatible_ports = [&](const std::string& past, const std::string& present) {
    const auto past_index = past.find("%d");
    const auto present_index = present.find("%d");
    if (past_index == std::string::npos || present_index == std::string::npos) return false;
    const auto prefix = past.substr(0, past_index);
    const auto suffix = past.substr(past_index + 2);
    bool found = false;
    for (const auto& name : input_names) {
      if (name.size() <= prefix.size() + suffix.size() || name.compare(0, prefix.size(), prefix) != 0 ||
          name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
      const auto layer = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
      if (!std::all_of(layer.begin(), layer.end(), [](char c) { return c >= '0' && c <= '9'; })) continue;
      auto output = present;
      output.replace(present_index, 2, layer);
      if (!model.session_info_.HasOutput(output)) return false;
      const auto input_shape = model.session_info_.GetInputShape(name);
      const auto output_shape = model.session_info_.GetOutputShape(output);
      if (input_shape.size() != 4 || output_shape.size() != 4) return false;
      // Accept dynamic KV or a uniform positive fixed capacity, matching the
      // fixed-cache allocation policy. Reject zero, mixed and mismatched axes.
      const auto sequence_dim = input_shape[2] < 0 && output_shape[2] < 0 ? int64_t{-1} : input_shape[2];
      if (sequence_dim == 0 || (sequence_dim > 0 && output_shape[2] != sequence_dim) ||
          (sequence_dim < 0 && output_shape[2] >= 0) ||
          (common_sequence_dim != 0 && sequence_dim != common_sequence_dim)) return false;
      common_sequence_dim = sequence_dim;
      found = true;
    }
    return found;
  };
  return compatible_ports(decoder.inputs.past_key_names, decoder.outputs.present_key_names) &&
         compatible_ports(decoder.inputs.past_value_names, decoder.outputs.present_value_names);
}

static void SetDefaultGraphCapture(Config& config) {
  const auto& decoder = config.model.decoder;
  const auto& options = decoder.session_options;
  // Default only the shared, single-beam decoder path. Other layouts keep their
  // existing policy, and an explicit provider option always takes precedence.
  if (!ModelType::IsLLM(config.model.type) || !config.search.past_present_share_buffer || config.search.num_beams != 1 ||
      options.providers.empty() || options.providers.front() != "WebGPU" ||
      !decoder.pipeline.empty() || decoder.sliding_window || decoder.state_groups || decoder.conv_cache_size != 0 ||
      config.engine.dynamic_batching || config.model.draft || !config.model.mtp.filename.empty() ||
      !std::all_of(decoder.layer_types.begin(), decoder.layer_types.end(),
                   [](const auto& type) { return type == "full_attention"; })) return;
  for (const auto& provider : options.provider_options) {
    if (provider.name != "WebGPU") continue;
    for (const auto& [key, value] : provider.options) {
      if (key == "enableGraphCapture") return;
    }
  }
  SetProviderOption(config, "WebGPU", "enableGraphCapture", "1");
}

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, std::unique_ptr<Config> config, ModelFactory create_model) {
  SetDefaultGraphCapture(*config);
  const auto& options = config->model.decoder.session_options;
  const bool webgpu_capture = config->search.past_present_share_buffer && config->search.num_beams == 1 &&
                              IsGraphCaptureEnabled(options) &&
                              !options.providers.empty() && options.providers.front() == "WebGPU";
  // Keep the original load inputs (including in-memory assets, custom ops and
  // shared initializers). Cloning options after EP registration would retain
  // the capture-enabled EP factory and could not disable session-level capture.
  auto eager_config = webgpu_capture ? std::make_unique<Config>(*config) : nullptr;
  try {
    return create_model(ort_env, std::move(config));
  } catch (const Ort::Exception& error) {
    const std::string_view message{error.what()};
    // ORT exposes these initialization refusals as FAIL rather than a distinct
    // status code. Recognize only explicit capture incompatibility diagnostics;
    // invalid models, missing assets and device/allocation errors still propagate.
    if (!eager_config ||
        (message.find("This session cannot use the graph capture feature as requested by the user") == std::string_view::npos &&
         message.find("is not supported when graph capture is enabled") == std::string_view::npos))
      throw;
    DisableGraphCapture(*eager_config);
    // The failed Model has been destroyed and no caller generation exists. Rebuild all
    // sessions and device/input policies consistently, with one eager attempt.
    // Metadata is available only after successful loading. Preserve the original
    // capture refusal for mixed/absent KV caches outside this fallback scope.
    auto model = create_model(ort_env, std::move(eager_config));
    if (!HasSharedKeyValueCacheShape(*model)) throw;
    if (g_log.enabled && g_log.warning)
      Log("warning") << "WebGPU graph capture unavailable; using eager mode: " << message << std::endl;
    return model;
  }
}

}  // namespace Generators::WebGPU

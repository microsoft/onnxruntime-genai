// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "session_options.h"
#include "kv_cache.h"

namespace Generators {

namespace CUDAExecutionProvider {

void AppendExecutionProvider(
    OrtSessionOptions& session_options,
    const Config::ProviderOptions& provider_options,
    bool is_primary_session_options,
    DeviceInterface*& p_device,
    std::unique_ptr<OrtArenaCfg>& arena_cfg) {
  auto ort_provider_options = OrtCUDAProviderOptionsV2::Create();
  std::vector<const char*> keys, values;

  // Memory management settings
  const char* arena_keys[] = {"max_mem", "arena_extend_strategy", "initial_chunk_size_bytes", "max_dead_bytes_per_chunk", "initial_growth_chunk_size_bytes"};
  size_t arena_values[] = {static_cast<size_t>(0), static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1), static_cast<size_t>(-1)};
  bool use_arena_management = false;

  for (auto& option : provider_options.options) {
    auto it = std::find(std::begin(arena_keys), std::end(arena_keys), option.first);

    if (it == std::end(arena_keys)) {
      keys.emplace_back(option.first.c_str());
      values.emplace_back(option.second.c_str());
    } else {
      size_t idx = std::distance(std::begin(arena_keys), it);
      arena_values[idx] = static_cast<size_t>(std::stoull(option.second));
      use_arena_management = true;
    }
  }
  ort_provider_options->Update(keys.data(), values.data(), keys.size());

  // Device type determines the scoring device.
  // Only use the primary session options to determine the device type
  if (is_primary_session_options) {
    p_device = GetDeviceInterface(DeviceType::CUDA);

    // Create and set our cudaStream_t
    ort_provider_options->UpdateValue("user_compute_stream", p_device->GetCudaStream());
  }

  // Use fine-grained memory management of BFC Arena
  if (use_arena_management) {
    if (arena_cfg == nullptr) arena_cfg = OrtArenaCfg::Create(arena_keys, arena_values, 5);
    ort_provider_options->UpdateValue("default_memory_arena_cfg", arena_cfg.get());
  }

  session_options.AppendExecutionProvider_CUDA_V2(*ort_provider_options);
}

}  // namespace CUDAExecutionProvider

namespace NvTensorRtRtxExecutionProvider {

void ConfigureProfile(const Config& config, OrtSessionOptions& session_options, bool is_multi_profile_enabled) {
  // Get model parameters from decoder config
  const int num_layers = config.model.decoder.num_hidden_layers;
  const int num_kv_heads = config.model.decoder.num_key_value_heads;
  const int head_dim = config.model.decoder.head_size;
  const int batch_size = config.search.batch_size * config.search.num_beams;

  // Get max context length from config
  const int max_context_len = config.model.context_length;

  // Extract KV cache name patterns from decoder config
  std::string_view past_key_pattern = config.model.decoder.inputs.past_key_names;
  std::string_view past_value_pattern = config.model.decoder.inputs.past_value_names;

  // Helper function to add KV cache with sequence length
  const auto add_key_value_cache_shapes = [](std::ostringstream& shapes,
                                             int batch_size,
                                             std::string_view key_pattern,
                                             std::string_view value_pattern,
                                             int seq_len,
                                             int num_layers,
                                             int num_kv_heads,
                                             int head_dim) {
    for (int i = 0; i < num_layers; i++) {
      // Use the existing function to format the key/value names
      const std::string key_name = ComposeKeyValueName(std::string(key_pattern), i);
      const std::string value_name = ComposeKeyValueName(std::string(value_pattern), i);

      shapes << "," << key_name << ":" << batch_size << "x" << num_kv_heads << "x" << seq_len << "x" << head_dim;
      shapes << "," << value_name << ":" << batch_size << "x" << num_kv_heads << "x" << seq_len << "x" << head_dim;
    }
  };

  if (is_multi_profile_enabled) {
    // Multi-profile mode: existing logic for context and generation phases
    const int opt_context_len = config.model.context_length / 2;
    const int min_seq_len = 1;

    // Helper function to add input shapes (input_ids, attention_mask, position_ids)
    const auto add_input_shapes = [](std::ostringstream& shapes, int batch_size, int seq_len, bool append = false) {
      if (append) shapes << ",";
      shapes << Config::Defaults::InputIdsName << ":" << batch_size << "x" << seq_len << ","
             << Config::Defaults::AttentionMaskName << ":" << batch_size << "x" << seq_len;
    };

    // Helper function to add generation phase input shapes
    const auto add_generation_input_shapes = [](std::ostringstream& shapes, int batch_size, int context_len) {
      shapes << "," << Config::Defaults::AttentionMaskName << ":" << batch_size << "x" << context_len << ","
             << Config::Defaults::InputIdsName << ":" << batch_size << "x1";
    };

    // Helper function to add empty KV cache shapes for all layers
    const auto add_empty_key_value_cache_shapes = [](std::ostringstream& shapes,
                                                     int batch_size,
                                                     std::string_view key_pattern,
                                                     std::string_view value_pattern,
                                                     int num_layers,
                                                     int num_kv_heads,
                                                     int head_dim) {
      for (int i = 0; i < num_layers; i++) {
        // Use the existing function to format the key/value names
        const std::string key_name = ComposeKeyValueName(std::string(key_pattern), i);
        const std::string value_name = ComposeKeyValueName(std::string(value_pattern), i);

        shapes << "," << key_name << ":" << batch_size << "x" << num_kv_heads << "x0x" << head_dim;
        shapes << "," << value_name << ":" << batch_size << "x" << num_kv_heads << "x0x" << head_dim;
      }
    };

    std::ostringstream min_shapes, opt_shapes, max_shapes;

    // MIN SHAPES (context phase and first token generation)
    add_input_shapes(min_shapes, batch_size, min_seq_len);
    add_empty_key_value_cache_shapes(min_shapes, batch_size, past_key_pattern, past_value_pattern, num_layers, num_kv_heads, head_dim);
    add_generation_input_shapes(min_shapes, batch_size, min_seq_len);
    add_key_value_cache_shapes(min_shapes, batch_size, past_key_pattern, past_value_pattern, min_seq_len, num_layers, num_kv_heads, head_dim);

    // OPT SHAPES (prefill with medium context and generation after medium context)
    add_input_shapes(opt_shapes, batch_size, opt_context_len);
    add_empty_key_value_cache_shapes(opt_shapes, batch_size, past_key_pattern, past_value_pattern, num_layers, num_kv_heads, head_dim);
    add_generation_input_shapes(opt_shapes, batch_size, opt_context_len);
    add_key_value_cache_shapes(opt_shapes, batch_size, past_key_pattern, past_value_pattern, opt_context_len - 1, num_layers, num_kv_heads, head_dim);

    // MAX SHAPES (prefill with maximum context and generation after maximum context)
    add_input_shapes(max_shapes, batch_size, max_context_len);
    add_key_value_cache_shapes(max_shapes, batch_size, past_key_pattern, past_value_pattern, max_context_len - 1, num_layers, num_kv_heads, head_dim);
    add_generation_input_shapes(max_shapes, batch_size, max_context_len);
    add_key_value_cache_shapes(max_shapes, batch_size, past_key_pattern, past_value_pattern, max_context_len - 1, num_layers, num_kv_heads, head_dim);

    // Add the constructed profiles to session options
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_min_shapes", min_shapes.str().c_str());
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_opt_shapes", opt_shapes.str().c_str());
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_max_shapes", max_shapes.str().c_str());
  } else {
    // Single profile mode: simple shapes with batch_dim=[1,1,batch_size] and seq_dim=[1,1024,max_context_len]
    std::ostringstream min_shapes, opt_shapes, max_shapes;

    // MIN SHAPES: batch_dim=1, seq_dim=1
    constexpr int min_context_len = 1;
    constexpr int min_batch_size = 1;
    min_shapes << Config::Defaults::InputIdsName << ":" << min_batch_size << "x" << min_context_len << ","
               << Config::Defaults::AttentionMaskName << ":" << min_batch_size << "x" << min_context_len;
    add_key_value_cache_shapes(min_shapes, min_batch_size, past_key_pattern, past_value_pattern, 0, num_layers, num_kv_heads, head_dim);

    // OPT SHAPES: batch_dim=1, seq_dim=1024
    const int opt_context_len = std::min(max_context_len / 2, 1024);  // Use a reasonable opt context length
    constexpr int opt_batch_size = 1;                                 // Use a opt batch size of 1
    // keeping seq length to 1 as optimizing for the gen phase
    opt_shapes << Config::Defaults::InputIdsName << ":" << opt_batch_size << "x" << 1 << ","
               << Config::Defaults::AttentionMaskName << ":" << opt_batch_size << "x" << opt_context_len;
    add_key_value_cache_shapes(opt_shapes, opt_batch_size, past_key_pattern, past_value_pattern, opt_context_len, num_layers, num_kv_heads, head_dim);

    // MAX SHAPES: seq_dim=max_context_len
    max_shapes << Config::Defaults::InputIdsName << ":" << batch_size << "x" << max_context_len << ","
               << Config::Defaults::AttentionMaskName << ":" << batch_size << "x" << max_context_len;
    add_key_value_cache_shapes(max_shapes, batch_size, past_key_pattern, past_value_pattern, max_context_len, num_layers, num_kv_heads, head_dim);

    // Add the constructed profiles to session options
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_min_shapes", min_shapes.str().c_str());
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_opt_shapes", opt_shapes.str().c_str());
    session_options.AddConfigEntry("ep.nvtensorrtrtxexecutionprovider.nv_profile_max_shapes", max_shapes.str().c_str());
  }
}

}  // namespace NvTensorRtRtxExecutionProvider

}  // namespace Generators
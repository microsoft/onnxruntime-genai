// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "generator/generators.h"
#include "../model.h"
#include "conv_kv_cache.h"
#include "ep_managed_sliding_kv_cache.h"
#include "kv_cache.h"
#include "model_managed_kv_cache.h"
#include "quantized_regular_kv_cache.h"
#include "regular_kv_cache.h"
#include "shared_kv_cache.h"
#include "static_kv_cache.h"
#include "windowed_kv_cache.h"
#include <algorithm>

namespace Generators {

std::string ComposeKeyValueName(const std::string& template_string, int index,
                                size_t max_length, std::string_view name_kind) {
  const auto placeholder = template_string.find("%d");
  if (placeholder == std::string::npos ||
      template_string.find('%') != placeholder ||
      template_string.find('%', placeholder + 2) != std::string::npos) {
    throw std::runtime_error("Unable to compose " + std::string{name_kind} +
                             " from malformed template " + template_string +
                             "; expected exactly one %d.");
  }

  std::string result;
  result.reserve(template_string.size() + 10);
  result.append(template_string, 0, placeholder);
  result.append(std::to_string(index));
  result.append(template_string, placeholder + 2);
  if (result.size() >= max_length) {
    throw std::runtime_error("Unable to compose " + std::string{name_kind} +
                             " from template " + template_string +
                             "; the expanded name is too long.");
  }
  return result;
}

namespace {

bool IsAttentionCacheNeeded(const Model& model) {
  const auto& key_template = model.config_->model.decoder.inputs.past_key_names;
  const auto placeholder = key_template.find('%');
  if (placeholder == std::string::npos) {
    return false;
  }

  auto prefix = key_template.substr(0, placeholder);
  auto suffix = key_template.substr(placeholder + 2);
  for (const auto& name : model.session_info_.GetInputNames()) {
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      return true;
  }
  return false;
}

}  // namespace

bool CanUseTopLevelDeviceKeyValueCachePolicy(const Model& model) {
  return model.config_->model.decoder.pipeline.empty();
}

bool UsesNonRewindableWindowedKeyValueCache(
    const Model& model, const Config::Model::Decoder& decoder) {
  return CanUseTopLevelDeviceKeyValueCachePolicy(model) &&
         model.p_device_kvcache_->UsesNonRewindableWindowedKeyValueCache(decoder);
}

std::unique_ptr<KeyValueCache> CreateStandardKeyValueCache(State& state) {
  // LFM2 models interleave attention and conv layers, requiring a cache that handles
  // both KV cache for attention layers and fixed-size conv state for conv layers.
  if (HasConvKeyValueCache(state.model_)) {
    return std::make_unique<ConvKeyValueCache>(state);
  }

  if (!IsAttentionCacheNeeded(state.model_)) {
    return nullptr;
  }

  if (UsesNonRewindableWindowedKeyValueCache(
          state.model_, state.model_.config_->model.decoder)) {
    return std::make_unique<WindowedKeyValueCache>(state);
  }

  const int kv_cache_quantization_bits =
      state.model_.p_device_kvcache_->GetKeyValueCacheQuantizationBits(state.model_.config_->model.decoder.session_options);

  if (ShouldUseSharedPastPresentKeyValueCache(state)) {
    const int windowed_cache_size = GetWindowedKeyValueCacheSize(
        state.model_, state.params_->search, state.params_->search.max_length);
    if (windowed_cache_size > 0) {
      return std::make_unique<EpManagedSlidingKeyValueCache>(state);
    }
    if (kv_cache_quantization_bits == 4 || kv_cache_quantization_bits == 8) {
      return std::make_unique<QuantizedSharedKeyValueCache>(state);
    }
    return std::make_unique<SharedKeyValueCache>(state);
  }

  if (kv_cache_quantization_bits == 4 || kv_cache_quantization_bits == 8) {
    return std::make_unique<QuantizedRegularKeyValueCache>(state);
  }
  return std::make_unique<RegularKeyValueCache>(state);
}

std::unique_ptr<KeyValueCache> CreateModelManagedKeyValueCache(State& state) {
  if (g_log.enabled)
    Log("info", "CreateKeyValueCache: Creating ModelManagedKeyValueCache");
  return std::make_unique<ModelManagedKeyValueCache>(state);
}

}  // namespace Generators

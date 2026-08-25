// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "generator/generators.h"
#include "../model.h"
#include "conv_kv_cache.h"
#include "ep_managed_sliding_kv_cache.h"
#include "kv_cache.h"
#include "model_managed_kv_cache.h"
#include "regular_kv_cache.h"
#include "shared_kv_cache.h"
#include "static_kv_cache.h"
#include "windowed_kv_cache.h"
#include <algorithm>

namespace Generators {

std::string ComposeKeyValueName(const std::string& template_string, int index) {
  constexpr int32_t KeyValueNameLength = 64;
  char key_value_name[KeyValueNameLength];
  if (auto length = snprintf(key_value_name, std::size(key_value_name), template_string.c_str(), index);
      length < 0 || length >= KeyValueNameLength) {
    throw std::runtime_error("Unable to compose key value name from the provided template " + template_string +
                             ". This could be either due to an encoding error or the name being too long.");
  }
  return std::string(key_value_name);
}

namespace {

bool IsAttentionCacheNeeded(const Model& model) {
  const auto& key_template = model.config_->model.decoder.inputs.past_key_names;
  auto prefix = key_template.substr(0, key_template.find('%'));
  auto suffix = key_template.substr(key_template.find('%') + 2);
  for (const auto& name : model.session_info_.GetInputNames()) {
    if (name.size() > prefix.size() + suffix.size() &&
        name.compare(0, prefix.size(), prefix) == 0 &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      return true;
  }
  return false;
}

}  // namespace

bool UsesNonRewindableWindowedKeyValueCache(
    const Model& model, const Config::Model::Decoder& decoder) {
  return model.p_device_kvcache_->UsesNonRewindableWindowedKeyValueCache(decoder);
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

  if (ShouldUseSharedPastPresentKeyValueCache(state)) {
    const int windowed_cache_size = state.model_.p_device_kvcache_->GetWindowedKeyValueCacheSize(
        state.model_.config_->model.decoder, state.params_->search, state.params_->search.max_length);
    if (windowed_cache_size > 0) {
      return std::make_unique<EpManagedSlidingKeyValueCache>(state);
    }
    return std::make_unique<SharedKeyValueCache>(state);
  }

  return std::make_unique<RegularKeyValueCache>(state);
}

std::unique_ptr<KeyValueCache> CreateModelManagedKeyValueCache(State& state) {
  if (g_log.enabled)
    Log("info", "CreateKeyValueCache: Creating ModelManagedKeyValueCache");
  return std::make_unique<ModelManagedKeyValueCache>(state);
}

}  // namespace Generators

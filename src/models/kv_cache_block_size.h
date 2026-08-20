// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Pure helpers for the kv_cache_block_size search option: decide whether the
// pre-allocated KV cache sequence length should be rounded up to a multiple of a
// block size, and apply that rounding while leaving local (sliding-window) layers
// untouched. Kept header-only and free of any model/session state so the sizing
// behavior can be unit tested directly (see test/kv_cache_block_size_test.cpp).

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace Generators {

// Round seq_len up to the nearest multiple of block_size. block_size <= 0 means
// "not configured" and leaves seq_len unchanged. block_size is bounded to int32 at
// the config boundary (SafeDoubleToInt) and seq_len is an int-sized cache dimension,
// so the int64 arithmetic cannot overflow.
inline int64_t RoundUpToKvCacheBlock(int64_t seq_len, int64_t block_size) {
  if (block_size <= 0)
    return seq_len;
  return ((seq_len + block_size - 1) / block_size) * block_size;
}

// Whether the KV cache sequence length should be block-rounded at all.
// Rounding only makes sense for a shared past/present buffer whose sequence
// dimension is fixed at allocation time, and is suppressed when:
//   * the model has a fixed KV shape dictated by its graph (fixed_kv_seq_len > 0), or
//   * the cache is a uniform sliding-window cache -- windowed with no explicit
//     per-layer list -- in which case every layer is local and must not grow.
inline bool ShouldRoundKvCacheToBlock(bool past_present_share_buffer,
                                      int64_t fixed_kv_seq_len,
                                      bool windowed_cache,
                                      bool sliding_window_layers_empty,
                                      const std::optional<size_t>& block_size) {
  if (!past_present_share_buffer || fixed_kv_seq_len > 0)
    return false;
  const bool uniform_local_cache = windowed_cache && sliding_window_layers_empty;
  if (uniform_local_cache)
    return false;
  return block_size.has_value() && static_cast<int64_t>(block_size.value()) > 0;
}

// Build a per-cache-slot flag marking local (sliding-window) attention layers
// (0 = global/full attention, 1 = local). sliding_window_layers holds model-layer
// indices; kv_layer_indices maps cache slot -> model layer for sparse KV layouts
// (empty means slot i == model layer i). Returns an all-zero vector of length
// layer_count when there are no sliding-window layers.
inline std::vector<std::uint8_t> ComputeLocalAttentionSlots(
    int layer_count,
    const std::vector<int>& kv_layer_indices,
    const std::vector<int>& sliding_window_layers) {
  std::vector<std::uint8_t> is_local_attention(layer_count > 0 ? static_cast<size_t>(layer_count) : 0, 0);
  if (layer_count <= 0 || sliding_window_layers.empty())
    return is_local_attention;

  std::unordered_map<int, int> model_layer_to_cache_slot;
  for (int slot = 0; slot < layer_count; ++slot) {
    int model_idx = kv_layer_indices.empty() ? slot : kv_layer_indices[slot];
    model_layer_to_cache_slot[model_idx] = slot;
  }
  for (int model_layer_idx : sliding_window_layers) {
    auto it = model_layer_to_cache_slot.find(model_layer_idx);
    if (it != model_layer_to_cache_slot.end())
      is_local_attention[static_cast<size_t>(it->second)] = 1;
  }
  return is_local_attention;
}

// Round the sequence dimension (index 2) of the shared uniform shape and every
// per-layer shape up to a multiple of block_size, skipping local attention slots.
// No-op when block_size <= 0. layer_shapes may be empty (uniform cache), in which
// case only the shared shape is adjusted.
inline void ApplyKvCacheBlockSize(int64_t block_size,
                                  const std::vector<std::uint8_t>& is_local_attention,
                                  std::array<int64_t, 4>& shape,
                                  std::vector<std::array<int64_t, 4>>& layer_shapes) {
  if (block_size <= 0)
    return;
  shape[2] = RoundUpToKvCacheBlock(shape[2], block_size);
  for (size_t layer_idx = 0; layer_idx < layer_shapes.size(); ++layer_idx) {
    if (layer_idx < is_local_attention.size() && is_local_attention[layer_idx])
      continue;
    layer_shapes[layer_idx][2] = RoundUpToKvCacheBlock(layer_shapes[layer_idx][2], block_size);
  }
}

}  // namespace Generators

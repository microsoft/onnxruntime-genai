// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// The geometry of the compact state-update replay of the "qwen38_flash_graph" model, checked on the host.
//
// The replay kernels of the plane (k_replay_gdn, k_replay_generic) are told the geometry of every recurrent state by a
// descriptor that the engine fills from the shapes of the package's tensors. A kernel that is handed a geometry it cannot hold
// does not fail: it rewrites the wrong elements, or none, and the device status bits do not see it. So each descriptor is
// validated while it is built, without a device, and a package whose state does not fit the kernels is rejected when the model
// is loaded. The limits of the kernels (QWEN38_PLANE_MAX_REPLAY_*) come from the plane ABI header that the kernels are built
// from.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace Generators {

// The state a replay rewrites.
enum class Qwen38ReplayKind {
  // state_update.<layer>.recurrent_capsule -> past_key_values.<layer>.recurrent_state [batch, value heads, value width, key width]
  // (float32). The capsule [batch, capacity * (heads_v + heads_k * key width + heads_v * value width)] (float32) holds, per
  // transition, the decays [heads_v], the keys [heads_k, key width] and the deltas [heads_v, value width].
  kGatedDeltaNet,
  // state_update.<layer>.conv_value | ple_conv_value -> past_key_values.<layer>.conv_state | ple_conv_state [batch, channels,
  // history]: the new state is the last `history` columns of the old state followed by the `kept` accepted columns of the
  // update [batch, capacity, channels].
  kConvolution,
  // state_update.<layer>.indexer_value -> past_key_values.<layer>.index_buffer [batch, entries, width] and index_lengths: the
  // accepted rows of the update [batch, capacity, width] extend the partly filled block of the compressed index keys.
  kIndexer,
};

// A tensor of a replay as the package declares it: its name (for the diagnostics), its shape (the engine maps the symbolic
// dimensions to 1) and the size of one element in bytes.
struct Qwen38ReplayTensor {
  std::string name;
  std::vector<int64_t> shape;
  size_t element_size{};
};

struct Qwen38ReplayRequest {
  Qwen38ReplayKind kind{};
  int capacity{};              // transitions the state_update outputs carry (the verify rows - 1)
  Qwen38ReplayTensor state;    // the state of the source bank (the recurrent state, conv state or index buffer)
  Qwen38ReplayTensor update;   // the state_update.<layer>.* output
  Qwen38ReplayTensor lengths;  // kIndexer: past_key_values.<layer>.index_lengths
  int compress_ratio{};        // kIndexer: tokens pooled into one compressed index key
};

// What a descriptor takes from the shapes (the unused fields stay 0).
struct Qwen38ReplayGeometry {
  uint64_t channel_count{};   // kGatedDeltaNet: value heads. kConvolution: channels
  uint64_t state_width{};     // kGatedDeltaNet: value width. kConvolution: history. kIndexer: width of an entry
  uint64_t key_width{};       // kGatedDeltaNet
  uint64_t key_head_count{};  // kGatedDeltaNet
  uint64_t aux_capacity{};    // kIndexer: entries of the buffer
  uint32_t element_size{};
  int blocks{};  // kGatedDeltaNet: thread blocks one launch of k_replay_gdn needs for the descriptor
};

// Returns the geometry of the replay `request` describes, or throws a std::runtime_error ("qwen38_graph: the compact replay of
// <update> into <state> is not possible: <rule>") naming the tensors, their shapes and the rule that the package breaks:
//   every tensor has the rank above, a batch of 1 and dimensions in [1, INT32_MAX]; 1 <= capacity <=
//   QWEN38_PLANE_MAX_REPLAY_TRANSITIONS
//   kGatedDeltaNet: float32 state and capsule; the key width is a multiple of 4 and at most QWEN38_PLANE_MAX_REPLAY_KEY_WIDTH
//     (the kernel's shared tile and float4 loads); the capsule width is a multiple of the capacity; what a transition holds
//     beyond the decays and the deltas is a whole number (>= 1) of keys of the key width, one per key head; the value heads
//     are a multiple of the key heads.
//   kConvolution: 2- or 4-byte elements, the same in the state and in the update (the kernel copies nothing else); the update
//     carries `capacity` transitions; the state is channel-first, i.e. its second dimension is the channel count of the update.
//     A history-first state [batch, history, channels] is rejected: the replay would shift its channels instead of its history.
//     A state whose history equals its channel count cannot be told apart and is taken as channel-first.
//   kIndexer: 2- or 4-byte elements, the same in the buffer and in the update; the update carries `capacity` transitions of the
//     width of the buffer entries; the buffer holds at least compress_ratio - 1 entries (a partly filled block); the lengths are
//     int32 words, at least two of them (the number of compressed keys and the entries in the buffer).
Qwen38ReplayGeometry ValidateQwen38ReplayDescriptor(const Qwen38ReplayRequest& request);

// The message of the exception RequireQwen38ReplayTensor throws.
std::string DescribeMissingQwen38ReplayTensor(const std::string& name, const std::string& layer, const std::string& where);

// Every banked state input in `bank_names` (`past_key_values.<layer>.<suffix>`) is overwritten in the destination bank by a
// two-bank compact-replay verify step, so on partial acceptance it must be replayed from its `state_update.<layer>.*` output.
// Throws std::runtime_error naming the banked state and the missing `state_update.*` output when a banked state has no update.
void RequireQwen38ReplayCoverage(const std::vector<std::string>& bank_names,
                                 const std::vector<std::string>& state_update_names);

// The tensor `name` of `tensors` (a map from a name to a unique_ptr to a tensor: the state tensors of a bank, ...), which the
// replay of layer `layer` needs. Throws a std::runtime_error naming the tensor, the layer and `where` (e.g. "bank 1") when the
// package does not declare it, instead of the bare std::out_of_range of at().
template <typename Tensors>
const typename Tensors::mapped_type::element_type& RequireQwen38ReplayTensor(const Tensors& tensors, const std::string& name,
                                                                             const std::string& layer,
                                                                             const std::string& where) {
  const auto found = tensors.find(name);
  if (found == tensors.end() || !found->second) {
    throw std::runtime_error(DescribeMissingQwen38ReplayTensor(name, layer, where));
  }
  return *found->second;
}

}  // namespace Generators

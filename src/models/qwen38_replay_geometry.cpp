// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qwen38_replay_geometry.h"

#include <cstdint>
#include <limits>

#include "qwen38_plane/qwen38_plane_abi.h"

namespace Generators {

namespace {

constexpr int64_t kMaxDimension = std::numeric_limits<int32_t>::max();

std::string ShapeText(const std::vector<int64_t>& shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) text += ", ";
    text += std::to_string(shape[i]);
  }
  return text + "]";
}

std::string Describe(const Qwen38ReplayTensor& tensor) {
  return tensor.name + " " + ShapeText(tensor.shape);
}

[[noreturn]] void Reject(const Qwen38ReplayRequest& request, const std::string& rule) {
  throw std::runtime_error("qwen38_graph: the compact replay of " + request.update.name + " into " + request.state.name +
                           " is not possible: " + rule);
}

uint64_t Dimension(const Qwen38ReplayTensor& tensor, size_t index) {
  return static_cast<uint64_t>(tensor.shape[index]);
}

// `layout` names the dimensions of the tensor's shape for the diagnostic.
void RequireShape(const Qwen38ReplayRequest& request, const Qwen38ReplayTensor& tensor, size_t rank, const char* layout) {
  if (tensor.shape.size() != rank) {
    Reject(request, Describe(tensor) + " must have rank " + std::to_string(rank) + ": " + layout);
  }
  for (const int64_t dimension : tensor.shape) {
    if (dimension < 1 || dimension > kMaxDimension) {
      Reject(request, Describe(tensor) + " must have dimensions in [1, " + std::to_string(kMaxDimension) + "]: " + layout);
    }
  }
  if (tensor.shape[0] != 1) {
    Reject(request, Describe(tensor) + " must have a batch of 1, the replay handles one sequence: " + layout);
  }
}

void RequireCapacity(const Qwen38ReplayRequest& request) {
  if (request.capacity < 1 || request.capacity > QWEN38_PLANE_MAX_REPLAY_TRANSITIONS) {
    Reject(request, "the replay capacity is " + std::to_string(request.capacity) + " but the kernels replay 1 to " +
                        std::to_string(QWEN38_PLANE_MAX_REPLAY_TRANSITIONS) + " transitions");
  }
}

// The update of a convolution or of an indexer carries one row per transition: [batch, capacity, ...].
void RequireTransitions(const Qwen38ReplayRequest& request) {
  if (request.update.shape[1] != request.capacity) {
    Reject(request, Describe(request.update) + " carries " + std::to_string(request.update.shape[1]) +
                        " transitions but the replay capacity is " + std::to_string(request.capacity));
  }
}

void RequireFloat32(const Qwen38ReplayRequest& request, const Qwen38ReplayTensor& tensor) {
  if (tensor.element_size != sizeof(float)) {
    Reject(request, tensor.name + " has " + std::to_string(tensor.element_size) +
                        "-byte elements but the GatedDeltaNet replay reads and writes float32");
  }
}

// k_replay_generic copies 16- or 32-bit elements and does nothing for any other size, which would leave the state stale.
void RequireCopyableElements(const Qwen38ReplayRequest& request) {
  const size_t size = request.state.element_size;
  if (size != 2 && size != 4) {
    Reject(request, request.state.name + " has " + std::to_string(size) +
                        "-byte elements but the replay copies 2- or 4-byte elements and would leave any other state stale");
  }
  if (request.update.element_size != size) {
    Reject(request, request.update.name + " has " + std::to_string(request.update.element_size) + "-byte elements but " +
                        request.state.name + " has " + std::to_string(size) + "-byte elements");
  }
}

Qwen38ReplayGeometry ValidateGatedDeltaNet(const Qwen38ReplayRequest& request) {
  const Qwen38ReplayTensor& state = request.state;
  const Qwen38ReplayTensor& capsule = request.update;
  RequireShape(request, state, 4, "[batch, value heads, value width, key width]");
  RequireShape(request, capsule, 2, "[batch, capsule width]");
  RequireFloat32(request, state);
  RequireFloat32(request, capsule);
  RequireCapacity(request);

  const uint64_t heads_v = Dimension(state, 1);
  const uint64_t width_v = Dimension(state, 2);
  const uint64_t width_k = Dimension(state, 3);
  if (width_k > QWEN38_PLANE_MAX_REPLAY_KEY_WIDTH) {
    Reject(request, "the key width " + std::to_string(width_k) + " of " + Describe(state) + " exceeds the " +
                        std::to_string(QWEN38_PLANE_MAX_REPLAY_KEY_WIDTH) + " that the replay kernel holds in its shared tile");
  }
  if (width_k % 4 != 0) {
    Reject(request, "the key width " + std::to_string(width_k) + " of " + Describe(state) +
                        " is not a multiple of 4, the replay kernel loads float4 along it");
  }

  const uint64_t capacity = static_cast<uint64_t>(request.capacity);
  const uint64_t capsule_width = Dimension(capsule, 1);
  if (capsule_width % capacity != 0) {
    Reject(request, "the capsule width " + std::to_string(capsule_width) + " of " + Describe(capsule) +
                        " is not a multiple of the " + std::to_string(capacity) + " transitions it holds");
  }
  // Per transition: the decays [heads_v], the keys [heads_k, key width] and the deltas [heads_v, value width].
  const uint64_t per_transition = capsule_width / capacity;
  const uint64_t decays_and_deltas = heads_v + heads_v * width_v;
  if (per_transition <= decays_and_deltas) {
    Reject(request, "a transition of " + Describe(capsule) + " holds " + std::to_string(per_transition) +
                        " values, no more than the " + std::to_string(decays_and_deltas) + " that its " +
                        std::to_string(heads_v) + " decays and " + std::to_string(heads_v) + " x " + std::to_string(width_v) +
                        " deltas take, so no key is left");
  }
  const uint64_t key_values = per_transition - decays_and_deltas;
  if (key_values % width_k != 0) {
    Reject(request, "the " + std::to_string(key_values) + " key values of a transition of " + Describe(capsule) +
                        " are not a whole number of keys of the key width " + std::to_string(width_k) + " of " +
                        Describe(state));
  }
  const uint64_t heads_k = key_values / width_k;  // at least 1: key_values is positive
  if (heads_v % heads_k != 0) {
    Reject(request, "the " + std::to_string(heads_v) + " value heads of " + Describe(state) +
                        " are not a multiple of the " + std::to_string(heads_k) + " key heads that " + Describe(capsule) +
                        " implies");
  }

  constexpr uint64_t kRows = QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK;
  const uint64_t blocks = heads_v * ((width_v + kRows - 1) / kRows);
  if (blocks > static_cast<uint64_t>(kMaxDimension)) {
    Reject(request, Describe(state) + " needs " + std::to_string(blocks) + " thread blocks, more than a launch holds");
  }

  Qwen38ReplayGeometry geometry;
  geometry.channel_count = heads_v;
  geometry.state_width = width_v;
  geometry.key_width = width_k;
  geometry.key_head_count = heads_k;
  geometry.element_size = static_cast<uint32_t>(state.element_size);
  geometry.blocks = static_cast<int>(blocks);
  return geometry;
}

Qwen38ReplayGeometry ValidateConvolution(const Qwen38ReplayRequest& request) {
  const Qwen38ReplayTensor& state = request.state;
  const Qwen38ReplayTensor& value = request.update;
  RequireShape(request, state, 3, "[batch, channels, history]");
  RequireShape(request, value, 3, "[batch, capacity, channels]");
  RequireCopyableElements(request);
  RequireCapacity(request);
  RequireTransitions(request);

  // The kernel indexes the state as [channels, history]. The channel count is that of the update, which the state has to
  // match in its second dimension; when it matches in its third instead, the state is history-first and replaying it as
  // channel-first would shift its channels. When history == channels the layouts cannot be told apart: the exporter writes
  // channel-first, which is what such a state is taken as.
  const int64_t channels = value.shape[2];
  if (state.shape[1] != channels) {
    if (state.shape[2] == channels) {
      Reject(request, Describe(state) + " has the " + std::to_string(channels) + " channels of " + Describe(value) +
                          " in its last dimension: history-first conv state is not supported by the compact replay, which "
                          "needs [batch, " +
                          std::to_string(channels) + " channels, history]");
    }
    Reject(request, "neither the second nor the third dimension of " + Describe(state) + " is the " +
                        std::to_string(channels) + " channels of " + Describe(value) +
                        ", expected a channel-first [batch, " + std::to_string(channels) + " channels, history] state");
  }

  Qwen38ReplayGeometry geometry;
  geometry.channel_count = static_cast<uint64_t>(channels);
  geometry.state_width = Dimension(state, 2);
  geometry.element_size = static_cast<uint32_t>(state.element_size);
  return geometry;
}

Qwen38ReplayGeometry ValidateIndexer(const Qwen38ReplayRequest& request) {
  const Qwen38ReplayTensor& buffer = request.state;
  const Qwen38ReplayTensor& value = request.update;
  RequireShape(request, buffer, 3, "[batch, entries, width]");
  RequireShape(request, value, 3, "[batch, capacity, width]");
  RequireCopyableElements(request);
  RequireCapacity(request);
  RequireTransitions(request);
  if (value.shape[2] != buffer.shape[2]) {
    Reject(request, Describe(value) + " has rows of " + std::to_string(value.shape[2]) + " values but the entries of " +
                        Describe(buffer) + " have " + std::to_string(buffer.shape[2]));
  }
  if (request.compress_ratio < 1) {
    Reject(request, "the compression ratio is " + std::to_string(request.compress_ratio));
  }
  if (buffer.shape[1] < request.compress_ratio - 1) {
    Reject(request, Describe(buffer) + " holds fewer than the " + std::to_string(request.compress_ratio - 1) +
                        " entries of a partly filled block of a compression ratio of " +
                        std::to_string(request.compress_ratio));
  }

  const Qwen38ReplayTensor& lengths = request.lengths;
  if (lengths.element_size != sizeof(int32_t)) {
    Reject(request, lengths.name + " has " + std::to_string(lengths.element_size) + "-byte elements but the lengths are int32");
  }
  uint64_t length_words = lengths.shape.empty() ? 0 : 1;
  for (const int64_t dimension : lengths.shape) {
    if (dimension < 1 || dimension > kMaxDimension) {
      Reject(request, Describe(lengths) + " must have dimensions in [1, " + std::to_string(kMaxDimension) + "]");
    }
    length_words = length_words < 2 ? length_words * static_cast<uint64_t>(dimension) : length_words;
  }
  if (length_words < 2) {
    Reject(request, Describe(lengths) + " must hold at least two words: the compressed keys and the entries in the buffer");
  }

  Qwen38ReplayGeometry geometry;
  geometry.state_width = Dimension(buffer, 2);
  geometry.aux_capacity = Dimension(buffer, 1);
  geometry.element_size = static_cast<uint32_t>(buffer.element_size);
  return geometry;
}

}  // namespace

Qwen38ReplayGeometry ValidateQwen38ReplayDescriptor(const Qwen38ReplayRequest& request) {
  switch (request.kind) {
    case Qwen38ReplayKind::kGatedDeltaNet:
      return ValidateGatedDeltaNet(request);
    case Qwen38ReplayKind::kConvolution:
      return ValidateConvolution(request);
    case Qwen38ReplayKind::kIndexer:
      return ValidateIndexer(request);
  }
  Reject(request, "unknown replay kind " + std::to_string(static_cast<int>(request.kind)));
}

std::string DescribeMissingQwen38ReplayTensor(const std::string& name, const std::string& layer, const std::string& where) {
  return "qwen38_graph: the state replay of layer " + layer + " needs the tensor " + name + " in " + where +
         ", which the target graph does not declare";
}

}  // namespace Generators

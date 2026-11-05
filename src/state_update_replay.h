// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "smartptrs.h"

namespace Generators {

// Applies compact fixed-state transitions to descriptors whose views are CPU-addressable.
// Device implementations may stage their views to CPU, call this helper, and copy the completed
// destination rows back.
inline void ReplayStateUpdatesOnCpu(
    const StateUpdateReplayDesc* descriptors, size_t count) {
  for (size_t descriptor_index = 0; descriptor_index < count; ++descriptor_index) {
    const auto& descriptor = descriptors[descriptor_index];
    if (descriptor.kind == StateUpdateReplayKind::Snapshot) {
      std::memcpy(
          descriptor.destination_state.Span().data(),
          descriptor.value.Span().data() +
              static_cast<uint64_t>(descriptor.kept_count - 1) * descriptor.state_width *
                  descriptor.element_size,
          descriptor.state_width * descriptor.element_size);
      continue;
    }
    if (descriptor.kind == StateUpdateReplayKind::Indexer) {
      const auto* source_lengths =
          reinterpret_cast<const int32_t*>(descriptor.source_lengths.Span().data());
      auto* destination_lengths =
          reinterpret_cast<int32_t*>(descriptor.destination_lengths.Span().data());
      const int32_t old_key_length = source_lengths[0];
      const int32_t old_buffer_length = source_lengths[1];
      if (descriptor.compress_ratio == 0 ||
          descriptor.compress_ratio - 1 > descriptor.aux_capacity ||
          old_key_length < 0 || old_buffer_length < 0 ||
          static_cast<uint64_t>(old_key_length) > descriptor.state_capacity ||
          static_cast<uint32_t>(old_buffer_length) >= descriptor.compress_ratio ||
          static_cast<uint64_t>(old_buffer_length) > descriptor.aux_capacity) {
        throw std::runtime_error("Indexer state_update has invalid committed lengths.");
      }
      const uint64_t retained = static_cast<uint64_t>(old_buffer_length) + descriptor.kept_count;
      if (retained / descriptor.compress_ratio >
              descriptor.state_capacity - static_cast<uint64_t>(old_key_length) ||
          retained % descriptor.compress_ratio > descriptor.aux_capacity ||
          retained / descriptor.compress_ratio > static_cast<uint64_t>(INT32_MAX - old_key_length)) {
        throw std::runtime_error("Indexer state_update replay exceeds state capacity.");
      }
      const auto* source = descriptor.source_state.Span().data();
      auto* destination = descriptor.destination_state.Span().data();
      const auto* source_aux = descriptor.source_aux_state.Span().data();
      auto* destination_aux = descriptor.destination_aux_state.Span().data();
      const uint64_t key_row_bytes = descriptor.state_width * descriptor.element_size;
      if (destination != source) {
        std::memcpy(destination, source, descriptor.state_capacity * key_row_bytes);
      }
      if (destination_aux != source_aux) {
        std::memcpy(destination_aux, source_aux, descriptor.aux_capacity * key_row_bytes);
      }
      int32_t key_length = old_key_length;
      int32_t buffer_length = old_buffer_length;
      for (uint32_t token = 0; token < descriptor.kept_count; ++token) {
        const auto* payload = descriptor.value.Span().data() +
                              static_cast<uint64_t>(token) * key_row_bytes;
        if (++buffer_length == static_cast<int32_t>(descriptor.compress_ratio)) {
          std::memcpy(destination + static_cast<uint64_t>(key_length++) * key_row_bytes,
                      payload, key_row_bytes);
          buffer_length = 0;
        } else {
          std::memcpy(destination_aux + static_cast<uint64_t>(buffer_length - 1) * key_row_bytes,
                      payload, key_row_bytes);
        }
      }
      destination_lengths[0] = key_length;
      destination_lengths[1] = buffer_length;
      continue;
    }
    if (descriptor.kind == StateUpdateReplayKind::CausalConv) {
      const auto* source = descriptor.source_state.Span().data();
      auto* destination = descriptor.destination_state.Span().data();
      const auto* values = descriptor.value.Span().data();
      for (uint64_t channel = 0; channel < descriptor.channel_count; ++channel) {
        for (uint64_t position = 0; position < descriptor.state_width; ++position) {
          const uint64_t shifted_position = position + descriptor.kept_count;
          const uint8_t* source_element{};
          if (shifted_position < descriptor.state_width) {
            source_element = source +
                             (channel * descriptor.state_width + shifted_position) *
                                 descriptor.element_size;
          } else {
            const uint64_t update_index = shifted_position - descriptor.state_width;
            source_element = values +
                             (update_index * descriptor.channel_count + channel) *
                                 descriptor.element_size;
          }
          std::memcpy(
              destination + (channel * descriptor.state_width + position) * descriptor.element_size,
              source_element, descriptor.element_size);
        }
      }
      continue;
    }

    const auto* source = reinterpret_cast<const float*>(descriptor.source_state.Span().data());
    auto* destination = reinterpret_cast<float*>(descriptor.destination_state.Span().data());
    const auto* decay = reinterpret_cast<const float*>(descriptor.decay.Span().data());
    const auto* key = reinterpret_cast<const float*>(descriptor.key.Span().data());
    const auto* delta = reinterpret_cast<const float*>(descriptor.delta.Span().data());
    for (uint64_t value_head = 0; value_head < descriptor.channel_count; ++value_head) {
      const uint64_t key_head =
          value_head * descriptor.key_head_count / descriptor.channel_count;
      for (uint64_t value_index = 0; value_index < descriptor.state_width; ++value_index) {
        for (uint64_t key_index = 0; key_index < descriptor.key_width; ++key_index) {
          const uint64_t state_index =
              (value_head * descriptor.state_width + value_index) * descriptor.key_width + key_index;
          float state = source[state_index];
          for (uint32_t transition = 0; transition < descriptor.kept_count; ++transition) {
            state *= decay[static_cast<uint64_t>(transition) * descriptor.channel_count + value_head];
            state = std::fma(
                key[(static_cast<uint64_t>(transition) * descriptor.key_head_count + key_head) *
                        descriptor.key_width +
                    key_index],
                delta[(static_cast<uint64_t>(transition) * descriptor.channel_count + value_head) *
                          descriptor.state_width +
                      value_index],
                state);
          }
          destination[state_index] = state;
        }
      }
    }
  }
}

}  // namespace Generators

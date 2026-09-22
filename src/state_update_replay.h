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
            state += key[(static_cast<uint64_t>(transition) * descriptor.key_head_count + key_head) *
                             descriptor.key_width +
                         key_index] *
                     delta[(static_cast<uint64_t>(transition) * descriptor.channel_count + value_head) *
                               descriptor.state_width +
                           value_index];
          }
          destination[state_index] = state;
        }
      }
    }
  }
}

}  // namespace Generators

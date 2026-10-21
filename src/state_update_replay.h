// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "smartptrs.h"
#include "models/utils.h"
#include <cmath>
#include <bit>

namespace Generators {

inline uint16_t RecurrentStateToFp16(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
  const uint32_t mantissa = bits & 0x7fffff;
  const int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
  if ((bits & 0x7f800000) == 0x7f800000) {
    return sign | 0x7c00 | (mantissa != 0 ? 0x200 : 0);
  }
  if (exponent >= 31) return sign | 0x7c00;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    const uint32_t significand = mantissa | 0x800000;
    const int shift = 14 - exponent;
    return sign | static_cast<uint16_t>(
                      (significand + ((1u << (shift - 1)) - 1) +
                       ((significand >> shift) & 1)) >>
                      shift);
  }
  // Round ties to even, including carry into the exponent.
  return sign | static_cast<uint16_t>(
                    (static_cast<uint32_t>(exponent) << 10) +
                    ((mantissa + 0xfff + ((mantissa >> 13) & 1)) >> 13));
}

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

    const bool fp16 = descriptor.element_size == sizeof(uint16_t);
    const auto* source = descriptor.source_state.Span().data();
    auto* destination = descriptor.destination_state.Span().data();
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
          float state = fp16 ? Float16ToFloat32(reinterpret_cast<const uint16_t*>(source)[state_index])
                             : reinterpret_cast<const float*>(source)[state_index];
          for (uint32_t transition = 0; transition < descriptor.kept_count; ++transition) {
            state *= decay[static_cast<uint64_t>(transition) * descriptor.channel_count + value_head];
            const float k = key[(static_cast<uint64_t>(transition) * descriptor.key_head_count + key_head) *
                                    descriptor.key_width +
                                key_index];
            const float d = delta[(static_cast<uint64_t>(transition) * descriptor.channel_count + value_head) *
                                      descriptor.state_width +
                                  value_index];
            state = fp16 ? Float16ToFloat32(RecurrentStateToFp16(std::fma(k, d, state)))
                         : state + k * d;
          }
          if (fp16) {
            reinterpret_cast<uint16_t*>(destination)[state_index] = RecurrentStateToFp16(state);
          } else {
            reinterpret_cast<float*>(destination)[state_index] = state;
          }
        }
      }
    }
  }
}

}  // namespace Generators

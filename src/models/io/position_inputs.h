#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "generator/generators.h"
#include "models/model.h"

namespace Generators {

struct PositionInputs {
  virtual ~PositionInputs() = default;
  virtual void Add() = 0;
  virtual void Update(DeviceSpan<int32_t> next_tokens, int total_length, int new_length) = 0;
  virtual void RewindTo(size_t index) = 0;
};

std::unique_ptr<PositionInputs> CreateStandardPositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name);

}  // namespace Generators

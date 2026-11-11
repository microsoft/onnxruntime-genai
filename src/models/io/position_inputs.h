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

std::unique_ptr<PositionInputs> CreateStandardPositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name);

}  // namespace Generators

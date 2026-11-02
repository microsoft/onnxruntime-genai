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

// Storage policies are shared by 2D position IDs and Qwen's 3D mRoPE IDs.
// Model implementations remain responsible for calculating the position values.
struct PositionIdsBuffer {
  virtual ~PositionIdsBuffer() = default;
  virtual void Create(Tensor& tensor, std::span<const int64_t> shape) const;
  virtual void Assign(Tensor& tensor, const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) const;
};

std::unique_ptr<PositionIdsBuffer> CreatePositionIdsBuffer(const GeneratorParams& params);

// Dynamic Qwen masks historically contain only ones after prefill. Keep that
// behavior separate from the default decoder's padding-preserving updates.
enum class DynamicMaskUpdate { PreservePadding,
                               FillOnes };

struct AttentionMaskBuffer {
  virtual ~AttentionMaskBuffer() = default;
  virtual void Initialize(Tensor& tensor, std::unique_ptr<OrtValue>& values, int num_beams,
                          std::array<int64_t, 2>& shape) = 0;
  virtual void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) = 0;
  virtual void Rewind(Tensor& tensor, std::array<int64_t, 2>& shape, size_t index) = 0;
};

std::unique_ptr<AttentionMaskBuffer> CreateAttentionMaskBuffer(
    const Model& model, const GeneratorParams& params, DynamicMaskUpdate update = DynamicMaskUpdate::PreservePadding);

std::unique_ptr<PositionInputs> CreateStandardPositionInputs(State& state, DeviceSpan<int32_t> sequence_lengths, const std::string& attention_mask_name);

}  // namespace Generators

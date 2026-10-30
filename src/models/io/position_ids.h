// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/model.h"

namespace Generators {

// Storage policies are shared by 2D position IDs and Qwen's 3D mRoPE IDs.
// Model implementations remain responsible for calculating the position values.
struct PositionIdsBuffer {
  virtual ~PositionIdsBuffer() = default;
  virtual void Create(Tensor& tensor, std::span<const int64_t> shape) const;
  virtual void Assign(Tensor& tensor, const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) const;

 protected:
  static void ExpandPositionIds(const Model& model, std::unique_ptr<OrtValue>& values, int num_beams);
};

std::unique_ptr<PositionIdsBuffer> CreatePositionIdsBuffer(const GeneratorParams& params);

std::unique_ptr<PositionIdsBuffer> CreateCapturedPositionIdsBuffer(int max_capture_length);

}  // namespace Generators

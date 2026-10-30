// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/model.h"

namespace Generators {

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

 protected:
  static void UpdateMaskOnDevice(Tensor& mask, Tensor* next, const std::array<int64_t, 2>& shape,
                                 int total_length, int new_length, int max_length);
};

std::unique_ptr<AttentionMaskBuffer> CreateAttentionMaskBuffer(
    const Model& model, const GeneratorParams& params, DynamicMaskUpdate update = DynamicMaskUpdate::PreservePadding);

std::unique_ptr<AttentionMaskBuffer> CreateDynamicAttentionMaskBuffer(const Model& model, DynamicMaskUpdate update);
std::unique_ptr<AttentionMaskBuffer> CreateStaticAttentionMaskBuffer(int max_length, bool cpu_staged);

}  // namespace Generators

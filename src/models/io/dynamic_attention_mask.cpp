// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_mask.h"

#include <algorithm>

namespace Generators {

namespace {

struct DynamicAttentionMaskBuffer : AttentionMaskBuffer {
  DynamicAttentionMaskBuffer(const Model& model, DynamicMaskUpdate update) : model_{model}, update_{update} {}

  void Initialize(Tensor& tensor, std::unique_ptr<OrtValue>& values, int num_beams,
                  std::array<int64_t, 2>& shape) override {
    tensor.ort_tensor_ = model_.ExpandInputs(values, num_beams);
    shape[0] *= num_beams;
  }

  void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) override {
    shape[1] = total_length;
    Tensor next{tensor.p_device_, tensor.GetType()};
    next.CreateTensor(shape);
    if (update_ == DynamicMaskUpdate::FillOnes) {
      auto span = next.GetByteSpan();
      auto bytes = span.CpuSpan();
      if (tensor.GetType() == Ort::TypeToTensorType<int32_t>)
        std::fill_n(reinterpret_cast<int32_t*>(bytes.data()), shape[0] * shape[1], int32_t{1});
      else
        std::fill_n(reinterpret_cast<int64_t*>(bytes.data()), shape[0] * shape[1], int64_t{1});
      span.CopyCpuToDevice();
    } else {
      UpdateMaskOnDevice(tensor, &next, shape, total_length, new_length, 0);
    }
    tensor.ort_tensor_ = std::move(next.ort_tensor_);
  }

  void Rewind(Tensor&, std::array<int64_t, 2>& shape, size_t index) override {
    shape[1] = static_cast<int64_t>(index);
  }

 private:
  const Model& model_;
  DynamicMaskUpdate update_;
};

}  // namespace

std::unique_ptr<AttentionMaskBuffer> CreateDynamicAttentionMaskBuffer(const Model& model, DynamicMaskUpdate update) {
  return std::make_unique<DynamicAttentionMaskBuffer>(model, update);
}

}  // namespace Generators

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_mask.h"

#include <algorithm>

namespace Generators {

namespace {

struct StaticAttentionMaskBuffer : AttentionMaskBuffer {
  explicit StaticAttentionMaskBuffer(int max_length) : max_length_{max_length} {}

  void Initialize(Tensor& tensor, std::unique_ptr<OrtValue>& values, int num_beams,
                  std::array<int64_t, 2>& shape) override {
    if (shape[1] > max_length_)
      throw std::runtime_error("Attention mask prompt exceeds max_length");
    const auto input_shape = shape;
    shape[0] *= num_beams;
    // Shared KV providers derive the present capacity from the mask width.
    shape[1] = max_length_;
    tensor.CreateTensor(shape, true);
    auto output = tensor.GetByteSpan();
    output.Zero();
    auto input = ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *values);
    const size_t row_bytes = input_shape[1] * Ort::SizeOf(tensor.GetType());
    const size_t stride = max_length_ * Ort::SizeOf(tensor.GetType());
    for (int64_t b = 0; b < input_shape[0]; ++b)
      for (int beam = 0; beam < num_beams; ++beam)
        output.subspan((b * num_beams + beam) * stride, row_bytes).CopyFrom(input.subspan(b * row_bytes, row_bytes));
  }

  void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) override {
    if (new_length <= 0 || total_length < new_length || total_length > max_length_)
      throw std::runtime_error("Attention mask update exceeds allocated capacity");
    UpdateMaskOnDevice(tensor, nullptr, shape, total_length, new_length, max_length_);
  }

  void Rewind(Tensor& tensor, std::array<int64_t, 2>& shape, size_t index) override {
    if (index > static_cast<size_t>(max_length_))
      throw std::runtime_error("Attention mask rewind exceeds max_length");
    auto span = tensor.GetByteSpan();
    auto data = span.CopyDeviceToCpu();
    const size_t row_bytes = max_length_ * Ort::SizeOf(tensor.GetType());
    for (int64_t b = 0; b < shape[0]; ++b)
      std::fill(data.begin() + b * row_bytes + index * Ort::SizeOf(tensor.GetType()),
                data.begin() + (b + 1) * row_bytes, uint8_t{0});
    span.CopyCpuToDevice();
  }

 private:
  int max_length_;
};

// WebGPU's device mask update assumes an unpadded prefix. Keep a CPU copy so
// prompt padding and rewound entries survive uploads without device readback.
struct CpuStagedAttentionMaskBuffer : StaticAttentionMaskBuffer {
  using StaticAttentionMaskBuffer::StaticAttentionMaskBuffer;

  void Initialize(Tensor& tensor, std::unique_ptr<OrtValue>& values, int num_beams,
                  std::array<int64_t, 2>& shape) override {
    if (!cpu_mask_)
      cpu_mask_ = std::make_unique<Tensor>(GetDeviceInterface(DeviceType::CPU), tensor.GetType());
    StaticAttentionMaskBuffer::Initialize(*cpu_mask_, values, num_beams, shape);
    tensor.CreateTensor(shape, true);
    tensor.GetByteSpan().CopyFrom(cpu_mask_->GetByteSpan());
  }

  void Update(Tensor& tensor, std::array<int64_t, 2>& shape, int total_length, int new_length) override {
    StaticAttentionMaskBuffer::Update(*cpu_mask_, shape, total_length, new_length);
    tensor.GetByteSpan().CopyFrom(cpu_mask_->GetByteSpan());
  }

  void Rewind(Tensor& tensor, std::array<int64_t, 2>& shape, size_t index) override {
    StaticAttentionMaskBuffer::Rewind(*cpu_mask_, shape, index);
    tensor.GetByteSpan().CopyFrom(cpu_mask_->GetByteSpan());
  }

 private:
  std::unique_ptr<Tensor> cpu_mask_;
};

}  // namespace

std::unique_ptr<AttentionMaskBuffer> CreateStaticAttentionMaskBuffer(int max_length, bool cpu_staged) {
  if (cpu_staged)
    return std::make_unique<CpuStagedAttentionMaskBuffer>(max_length);
  return std::make_unique<StaticAttentionMaskBuffer>(max_length);
}

}  // namespace Generators

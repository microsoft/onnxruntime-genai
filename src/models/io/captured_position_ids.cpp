// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "position_ids.h"

namespace Generators {

namespace {

struct CapturedPositionIdsBuffer : PositionIdsBuffer {
  explicit CapturedPositionIdsBuffer(int max_capture_length) : max_capture_length_{max_capture_length} {}

  void Create(Tensor& tensor, std::span<const int64_t> shape) const override {
    const bool capture = shape.back() >= 1 && shape.back() <= max_capture_length_;
    // Pre-size for every captured length, including a short initial prompt.
    size_t capacity = Ort::SizeOf(tensor.GetType()) * max_capture_length_;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
      capacity *= shape[i];
    tensor.CreateTensor(shape, capture, capture ? capacity : 0);
  }

  void Assign(Tensor& tensor, const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) const override {
    auto shape = values->GetTensorTypeAndShapeInfo()->GetShape();
    if (shape.back() < 1 || shape.back() > max_capture_length_) {
      // Tensor retains its static allocation across longer, uncaptured prefills after rewind.
      PositionIdsBuffer::Assign(tensor, model, values, num_beams);
      return;
    }
    ExpandPositionIds(model, values, num_beams);
    shape = values->GetTensorTypeAndShapeInfo()->GetShape();
    Create(tensor, shape);
    tensor.GetByteSpan().CopyFrom(ByteWrapTensor(*GetDeviceInterface(DeviceType::CPU), *values));
  }

 private:
  int max_capture_length_;
};

}  // namespace

std::unique_ptr<PositionIdsBuffer> CreateCapturedPositionIdsBuffer(int max_capture_length) {
  return std::make_unique<CapturedPositionIdsBuffer>(max_capture_length);
}

}  // namespace Generators

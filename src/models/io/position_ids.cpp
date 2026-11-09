// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "position_ids.h"

namespace Generators {

// The batch axis precedes the sequence axis in both 2D IDs and 3D mRoPE IDs.
// Model::ExpandInputs expands axis 0, which is the rotary axis for mRoPE.
void PositionIdsBuffer::ExpandPositionIds(const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) {
  if (num_beams == 1)
    return;
  auto shape = values->GetTensorTypeAndShapeInfo()->GetShape();
  const auto type = values->GetTensorTypeAndShapeInfo()->GetElementType();
  const size_t row_bytes = shape.back() * Ort::SizeOf(type);
  const size_t rows = values->GetTensorTypeAndShapeInfo()->GetElementCount() / shape.back();
  shape[shape.size() - 2] *= num_beams;
  auto expanded = OrtValue::CreateTensor(model.allocator_cpu_, shape, type);
  auto& cpu = *GetDeviceInterface(DeviceType::CPU);
  auto input = ByteWrapTensor(cpu, *values);
  auto output = ByteWrapTensor(cpu, *expanded);
  for (size_t row = 0; row < rows; ++row)
    for (int beam = 0; beam < num_beams; ++beam)
      output.subspan((row * num_beams + beam) * row_bytes, row_bytes).CopyFrom(input.subspan(row * row_bytes, row_bytes));
  values = std::move(expanded);
}

void PositionIdsBuffer::Create(Tensor& tensor, std::span<const int64_t> shape) const {
  tensor.CreateTensor(shape);
}

void PositionIdsBuffer::Assign(Tensor& tensor, const Model& model, std::unique_ptr<OrtValue>& values, int num_beams) const {
  ExpandPositionIds(model, values, num_beams);
  tensor.ort_tensor_ = model.ExpandInputs(values, 1);
}

std::unique_ptr<PositionIdsBuffer> CreatePositionIdsBuffer(const GeneratorParams& params) {
  if (params.use_graph_capture)
    return CreateCapturedPositionIdsBuffer(params.max_graph_capture_length);
  return std::make_unique<PositionIdsBuffer>();
}

}  // namespace Generators

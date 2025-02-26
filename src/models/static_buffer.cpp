// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "../generators.h"
#include "model.h"
#include "static_buffer.h"

namespace Generators {

StaticBuffer::StaticBuffer(Ort::Allocator* allocator, size_t max_beam_batch_size) : allocator_{allocator}, info_{allocator_->GetInfo()}, max_beam_batch_size_{max_beam_batch_size} {
}

std::unique_ptr<OrtValue> StaticBuffer::CreateTensorOnStaticBuffer(std::span<const int64_t> shape,
                                                                   ONNXTensorElementDataType type) {
  size_t new_bytes = Ort::SizeOf(type) * GetNumElements(shape);
  if (buffer_ == nullptr) {
    // Assuming the first dimension is the batch size
    bytes_ = new_bytes * (max_beam_batch_size_ / shape[0]);
    buffer_ = allocator_->Alloc(bytes_);
    return OrtValue::CreateTensor(info_, buffer_, new_bytes, shape, type);
  }
  if (new_bytes > bytes_) {
    std::runtime_error("StaticBuffer: new_bytes > bytes_");
  }
  return OrtValue::CreateTensor(info_, buffer_, new_bytes, shape, type);
}

size_t StaticBuffer::GetNumElements(std::span<const int64_t> shape) {
  size_t num_elements = 1;
  for (auto dim : shape) {
    num_elements *= dim;
  }
  return num_elements;
}

StaticBuffer::~StaticBuffer() {
  if (buffer_ != nullptr) {
    allocator_->Free(buffer_);
  }
}

}  // namespace Generators
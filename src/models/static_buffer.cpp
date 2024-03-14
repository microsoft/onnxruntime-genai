#include "../generators.h"
#include "../span.h"
#include "static_buffer.h"

namespace Generators {

StaticBuffer::StaticBuffer(Ort::Allocator* allocator, size_t max_beam_batch_size) :
  allocator_{allocator}, info_{allocator_->GetInfo()}, max_beam_batch_size_{max_beam_batch_size} {
}

std::unique_ptr<OrtValue> StaticBuffer::GetOrCreateTensor(std::span<const int64_t> shape,
                                                          ONNXTensorElementDataType type) {
  size_t new_bytes = GetElementSize(type) * GetNumElements(shape);
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

// TODO: same as GetOrtTypeSize() in model.cc. Should be moved to a common place
size_t StaticBuffer::GetElementSize(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    default:
      throw std::runtime_error("Unsupported tensor element data type");
  }
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
#include "../generators.h"
#include "../span.h"
#include "static_buffer.h"
#include <iostream>

namespace Generators {

StaticBuffer::StaticBuffer(Ort::Allocator* allocator) : allocator_{allocator}, info_{allocator_->GetInfo()} {
}

std::unique_ptr<OrtValue> StaticBuffer::GetOrCreateTensor(std::span<const int64_t> shape,
                                                          ONNXTensorElementDataType type) {
  size_t new_bytes = GetElementSize(type) * shape.size();
  std::cout << "GetOrCreateTensor: new_bytes = " << new_bytes << std::endl;
  if (buffer_ == nullptr) {
    buffer_ = allocator_->Alloc(new_bytes);
    bytes_ = new_bytes;
    return OrtValue::CreateTensor(info_, buffer_, bytes_, shape, type);
  }
  assert(new_bytes <= bytes_);
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

StaticBuffer::~StaticBuffer() {
  if (buffer_ != nullptr) {
    allocator_->Free(buffer_);
  }
}

}  // namespace Generators
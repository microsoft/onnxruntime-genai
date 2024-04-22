#pragma once

#include <memory>
#include "../span.h"

namespace Ort {
struct Allocator;
}

namespace Generators {

struct StaticBuffer {
  // Add max_beam_batch_size to the constructor
  StaticBuffer(Ort::Allocator* allocator, size_t max_beam_batch_size);
  ~StaticBuffer();

  std::unique_ptr<OrtValue> CreateTensorOnStaticBuffer(std::span<const int64_t> shape,
                                                       ONNXTensorElementDataType type);

 private:
  size_t GetElementSize(ONNXTensorElementDataType type);
  size_t GetNumElements(std::span<const int64_t> shape);

  Ort::Allocator* allocator_{nullptr};
  const OrtMemoryInfo& info_;
  void* buffer_{nullptr};
  size_t bytes_{0};
  size_t max_beam_batch_size_{0};
};

}  // namespace Generators
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// #include <memory>
// #include "onnxruntime_api.h"
// #include "../span.h"

// namespace Ort {
// struct Allocator;
// }

// namespace Generators {

// struct StaticBuffer {
//   // Add max_beam_batch_size to the constructor
//   StaticBuffer(Ort::Allocator* allocator, size_t max_beam_batch_size);
//   StaticBuffer(const StaticBuffer&) = delete;
//   StaticBuffer& operator=(const StaticBuffer&) = delete;
//   ~StaticBuffer();

//   std::unique_ptr<OrtValue> CreateTensorOnStaticBuffer(std::span<const int64_t> shape,
//                                                        ONNXTensorElementDataType type);

//  private:
//   size_t GetNumElements(std::span<const int64_t> shape);

//   Ort::Allocator* allocator_{};
//   const OrtMemoryInfo& info_;
//   void* buffer_{};
//   size_t bytes_{};
//   size_t max_beam_batch_size_{};
// };

// }  // namespace Generators

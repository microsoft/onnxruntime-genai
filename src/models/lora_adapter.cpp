// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_api.h"
#include "lora_adapter.h"
#include "../span.h"

namespace Generators {
namespace details {
std::string LoraCacheKey(std::string_view adapter_name, std::string param_name) {
  std::string result;
  result.reserve(adapter_name.size() + param_name.size() + 1U);
  result.append(adapter_name).append(".").append(param_name);
  return result;
}

constexpr std::array<int64_t, 2> empty_2D_shape = {0, 0};

std::shared_ptr<OrtValue> CreateEmptyInput(Ort::Allocator* allocator, ONNXTensorElementDataType type) {
  return OrtValue::CreateTensor(allocator->GetInfo(), nullptr, 0, empty_2D_shape, type);
}

}  // namespace details

}  // namespace Generators
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cstdint>
#include "../models/onnxruntime_api.h"

namespace Generators {
namespace WebGPU {

// Helper to create a minimal ONNX model with a Cast operator
// This builds the protobuf manually without depending on onnx library
std::vector<uint8_t> CreateCastModelBytes(ONNXTensorElementDataType input_type, ONNXTensorElementDataType output_type);

}  // namespace WebGPU
}  // namespace Generators

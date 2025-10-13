// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <unordered_map>
#include <string>

#ifdef USE_WEBGPU
// Forward declarations for WebGPU types - must be in global scope
namespace wgpu {
class Device;
class Instance;
}  // namespace wgpu
#endif

// Forward declaration for ONNX Runtime types
struct OrtSessionOptions;

namespace Generators {

DeviceInterface* GetWebGPUInterface();
void InitWebGPUInterface();
void SetWebGPUProvider(OrtSessionOptions& session_options, const std::unordered_map<std::string, std::string>& provider_options);

}  // namespace Generators
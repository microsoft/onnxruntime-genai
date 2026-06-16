// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

struct Config;
struct DeviceInterface;
enum struct DeviceType;
struct Model;

DeviceInterface* GetQNNInterface(DeviceType device_type);

bool IsQNNGPUBackend(const Config& config);
bool IsQNNStatefulModel(const Model& model);

}  // namespace Generators
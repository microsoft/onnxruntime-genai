// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

namespace Generators {

struct Config;
struct DeviceInterface;
enum struct DeviceType;
struct Model;

// Creates a fresh QNN DeviceInterface instance. Ownership is taken by OrtGlobals.
std::unique_ptr<DeviceInterface> CreateQNNInterface(DeviceType device_type);

bool IsQNNStatefulModel(const Model& model);

}  // namespace Generators
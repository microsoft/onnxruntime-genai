// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

struct Config;
struct DeviceInterface;
struct Model;

DeviceInterface* GetQNNInterface();

bool IsQNNGPUBackend(const Config& config);
bool IsQNNStatefulModel(const Model& model);

}  // namespace Generators
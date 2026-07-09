// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

// Creates a fresh QNN DeviceInterface instance. Ownership is taken by OrtGlobals.
std::unique_ptr<DeviceInterface> CreateQNNInterface();

struct Model;
bool IsQNNStatefulModel(const Model& model);

}  // namespace Generators
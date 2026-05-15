// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

DeviceInterface* GetQNNInterface();

struct Model;
bool IsQNNStatefulModel(const Model& model);

}  // namespace Generators
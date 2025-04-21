// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

DeviceInterface* GetOpenVINOInterface();

struct Model;
bool IsOpenVINOStatefulModel(const Model& model);

}  // namespace Generators
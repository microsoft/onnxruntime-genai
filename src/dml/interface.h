// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef _WIN32
using LUID = void;
#endif

namespace Generators {

void InitDmlInterface(LUID* p_device_luid);
void SetDmlProvider(OrtSessionOptions& options);

DeviceInterface* GetDmlInterface();

}  // namespace Generators
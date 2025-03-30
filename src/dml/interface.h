// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
#include <windows.h>
#else
typedef struct _LUID {
  uint32_t LowPart;
  int32_t HighPart;
} LUID, *PLUID;
#endif

struct OrtSessionOptions;

namespace Generators {
struct DeviceInterface;

void InitDmlInterface(LUID* p_device_luid);
void SetDmlProvider(OrtSessionOptions& options);

DeviceInterface* GetDmlInterface();

}  // namespace Generators

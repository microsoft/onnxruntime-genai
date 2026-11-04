// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>

struct GenaiInterface;
struct OrtApi;
namespace Generators {
struct DeviceInterface;
}

#if defined(_WIN32)
#define LEGACY_ADDON_EXPORT __declspec(dllexport)
#else
#define LEGACY_ADDON_EXPORT __attribute__((visibility("default")))
#endif

namespace {
uint32_t interface_calls{};
}

extern "C" {
LEGACY_ADDON_EXPORT uint32_t GetInterfaceVersion() {
  return 19;
}

LEGACY_ADDON_EXPORT Generators::DeviceInterface* GetInterface(GenaiInterface*, const char*, const OrtApi*) {
  ++interface_calls;
  return nullptr;
}

LEGACY_ADDON_EXPORT uint32_t GetLegacyInterfaceCallCount() {
  return interface_calls;
}
}

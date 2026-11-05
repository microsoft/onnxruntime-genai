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
uint32_t interface_version{19};
uint32_t interface_calls{};
}  // namespace

extern "C" {
LEGACY_ADDON_EXPORT uint32_t GetInterfaceVersion() {
  return interface_version;
}

LEGACY_ADDON_EXPORT void SetLegacyInterfaceVersion(uint32_t version) {
  interface_version = version;
}

LEGACY_ADDON_EXPORT Generators::DeviceInterface* GetInterface(GenaiInterface*, const char*, const OrtApi*) {
  ++interface_calls;
  return nullptr;
}

LEGACY_ADDON_EXPORT uint32_t GetLegacyInterfaceCallCount() {
  return interface_calls;
}
}

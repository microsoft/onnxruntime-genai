// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pal/PAL.hpp"

#include <string_view>

#if defined(_WIN32)
#define TELEMETRY_RELOAD_EXPORT __declspec(dllexport)
#else
#define TELEMETRY_RELOAD_EXPORT __attribute__((visibility("default")))
#endif

extern "C" TELEMETRY_RELOAD_EXPORT const char* TelemetryReloadSdkVersion() {
  // Also accepts the old string-returning getter for the unload leak negative control.
  return std::string_view(Microsoft::Applications::Events::PlatformAbstraction::getSdkVersion()).data();
}

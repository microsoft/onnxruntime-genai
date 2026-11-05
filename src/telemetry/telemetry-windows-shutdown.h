// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(_WIN32)
#include <Windows.h>

namespace Generators::TelemetryInternal {

inline bool IsDllShutdownInProgress() noexcept {
  using RtlDllShutdownInProgressFn = BOOLEAN(NTAPI*)();
  const HMODULE ntdll = ::GetModuleHandleW(L"ntdll.dll");
  const auto is_shutting_down =
      ntdll == nullptr ? nullptr
                       : reinterpret_cast<RtlDllShutdownInProgressFn>(
                             ::GetProcAddress(ntdll, "RtlDllShutdownInProgress"));
  return is_shutting_down != nullptr && is_shutting_down();
}

}  // namespace Generators::TelemetryInternal
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Compile the production implementation in this TU so the test peer can install an inert uploader.
#include "telemetry/telemetry.cpp"
#include "NullObjects.hpp"

#include <cwchar>

namespace {
namespace sdk = Microsoft::Applications::Events;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s (Win32 error: %lu)\n", message, GetLastError());
    std::fflush(stderr);
    TerminateProcess(GetCurrentProcess(), EXIT_FAILURE);
    std::abort();
  }
}

int release_count = 0;
sdk::ILogConfiguration* installed_config = nullptr;

class MockLogManager final : public sdk::NullLogManager {
 public:
  explicit MockLogManager(bool pending) : callback_(CreateEventW(nullptr, TRUE, !pending, nullptr)) {
    Check(callback_ != nullptr, "cannot create mock callback");
  }
  ~MockLogManager() override { CloseHandle(callback_); }

  sdk::status_t Flush() override {
    ++flush_count;
    return sdk::STATUS_SUCCESS;
  }

  void FlushAndTeardown() override {
    Check(flush_count == 1, "expected Flush before teardown");
    ++teardown_count;
    Check(WaitForSingleObject(callback_, 2000) == WAIT_OBJECT_0, "cannot drain callback after process termination");
  }

  int flush_count = 0;
  int teardown_count = 0;

 private:
  HANDLE callback_;
};
}  // namespace

namespace Microsoft::Applications::Events {
// Replace provider ownership only in this DLL. No production uploader can be created or transmitted.
ILogManager* LogManagerProvider::Get(ILogConfiguration&, status_t&) {
  Check(false, "unexpected real uploader initialization");
  return nullptr;
}
ILogManager* LogManagerProvider::Get(const char*, status_t&) {
  Check(false, "unexpected real uploader initialization");
  return nullptr;
}
status_t LogManagerProvider::Release(const char*) {
  Check(false, "unexpected named uploader release");
  return STATUS_EFAIL;
}
status_t LogManagerProvider::Release(ILogConfiguration& config) {
  Check(&config == installed_config, "released a different uploader configuration");
  ++release_count;
  return STATUS_SUCCESS;
}
}  // namespace Microsoft::Applications::Events

namespace Generators {
class GenAiTelemetryWindowsTestPeer {
 public:
  static GenAiTelemetry* Create(sdk::ILogManager& manager) {
    auto* telemetry = new GenAiTelemetry();
    telemetry->impl_ = std::make_unique<GenAiTelemetry::Impl>();
    telemetry->impl_->log_manager = &manager;
    telemetry->initialized_.store(true);
    installed_config = &telemetry->impl_->config;
    return telemetry;
  }

  static bool IsClean(const GenAiTelemetry& telemetry) {
    return !telemetry.impl_ && !telemetry.initialized_.load();
  }

  static void Destroy(GenAiTelemetry* telemetry) { delete telemetry; }

  static void Lock(GenAiTelemetry& telemetry) { telemetry.mutex_.lock(); }
};
}  // namespace Generators

namespace {
using Peer = Generators::GenAiTelemetryWindowsTestPeer;

class TelemetryLifetime {
 public:
  TelemetryLifetime(bool process_exit, bool explicit_shutdown, HANDLE verified)
      : manager_(process_exit),
        process_exit_(process_exit),
        explicit_shutdown_(explicit_shutdown),
        verified_(verified),
        telemetry_(Peer::Create(manager_)) {}

  ~TelemetryLifetime() {
    if (telemetry_) Destroy();
  }

  void Destroy() {
    Check(Generators::TelemetryInternal::IsDllShutdownInProgress() == process_exit_, "unexpected DLL shutdown state");
    Check(Generators::GenAiTelemetry::IsDestroyed() == process_exit_, "unexpected telemetry destruction state");
    if (explicit_shutdown_) {
      telemetry_->Shutdown();
      Check(Peer::IsClean(*telemetry_) != process_exit_, "unexpected shutdown state retention");
      telemetry_->Shutdown();
    }
    Peer::Destroy(telemetry_);
    telemetry_ = nullptr;
    const int expected = process_exit_ ? 0 : 1;
    Check(manager_.flush_count == expected, "unexpected flush count");
    Check(manager_.teardown_count == expected, "unexpected teardown count");
    Check(release_count == expected, "unexpected provider release count");
    Check(SetEvent(verified_) != FALSE, "cannot signal verified destruction");
  }

  void HoldLockUntilExit() {
    locked_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    Check(locked_ != nullptr, "cannot create lock handshake");
    const HANDLE thread = CreateThread(nullptr, 0, HoldLock, this, 0, nullptr);
    Check(thread != nullptr, "cannot create lock owner");
    Check(WaitForSingleObject(locked_, 2000) == WAIT_OBJECT_0, "worker did not acquire telemetry mutex");
    CloseHandle(thread);
  }

 private:
  static DWORD WINAPI HoldLock(void* context) {
    auto& lifetime = *static_cast<TelemetryLifetime*>(context);
    Peer::Lock(*lifetime.telemetry_);
    Check(SetEvent(lifetime.locked_) != FALSE, "cannot signal acquired mutex");
    Sleep(INFINITE);
    return 0;
  }

  MockLogManager manager_;
  bool process_exit_;
  bool explicit_shutdown_;
  HANDLE verified_;
  Generators::GenAiTelemetry* telemetry_;
  HANDLE locked_ = nullptr;
};
}  // namespace

extern "C" __declspec(dllexport) void GenAiTelemetryTestLifecycle(const wchar_t* mode, HANDLE verified) {
  const bool process_exit = std::wcscmp(mode, L"process-exit") == 0 ||
                            std::wcscmp(mode, L"locked-process-exit") == 0 ||
                            std::wcscmp(mode, L"shutdown-locked-process-exit") == 0;
  // This DLL-owned owner is destroyed by the CRT during real DLL process detach.
  static TelemetryLifetime lifetime(process_exit,
                                    std::wcscmp(mode, L"explicit-shutdown") == 0 ||
                                        std::wcscmp(mode, L"shutdown-locked-process-exit") == 0,
                                    verified);
  if (std::wcscmp(mode, L"locked-process-exit") == 0 ||
      std::wcscmp(mode, L"shutdown-locked-process-exit") == 0) lifetime.HoldLockUntilExit();
  if (std::wcscmp(mode, L"normal-destruction") == 0 ||
      std::wcscmp(mode, L"explicit-shutdown") == 0) lifetime.Destroy();
}

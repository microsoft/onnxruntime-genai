// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry/telemetry.h"
#include "DebugEvents.hpp"
#include "ITaskDispatcher.hpp"
#include "mat/config.h"
#include "http/HttpClientFactory.hpp"
#include "pal/PAL.hpp"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string_view>
#include <system_error>
#include <thread>

#if defined(_WIN32)
#include <Windows.h>
#endif

namespace {
namespace sdk = Microsoft::Applications::Events;
namespace pal = Microsoft::Applications::Events::PlatformAbstraction;
using namespace std::chrono_literals;

[[noreturn]] void Fail(int code, const char* message) {
  std::fprintf(stderr, "FAIL: %s\n", message);
#if defined(_WIN32)
  // _Exit can wait for the CRT exit lock held by the thread currently joining this SDK worker.
  std::fflush(stderr);
  TerminateProcess(GetCurrentProcess(), static_cast<UINT>(code));
#endif
  std::_Exit(code);
}

struct State {
  std::mutex mutex;
  std::condition_variable changed;
  sdk::DebugEventSource source;
  std::weak_ptr<sdk::ITaskDispatcher> worker;
  std::thread::id construction_thread = std::this_thread::get_id();
  void* worker_address = nullptr;
  bool owner_ready = false;
  bool no_worker_expected = false;
  bool mutex_before_owner = false;
  bool mutex_retired = false;
  bool callback_started = false;
  bool callback_completed = false;
  bool release_callback = false;
  bool worker_joined = false;
  bool upload_at_exit = false;
  bool encode_started = false;
  bool version_destroyed = false;
  int version_reads = 0;
  int uploads_completed = 0;
  int workers_started = 0;
  int collector_events = 0;
  int priming_failure = 0;
  int priming_attempts = 0;

  ~State() {
    if (priming_failure != 0) {
      if (priming_attempts != 1 || mutex_before_owner || mutex_retired) {
        Fail(32, "failed priming was retried or unexpectedly initialized the SDK mutex");
      }
    } else if (!mutex_before_owner || !mutex_retired) {
      Fail(32, "SDK mutex initialization and retirement were not exercised");
    }
    if (no_worker_expected) {
      if (workers_started != 0 || collector_events != 0) {
        Fail(32, "priming or suppression started a worker or emitted a collector event");
      }
    } else if ((!upload_at_exit && (!callback_started || !callback_completed)) ||
               !worker_joined || !worker.expired()) {
      Fail(32, "owner, native SDK worker, and mutex teardown were not fully exercised");
    }
    if (upload_at_exit && (!encode_started || uploads_completed == 0 || version_reads < 2)) {
      Fail(32, "shutdown did not exercise production decoration, encoding, and upload completion");
    }
    std::fputs("PASS: actual telemetry owner preserved SDK dependency lifetimes\n", stderr);
  }
};

State& GetState() {
  // This is constructed by the first SDK lock hook, before the SDK mutex, including before main.
  static State state;
  return state;
}

struct MutexLifetime {
  MutexLifetime() {
    auto& state = GetState();
    std::lock_guard lock(state.mutex);
    state.mutex_before_owner = !state.owner_ready;
  }

  ~MutexLifetime() {
    auto& state = GetState();
    std::lock_guard lock(state.mutex);
    // The observer is registered immediately after the real mutex. Fail before libc++ destroys it.
    if (!state.no_worker_expected &&
        ((!state.upload_at_exit && !state.callback_completed) || !state.worker_joined || !state.worker.expired())) {
      Fail(21, "SDK mutex retired before its actual owner joined the worker");
    }
    state.mutex_retired = true;
  }
};

class DiagnosticTask final : public sdk::Task {
 public:
  DiagnosticTask() {
    Type = Call;
    TargetTime = 0;
    TypeName = "TelemetryShutdownRegression";
  }

  void operator()() override {
    auto& state = GetState();
    if (std::this_thread::get_id() == state.construction_thread) {
      Fail(32, "diagnostic task did not run on an SDK worker");
    }
    state.source.DispatchEvent({});
    {
      std::unique_lock lock(state.mutex);
      state.callback_started = true;
      state.changed.notify_all();
      if (!state.changed.wait_for(lock, 30s, [&] { return state.release_callback; })) {
        Fail(32, "process exit did not release the SDK diagnostic task");
      }
    }
    state.source.DispatchEvent({});
    std::lock_guard lock(state.mutex);
    state.callback_completed = true;
  }
};

void ReleaseCallback() {
  auto& state = GetState();
  std::lock_guard lock(state.mutex);
  state.release_callback = true;
  state.changed.notify_all();
}

void InitializeOwner(bool initialize_sdk) {
  auto& state = GetState();
  {
    std::lock_guard lock(state.mutex);
    state.construction_thread = std::this_thread::get_id();
  }
  auto& telemetry = Generators::GenAiTelemetry::Instance();
  {
    std::lock_guard lock(state.mutex);
    state.owner_ready = true;
    state.changed.notify_all();
  }
  if (initialize_sdk) {
    telemetry.Initialize();
    if (telemetry.IsEnabled() == state.no_worker_expected) {
      Fail(32, "unexpected telemetry availability (check suppression and SDK initialization)");
    }
  }
}

class LocalHttpClient final : public sdk::IHttpClient {
 public:
  sdk::IHttpRequest* CreateRequest() override { return new sdk::SimpleHttpRequest(pal::generateUuidString()); }

  void SendRequestAsync(sdk::IHttpRequest* request, sdk::IHttpResponseCallback* callback) override {
    if (request->GetHeaders().get("SDK-Version").empty() || request->GetBody().empty()) {
      Fail(32, "upload did not contain an encoded event and SDK version");
    }
    auto* response = new sdk::SimpleHttpResponse(request->GetId());
    response->m_result = sdk::HttpResult_OK;
    response->m_statusCode = 200;
    {
      auto& state = GetState();
      std::lock_guard lock(state.mutex);
      ++state.uploads_completed;
    }
    callback->OnHttpResponse(response);
  }

  void CancelRequestAsync(const std::string&) override {}
};
}  // namespace

namespace Microsoft::Applications::Events {
// Keep production event processing, but never transmit collector data from a regression process.
std::shared_ptr<IHttpClient> HttpClientFactory::Create() { return std::make_shared<LocalHttpClient>(); }
}  // namespace Microsoft::Applications::Events

extern "C" void TelemetryTestVersionDestroyed() {
  auto& state = GetState();
  std::lock_guard lock(state.mutex);
  state.version_destroyed = true;
}

extern "C" void TelemetryTestVersionAccess() {
  auto& state = GetState();
  std::lock_guard lock(state.mutex);
  if (state.version_destroyed) Fail(21, "SDK read its version after string destruction");
  ++state.version_reads;
}

extern "C" void TelemetryTestBeforeEncode() {
  auto& state = GetState();
  std::unique_lock lock(state.mutex);
  if (!state.upload_at_exit) return;
  state.encode_started = true;
  state.changed.notify_all();
  if (!state.changed.wait_for(lock, 10s, [&] { return state.release_callback; })) {
    Fail(32, "process exit did not release production upload encoding");
  }
}

void TelemetryTestBeforeDebugLock() {
  auto& state = GetState();
  std::unique_lock lock(state.mutex);
  if (state.priming_failure != 0) {
    ++state.priming_attempts;
    if (state.priming_failure == 1) {
      throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
    }
    throw std::bad_alloc();
  }
  if (std::this_thread::get_id() != state.construction_thread &&
      !state.changed.wait_for(lock, 10s, [&] { return state.owner_ready; })) {
    Fail(32, "initialization depended on a background diagnostic");
  }
  if (state.mutex_retired) {
    Fail(21, "SDK accessed its diagnostic mutex after retirement");
  }
}

extern "C" void TelemetryTestDebugLockInitialized() {
  static MutexLifetime lifetime;
}

extern "C" void TelemetryTestWorkerStarted(void*) {
  auto& state = GetState();
  std::lock_guard lock(state.mutex);
  ++state.workers_started;
}

extern "C" void TelemetryTestWorkerJoined(void* address) {
  auto& state = GetState();
  std::lock_guard lock(state.mutex);
  if (address == state.worker_address) state.worker_joined = true;
}

extern "C" void TelemetryTestCollectorEvent(const char*) {
  auto& state = GetState();
  std::lock_guard lock(state.mutex);
  ++state.collector_events;
}

int main(int argc, char* argv[]) {
  if (argc != 2) Fail(32, "expected a shutdown regression mode");
  const std::string_view mode{argv[1]};
  if (mode != "retained" && mode != "threaded" && mode != "recreated" &&
      mode != "explicit" && mode != "prime-only" && mode != "disabled" && mode != "ci-disabled" &&
      mode != "prime-failed" && mode != "prime-allocation-failed" &&
      mode != "upload-at-exit" && mode != "upload-after-reinit") {
    Fail(32, "unknown shutdown regression mode");
  }
  auto& state = GetState();
  state.upload_at_exit = mode == "upload-at-exit" || mode == "upload-after-reinit";
  // Register after the singleton, but before the first event lazily initializes the SDK version.
  // The original SDK destroys that version before this callback releases the blocked encoder.
  if (state.upload_at_exit) {
    InitializeOwner(false);
    if (std::atexit(ReleaseCallback) != 0) Fail(32, "cannot register production upload release");
  }
  state.priming_failure = mode == "prime-failed" ? 1 : mode == "prime-allocation-failed" ? 2
                                                                                         : 0;
  state.no_worker_expected =
      mode == "prime-only" || mode == "disabled" || mode == "ci-disabled" || state.priming_failure != 0;
  if (mode == "threaded") {
    std::thread initializer([] { InitializeOwner(true); });
    initializer.join();
  } else {
    InitializeOwner(mode != "prime-only");
  }
  if (state.priming_failure != 0) {
    auto& telemetry = Generators::GenAiTelemetry::Instance();
    telemetry.Shutdown();
    telemetry.Initialize();
    if (telemetry.IsEnabled()) Fail(32, "failed priming did not permanently suppress telemetry");
  }
  if (state.no_worker_expected) return 0;

  if (mode == "recreated" || mode == "upload-after-reinit") {
    auto previous_worker = pal::getDefaultTaskDispatcher();
    std::weak_ptr<sdk::ITaskDispatcher> previous = previous_worker;
    previous_worker.reset();
    Generators::GenAiTelemetry::Instance().Shutdown();
    if (!previous.expired()) Fail(32, "ordinary shutdown retained the SDK worker");
    InitializeOwner(true);
  }

  if (state.upload_at_exit) {
    auto dispatcher = pal::getDefaultTaskDispatcher();
    if (!dispatcher) Fail(32, "actual SDK upload dispatcher unavailable");
    {
      std::lock_guard lock(state.mutex);
      state.worker = dispatcher;
      state.worker_address = dispatcher.get();
    }
    dispatcher.reset();
    Generators::GenAiTelemetry::Instance().LogProcessInfo();
    std::unique_lock lock(state.mutex);
    if (!state.changed.wait_for(lock, 30s, [&] { return state.encode_started; })) {
      Fail(32, "production event never reached upload encoding");
    }
    return 0;
  }

  if (std::atexit(ReleaseCallback) != 0) Fail(32, "cannot register task release");
  {
    auto dispatcher = pal::getDefaultTaskDispatcher();
    if (!dispatcher) Fail(32, "actual SDK dispatcher unavailable");
    {
      std::lock_guard lock(state.mutex);
      state.worker = dispatcher;
      state.worker_address = dispatcher.get();
    }
    dispatcher->Queue(new DiagnosticTask());
  }
  {
    std::unique_lock lock(state.mutex);
    if (!state.changed.wait_for(lock, 10s, [&] { return state.callback_started; })) {
      Fail(32, "SDK task never started");
    }
  }
  if (mode == "explicit") {
    ReleaseCallback();
    Generators::GenAiTelemetry::Instance().Shutdown();
    if (!state.worker.expired()) Fail(32, "explicit shutdown retained the SDK worker");
  }
  return 0;
}

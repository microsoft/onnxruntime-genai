// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Dedicated test binary for GenAI shutdown / re-initialization.
//
// These tests call OgaShutdown() to prove GenAI can be torn down and re-initialized in-process
// (for example a host such as Foundry Local's `Manager` that recreates its wrapper in-process).
// They live in their OWN executable rather than in unit_tests because OgaShutdown() resets
// process-global GenAI state (the ORT environment, the device interfaces, the genai add-on
// libraries, and any registered plugin EP libraries). Running them inside the shared suite would
// couple every other test to their teardown and depend on test ordering; isolating them removes
// that fragility. Re-init is always exercised on the always-present CPU path; if plugin EPs are
// discovered (WinML packages and/or --ep_dir) they are additionally registered and torn down each
// cycle.

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <filesystem>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ort_genai.h"
#include "generator/generators.h"  // Generators::GetDeviceInterface / DeviceType (from the genai object library)
#include "models/graph_executor.h"
#include "models/io/kv_cache.h"
#include "search.h"
#include "telemetry/telemetry.h"
#if USE_DML
#include "ep/dml/interface.h"  // Generators::InitDmlInterface / CloseDmlInterface / GetDmlInterface
#endif
#if USE_CUDA
#include "ep/cuda/session_options.h"  // Generators::CUDAExecutionProvider::ResolveStreamBindingOptions
#include "models/session_options.h"
#include "models/qwen38_graph_plane.h"  // Generators::Qwen38Plane: the plane that pins the CUDA add-on
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

#include <gtest/gtest.h>

#include "ep_registration.h"
#include "telemetry_test_environment.h"
#include "test_utils.h"

// Plugin EPs discovered at startup (WinML packages by default, plus any --ep_dir). Registered on
// genai's env at the start of each re-init cycle by the test below, so their device interfaces are
// created and then torn down by OgaShutdown() each cycle. (generators.h defines a global `fs` type,
// so this file qualifies std::filesystem explicitly rather than adding its own alias.)
test_ep::EpRegistrar g_ep_registrar;

namespace {

// Runs a minimal CPU generate workload and validates the output. Every GenAI object is destroyed
// before returning, honoring the lifetime contract that no object holding device memory may
// outlive a subsequent OgaShutdown().
void RunCpuWorkload() {
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};
  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};
  const int batch_size = 2;
  const int max_length = 10;

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);
  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone())
    generator->GenerateNextToken();

  for (int i = 0; i < batch_size; ++i) {
    const auto sequence_length = generator->GetSequenceCount(i);
    const auto* sequence_data = generator->GetSequenceData(i);
    ASSERT_LE(sequence_length, static_cast<size_t>(max_length));
    EXPECT_TRUE(0 == std::memcmp(&expected_output[i * max_length], sequence_data,
                                 sequence_length * sizeof(int32_t)));
  }
}

// Loads the tiny model (optionally forced onto `provider` via the genai config) and runs a short
// generate, destroying every GenAI object before returning. `provider` empty => the model's
// configured default (CPU for the test model). Throws if the provider is not built in / unavailable.
void RunGenerateOnce(const char* provider) {
  auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  if (provider && *provider) {
    config->ClearProviders();
    config->AppendProvider(provider);
  }

  std::vector<int32_t> input_ids{0, 0, 0, 52};
  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 10);
  params->SetSearchOption("batch_size", 1);
  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone())
    generator->GenerateNextToken();
  ASSERT_GT(generator->GetSequenceCount(0), 0u);
}

// Forces creation of the DeviceInterface for `provider` directly via genai's internal
// GetDeviceInterface (no model load), so the create -> teardown -> recreate path is exercised across
// re-init cycles (for the CUDA add-on this also loads and later unloads the add-on library).
// The created interface is owned by OrtGlobals, so the subsequent OgaShutdown() tears it down.
void ExerciseDeviceInterface(const std::string& provider) {
  Generators::DeviceType device_type;
  if (provider == "WebGPU") {
    device_type = Generators::DeviceType::WEBGPU;
  } else if (provider == "cuda") {
    device_type = Generators::DeviceType::CUDA;
  } else if (provider == "OpenVINO") {
    device_type = Generators::DeviceType::OpenVINO;
  } else if (provider == "NvTensorRtRtx") {
    device_type = Generators::DeviceType::NvTensorRtRtx;
  } else {
    return;  // no genai DeviceInterface mapping for this provider
  }

  // Creating the interface registers it with OrtGlobals (so OgaShutdown tears it down / unloads the
  // CUDA add-on). It may throw if the EP/add-on is not usable on this machine (e.g. the CUDA add-on
  // library is absent), which the caller treats as skip.
  Generators::GetDeviceInterface(device_type);
}

}  // namespace

// Primary driver: after OgaShutdown() GenAI returns to a just-loaded state and the next
// GenAI call re-initializes cleanly with a fresh OrtEnv. Each cycle registers the discovered plugin
// EPs (on the fresh env), runs a CPU workload plus a workload per available provider (to create and
// then tear down each EP's device interface), and calls OgaShutdown(). Repeating this proves that
// repeated register -> use -> shutdown -> re-init is stable, including reloading the CUDA add-on.
TEST(ReInitTests, ShutdownReInitCycle) {
  const int kCycles = 3;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    SCOPED_TRACE("re-init cycle " + std::to_string(cycle));

    // Register discovered plugin EPs on the current (fresh, after the previous OgaShutdown) env.
    const std::vector<std::string>& providers = g_ep_registrar.RegisterAll();

    RunCpuWorkload();  // always-present CPU path (full model/session/generator teardown)

    // Force creation + teardown of each available EP's device interface. Best-effort: an EP's genai
    // add-on may be absent (e.g. CUDA without onnxruntime-genai-cuda) or its allocator unavailable,
    // so a failure just skips that provider this cycle.
    for (const std::string& provider : providers) {
      try {
        ExerciseDeviceInterface(provider);
        std::cout << "cycle " << cycle << ": exercised device interface for '" << provider << "'"
                  << std::endl;
      } catch (const std::exception& e) {
        std::cout << "cycle " << cycle << ": skipped provider '" << provider << "': " << e.what()
                  << std::endl;
      }
    }

    OgaShutdown();  // full teardown: env + all created device interfaces + add-on libraries
    EXPECT_FALSE(Generators::GenAiTelemetry::IsDestroyed());
  }
}

// Regression guard for the DeviceInterface cache in OrtGlobals::GetDeviceInterface. Creating a
// model, destroying it, then creating another model within the SAME env cycle (no OgaShutdown
// between) must hand the second model a live DeviceInterface, not a dangling one.
//
// The CPU path exercises the real model create -> free -> create seam on every build. DML is the
// case that actually motivated the OrtGlobals no-cache fix, but the tiny CPU test model cannot run
// on DML (DML forces graph capture, which requires full DML partitioning), so DML is validated
// directly at the interface level in the DmlInterfaceNotCachedAcrossReload test below.
TEST(ReInitTests, SequentialModelLoads) {
  RunGenerateOnce("");  // first load (default provider == CPU for the test model)
  RunGenerateOnce("");  // second load after the first was fully destroyed -- must not dangle

  OgaShutdown();  // leave process-global state clean for any following test
}

class DeviceResourceReleaseTests : public testing::Test {
 protected:
  void SetUp() override { OgaShutdown(); }
  void TearDown() override { OgaShutdown(); }
};

TEST_F(DeviceResourceReleaseTests, FailedCudaModelCreationThenRepeatedRelease) {
  std::string initialization_error;
  {
    auto config = OgaConfig::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
    config->ClearProviders();
    config->AppendProvider("cuda");
    try {
      auto model = OgaModel::Create(*config);
    } catch (const std::exception& e) {
      initialization_error = e.what();
    }
  }  // Destroy the config and any model objects before releasing device resources.

  if (initialization_error.find("Cuda interface not available:") == std::string::npos)
    GTEST_SKIP() << "Requires an unavailable CUDA add-on; initialization result: " << initialization_error;

  EXPECT_NO_THROW(OgaCheckResult(OgaReleaseDeviceResources("CUDA")));
  EXPECT_NO_THROW(OgaCheckResult(OgaReleaseDeviceResources("CUDA")));
}

TEST_F(DeviceResourceReleaseTests, CudaReleaseThenReinitialize) {
#if USE_CUDA
  Generators::DeviceInterface* device{};
  try {
    device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
    const Generators::ProviderOptions options{"cuda", {}};
    (void)device->GetDeviceId(&options);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "CUDA is unavailable: " << e.what();
  }

  ASSERT_NO_THROW(RunGenerateOnce("cuda"));
  auto& allocator = Generators::GetOrtGlobals()->device_allocators_[static_cast<int>(Generators::DeviceType::CUDA)];
  ASSERT_NE(allocator.allocator_, nullptr);
  for (int cycle = 0; cycle < 2; ++cycle) {
    SCOPED_TRACE("release/reinitialize cycle " + std::to_string(cycle));
    ASSERT_NO_THROW(OgaCheckResult(OgaReleaseDeviceResources("CUDA")));
    ASSERT_NO_THROW(OgaCheckResult(OgaReleaseDeviceResources("CUDA")));
    EXPECT_EQ(allocator.allocator_, nullptr);
    EXPECT_EQ(allocator.session_, nullptr);
    EXPECT_EQ(Generators::GetDeviceInterface(Generators::DeviceType::CUDA), device);

    ASSERT_NO_THROW(RunGenerateOnce("cuda"));
    EXPECT_NE(allocator.allocator_, nullptr);
    EXPECT_NE(allocator.session_, nullptr);
  }
#else
  GTEST_SKIP() << "CUDA execution provider is not enabled in this build (USE_CUDA=OFF).";
#endif
}

namespace {

constexpr auto kConcurrencyTimeout = std::chrono::seconds(10);

void ConcurrencyTimeout(const char* operation) {
  std::cerr << "Timed out waiting for " << operation << std::endl;
  std::abort();  // Joining a deadlocked worker would otherwise hang the test executable.
}

class TestEvent {
 public:
  void Set() {
    std::scoped_lock lock{mutex_};
    set_ = true;
    cv_.notify_all();
  }

  bool WaitFor(std::chrono::milliseconds timeout) {
    std::unique_lock lock{mutex_};
    return cv_.wait_for(lock, timeout, [&] { return set_; });
  }

  void Wait() {
    if (!WaitFor(kConcurrencyTimeout))
      ConcurrencyTimeout("test barrier");
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool set_{};
};

class ConcurrentOperation {
 public:
  explicit ConcurrentOperation(std::function<void()> operation) {
    std::packaged_task<void()> task{std::move(operation)};
    result_ = task.get_future();
    thread_ = std::thread{std::move(task)};
  }

  ~ConcurrentOperation() { Join(); }

  bool WaitFor(std::chrono::milliseconds timeout) {
    return result_.wait_for(timeout) == std::future_status::ready;
  }

  void Get() {
    Join();
    result_.get();
  }

 private:
  void Join() {
    if (!thread_.joinable())
      return;
    if (!WaitFor(kConcurrencyTimeout))
      ConcurrencyTimeout("concurrent operation");
    thread_.join();
  }

  std::future<void> result_;
  std::thread thread_;
};

// Intercept only the synchronization points; all sessions, allocators and their destruction are real ORT operations.
class ScopedOrtCallbacks {
 public:
  ScopedOrtCallbacks() : original_{Ort::api}, api_{*original_} {
    current_ = this;
    api_.CreateSessionOptions = CreateSessionOptions;
    api_.ReleaseAllocator = ReleaseAllocator;
    api_.ReleaseSession = ReleaseSession;
    api_.ReleaseEnv = ReleaseEnv;
    Ort::api = &api_;
  }

  ~ScopedOrtCallbacks() {
    Ort::api = original_;
    current_ = nullptr;
  }

  std::function<void()> on_session_options;
  std::function<void()> on_release_allocator;
  std::function<void(OrtSession*)> on_release_session;
  std::function<void()> on_release_env;

 private:
  static OrtStatus* ORT_API_CALL CreateSessionOptions(OrtSessionOptions** options) noexcept {
    if (current_->on_session_options)
      current_->on_session_options();
    return current_->original_->CreateSessionOptions(options);
  }

  static void ORT_API_CALL ReleaseAllocator(OrtAllocator* allocator) noexcept {
    if (current_->on_release_allocator)
      current_->on_release_allocator();
    current_->original_->ReleaseAllocator(allocator);
  }

  static void ORT_API_CALL ReleaseSession(OrtSession* session) noexcept {
    if (current_->on_release_session)
      current_->on_release_session(session);
    current_->original_->ReleaseSession(session);
  }

  static void ORT_API_CALL ReleaseEnv(OrtEnv* env) noexcept {
    if (current_->on_release_env)
      current_->on_release_env();
    current_->original_->ReleaseEnv(env);
  }

  const OrtApi* original_;
  OrtApi api_;
  inline static ScopedOrtCallbacks* current_{};
};

// A CPU-backed stand-in for a second device: exercises the full non-CPU EnsureDeviceOrtInit path
// without requiring an installed plugin EP. No device allocations or inference are mocked.
struct InitTestDevice : Generators::DeviceInterface {
  Generators::DeviceType GetType() const override { return Generators::DeviceType::QnnHtp; }
  std::string GetExecutionProviderName() const override { return "CPU"; }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return OrtMemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  }
  void InitOrt(const OrtApi&, Ort::Allocator& allocator) override { allocator_ = &allocator; }
  Ort::Allocator& GetAllocator() override { return *allocator_; }
  std::shared_ptr<Generators::DeviceBuffer> AllocateBase(size_t) override { return {}; }
  std::shared_ptr<Generators::DeviceBuffer> WrapMemoryBase(void*, size_t) override { return {}; }
  std::unique_ptr<Generators::Search> CreateGreedy(const Generators::GeneratorParams&) override { return {}; }
  std::unique_ptr<Generators::Search> CreateBeam(const Generators::GeneratorParams&) override { return {}; }
  std::unique_ptr<Generators::KeyValueCache> CreateKeyValueCache(Generators::State&) override { return {}; }
  void Synchronize() override {}
  Ort::Allocator* allocator_{};
};

class DeviceResourceReleaseConcurrencyTests : public testing::Test {
 protected:
  void SetUp() override {
    OgaShutdown();
    globals_ = Generators::GetOrtGlobals().get();
  }

  void TearDown() override { OgaShutdown(); }

  std::unique_ptr<OrtSession> CreateCpuSession() {
    auto options = OrtSessionOptions::Create();
    options->SetIntraOpNumThreads(1);
    const std::filesystem::path path{MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32/past.onnx"};
    return OrtSession::Create(Generators::GetOrtEnv(), path.c_str(), options.get());
  }

  void SeedCudaResources() {
    // Tag real CPU resources as CUDA so the production cleanup and locks run on CPU-only CI too.
    globals_->graph_session_cache_.sessions_.emplace(
        0, Generators::OrtGlobals::SessionCache::Entry{Generators::DeviceType::CUDA, CreateCpuSession()});
    auto& allocator = globals_->device_allocators_[static_cast<int>(Generators::DeviceType::CUDA)];
    allocator.session_ = CreateCpuSession();
    auto memory_info = OrtMemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    allocator.allocator_ = Ort::Allocator::Create(*allocator.session_, *memory_info);
  }

  void WaitForReleasePin() {
    const auto deadline = std::chrono::steady_clock::now() + kConcurrencyTimeout;
    while (globals_->active_resource_releases_.load() == 0) {
      if (std::chrono::steady_clock::now() >= deadline)
        ConcurrencyTimeout("resource-release lifetime pin");
      std::this_thread::yield();
    }
  }

  Generators::OrtGlobals* globals_{};
};

}  // namespace

TEST_F(DeviceResourceReleaseConcurrencyTests, CudaReleaseWithWebGpuCacheMiss) {
  SeedCudaResources();
  TestEvent cache_miss, continue_cache_miss;
  ScopedOrtCallbacks callbacks;
  callbacks.on_session_options = [&] {
    // GetOrCreateSession holds graph_session_cache_.mutex_, before CreateSession calls GetOrtEnv.
    cache_miss.Set();
    continue_cache_miss.Wait();
  };
  float input[]{1.0f, -2.0f};
  int32_t output[2]{};
  auto memory_info = OrtMemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  ConcurrentOperation graph{[&] {
    // Use the real WebGPU cache/creation path with CPU execution to avoid a WebGPU runtime dependency.
    Generators::ExecuteCastOp(input, output, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                              2, Generators::DeviceType::WEBGPU, "", memory_info.get());
  }};
  cache_miss.Wait();
  ConcurrentOperation release{[] { Oga::ReleaseDeviceResources("CUDA"); }};
  WaitForReleasePin();
  continue_cache_miss.Set();
  EXPECT_NO_THROW(graph.Get());
  EXPECT_NO_THROW(release.Get());

  EXPECT_EQ(output[0], 1);
  EXPECT_EQ(output[1], -2);
  ASSERT_EQ(globals_->graph_session_cache_.sessions_.size(), 1u);
  EXPECT_EQ(globals_->graph_session_cache_.sessions_.begin()->second.device_type, Generators::DeviceType::WEBGPU);
  EXPECT_FALSE(globals_->device_allocators_[static_cast<int>(Generators::DeviceType::CUDA)].allocator_);
}

TEST_F(DeviceResourceReleaseConcurrencyTests, CudaReleaseWithOtherDeviceInitialization) {
  SeedCudaResources();
  TestEvent initializing, continue_initialization;
  ScopedOrtCallbacks callbacks;
  callbacks.on_session_options = [&] {
    // EnsureDeviceOrtInit holds device_interfaces_mutex_, before provider setup/GetOrtEnv.
    initializing.Set();
    continue_initialization.Wait();
  };
  InitTestDevice device;
  Generators::Config config;
  ConcurrentOperation initialize{[&] { Generators::EnsureDeviceOrtInit(device, config); }};
  initializing.Wait();
  ConcurrentOperation release{[] { Oga::ReleaseDeviceResources("CUDA"); }};
  WaitForReleasePin();
  continue_initialization.Set();
  EXPECT_NO_THROW(initialize.Get());
  EXPECT_NO_THROW(release.Get());

  auto& allocator = globals_->device_allocators_[static_cast<int>(device.GetType())];
  ASSERT_NE(device.allocator_, nullptr);
  EXPECT_EQ(device.allocator_, allocator.allocator_.get());
  EXPECT_NE(allocator.session_, nullptr);
  EXPECT_FALSE(globals_->device_allocators_[static_cast<int>(Generators::DeviceType::CUDA)].allocator_);
  EXPECT_TRUE(globals_->graph_session_cache_.sessions_.empty());
}

TEST_F(DeviceResourceReleaseConcurrencyTests, ShutdownWaitsForEntireCudaRelease) {
  SeedCudaResources();
  auto* env = globals_->env_.get();
  auto* backing_session = globals_->device_allocators_[static_cast<int>(Generators::DeviceType::CUDA)].session_.get();
  TestEvent releasing_session, continue_release, shutdown_started;
  std::atomic<int> destruction_stage{0};
  ScopedOrtCallbacks callbacks;
  callbacks.on_release_allocator = [&] { EXPECT_EQ(destruction_stage.exchange(1), 0); };
  callbacks.on_release_session = [&](OrtSession* session) {
    if (session != backing_session)
      return;
    EXPECT_EQ(destruction_stage.exchange(2), 1);  // Allocator must precede its backing session.
    releasing_session.Set();
    continue_release.Wait();
    EXPECT_EQ(&Generators::GetOrtEnv(), env);
    EXPECT_EQ(Generators::GetOrtGlobals().get(), globals_);
  };
  callbacks.on_release_env = [&] { EXPECT_EQ(destruction_stage.exchange(3), 2); };
  ConcurrentOperation release{[] { Oga::ReleaseDeviceResources("CUDA"); }};
  releasing_session.Wait();
  ASSERT_EQ(globals_->active_resource_releases_.load(), 1u);
  ConcurrentOperation shutdown{[&] {
    shutdown_started.Set();
    OgaShutdown();
  }};
  shutdown_started.Wait();
  EXPECT_FALSE(shutdown.WaitFor(std::chrono::milliseconds(100)));
  EXPECT_EQ(destruction_stage.load(), 2);

  // Shutdown must not retain the global mutex while waiting: both unrelated lookups and release's
  // own teardown callback must still be able to obtain the pinned globals/environment.
  ConcurrentOperation lookup{[&] {
    EXPECT_EQ(&Generators::GetOrtEnv(), env);
    EXPECT_EQ(Generators::GetOrtGlobals().get(), globals_);
  }};
  EXPECT_NO_THROW(lookup.Get());
  continue_release.Set();
  EXPECT_NO_THROW(release.Get());
  EXPECT_NO_THROW(shutdown.Get());
  EXPECT_EQ(destruction_stage.load(), 3);

  EXPECT_NE(Generators::GetOrtGlobals()->env_, nullptr);  // Lazy recreation still works after shutdown.
}

TEST_F(DeviceResourceReleaseConcurrencyTests, ReleaseAndRecreationWaitForShutdownTeardown) {
  TestEvent tearing_down, continue_teardown, release_started, lookup_started;
  ScopedOrtCallbacks callbacks;
  callbacks.on_release_env = [&] {
    tearing_down.Set();
    continue_teardown.Wait();
  };
  ConcurrentOperation shutdown{[] { OgaShutdown(); }};
  tearing_down.Wait();
  ConcurrentOperation release{[&] {
    release_started.Set();
    Oga::ReleaseDeviceResources("CUDA");
  }};
  ConcurrentOperation lookup{[&] {
    lookup_started.Set();
    EXPECT_NE(Generators::GetOrtGlobals()->env_, nullptr);
  }};
  release_started.Wait();
  lookup_started.Wait();
  EXPECT_FALSE(release.WaitFor(std::chrono::milliseconds(50)));
  EXPECT_FALSE(lookup.WaitFor(std::chrono::milliseconds(50)));
  continue_teardown.Set();
  EXPECT_NO_THROW(shutdown.Get());
  EXPECT_NO_THROW(release.Get());
  EXPECT_NO_THROW(lookup.Get());
}

#if USE_CUDA
namespace {

// The CUDA add-on owns one compute stream that every session shares, and the first session to name a GPU binds the
// stream to it. OgaShutdown() destroys the stream together with the interface, so a stream bound by an earlier test
// cannot survive into this one. Returns nullptr and the reason when the add-on is not installed.
Generators::DeviceInterface* LoadCudaInterfaceWithoutAStream(std::string& unavailable_reason) {
  OgaShutdown();
  try {
    return Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
  } catch (const std::exception& e) {
    unavailable_reason = e.what();
    return nullptr;
  }
}

Generators::ProviderOptions CudaDeviceOptions(const std::string& device_id) {
  Generators::ProviderOptions options;
  options.name = "cuda";
  options.options.emplace_back("device_id", device_id);
  return options;
}

// Empty when the add-on is installed and each of GPUs 0..gpu_count-1 can be bound, otherwise why the test cannot run.
std::string CudaUnavailableReason(int gpu_count) {
  std::string reason;
  for (int gpu = 0; gpu < gpu_count && reason.empty(); ++gpu) {
    Generators::DeviceInterface* cuda = LoadCudaInterfaceWithoutAStream(reason);
    if (!cuda) break;
    try {
      const auto options = CudaDeviceOptions(std::to_string(gpu));
      cuda->GetDeviceId(&options);
    } catch (const std::exception& e) {
      reason = "GPU " + std::to_string(gpu) + " cannot be used: " + e.what();
    }
  }
  OgaShutdown();
  return reason;
}

// Runs every job on its own thread, releasing them all at the same moment so that they contend for the stream.
void RunTogether(const std::vector<std::function<void()>>& jobs) {
  std::atomic<size_t> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (size_t i = 0; i < jobs.size(); ++i) {
    threads.emplace_back([&, i] {
      ++ready;
      while (!go.load()) std::this_thread::yield();
      jobs[i]();
    });
  }
  while (ready.load() < jobs.size()) std::this_thread::yield();
  go = true;
  for (auto& thread : threads) thread.join();
}

bool Contains(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }

constexpr int kStreamRaceRounds = 10;

// Whether the CUDA add-on is still mapped in the process, without loading it.
bool CudaAddOnIsMapped() {
#if defined(_WIN32)
  return GetModuleHandleW(L"onnxruntime-genai-cuda.dll") != nullptr;
#else
  void* handle = dlopen("libonnxruntime-genai-cuda.so", RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) return false;
  dlclose(handle);
  return true;
#endif
}

}  // namespace

// Concurrent first uses of the add-on all target the same GPU before any of them has a stream: half of the threads
// name GPU 0 the way a session does, the others bind through their first allocation or copy. The stream has to be
// created once, so every thread must be handed the same handle (a second creation overwrote the stream that earlier
// sessions had already been given).
TEST(ReInitTests, ConcurrentCudaBindingsToOneGpuShareOneStream) {
  if (const std::string reason = CudaUnavailableReason(1); !reason.empty()) GTEST_SKIP() << reason;

  constexpr size_t kThreads = 16;
  for (int round = 0; round < kStreamRaceRounds; ++round) {
    SCOPED_TRACE("round " + std::to_string(round));
    std::string unavailable;
    Generators::DeviceInterface* cuda = LoadCudaInterfaceWithoutAStream(unavailable);
    ASSERT_NE(cuda, nullptr) << unavailable;

    std::vector<void*> streams(kThreads, nullptr);
    std::vector<std::string> errors(kThreads);
    std::vector<std::function<void()>> jobs;
    for (size_t t = 0; t < kThreads; ++t) {
      jobs.push_back([&, t] {
        try {
          if (t % 2 == 0) {
            const auto options = CudaDeviceOptions("0");
            cuda->GetDeviceId(&options);
          }
          streams[t] = cuda->GetCudaStream();
        } catch (const std::exception& e) {
          errors[t] = e.what();
        }
      });
    }
    RunTogether(jobs);

    void* const final_stream = cuda->GetCudaStream();
    ASSERT_NE(final_stream, nullptr);
    for (size_t t = 0; t < kThreads; ++t) {
      EXPECT_TRUE(errors[t].empty()) << "thread " << t << ": " << errors[t];
      EXPECT_EQ(streams[t], final_stream) << "thread " << t << " was handed a different stream";
    }
  }
  OgaShutdown();
}

// Every session shares one stream, so the first GPU a session names owns it and a session naming another GPU has to be
// refused. Two sessions that name different GPUs at the same moment must not both be accepted.
TEST(ReInitTests, ConcurrentCudaBindingsToDifferentGpusAdmitOnlyOne) {
  if (const std::string reason = CudaUnavailableReason(2); !reason.empty()) GTEST_SKIP() << reason;

  for (int round = 0; round < kStreamRaceRounds; ++round) {
    SCOPED_TRACE("round " + std::to_string(round));
    std::string unavailable;
    Generators::DeviceInterface* cuda = LoadCudaInterfaceWithoutAStream(unavailable);
    ASSERT_NE(cuda, nullptr) << unavailable;

    std::vector<int> bound(2, -1);
    std::vector<std::string> errors(2);
    std::vector<std::function<void()>> jobs;
    for (int gpu = 0; gpu < 2; ++gpu) {
      jobs.push_back([&, gpu] {
        try {
          const auto options = CudaDeviceOptions(std::to_string(gpu));
          bound[gpu] = cuda->GetDeviceId(&options);
        } catch (const std::exception& e) {
          errors[gpu] = e.what();
        }
      });
    }
    RunTogether(jobs);

    ASSERT_NE(errors[0].empty(), errors[1].empty())
        << "exactly one session must win the stream; GPU 0: '" << errors[0] << "', GPU 1: '" << errors[1] << "'";
    const int winner = errors[0].empty() ? 0 : 1;
    EXPECT_EQ(bound[winner], winner);
    EXPECT_TRUE(Contains(errors[1 - winner], "conflicts")) << errors[1 - winner];
    EXPECT_EQ(cuda->GetDeviceId(nullptr), winner);
    EXPECT_EQ(cuda->GetMemoryInfo()->GetDeviceId(), winner);
  }
  OgaShutdown();
}

TEST(ReInitTests, ACudaSessionOnAnotherGpuIsRefusedWhileTheStreamIsBound) {
  if (const std::string reason = CudaUnavailableReason(2); !reason.empty()) GTEST_SKIP() << reason;

  std::string unavailable;
  Generators::DeviceInterface* cuda = LoadCudaInterfaceWithoutAStream(unavailable);
  ASSERT_NE(cuda, nullptr) << unavailable;

  const auto gpu0 = CudaDeviceOptions("0");
  const auto gpu1 = CudaDeviceOptions("1");
  ASSERT_EQ(cuda->GetDeviceId(&gpu1), 1);
  void* const stream = cuda->GetCudaStream();
  ASSERT_NE(stream, nullptr);

  try {
    cuda->GetDeviceId(&gpu0);
    ADD_FAILURE() << "a session on GPU 0 must be refused while the stream belongs to GPU 1";
  } catch (const std::runtime_error& error) {
    EXPECT_TRUE(Contains(error.what(), "conflicts")) << error.what();
  }

  // The refusal leaves the binding alone: the same stream still serves GPU 1, and a session that names no GPU takes it.
  EXPECT_EQ(cuda->GetDeviceId(&gpu1), 1);
  EXPECT_EQ(cuda->GetDeviceId(nullptr), 1);
  EXPECT_EQ(cuda->GetCudaStream(), stream);
  OgaShutdown();
}

// On the provider-bridge path a session that names no device is placed on GPU 0 by ONNX Runtime, so the stream
// binding resolves it to GPU 0 as well. The stream bound to another GPU must then be refused rather than handed to a
// session that runs on GPU 0.
TEST(ReInitTests, ALegacyCudaSessionWithoutAnIdIsBoundToGpuZero) {
  if (const std::string reason = CudaUnavailableReason(2); !reason.empty()) GTEST_SKIP() << reason;

  Generators::ProviderOptions no_device;
  no_device.name = "cuda";
  const auto resolved = Generators::CUDAExecutionProvider::ResolveStreamBindingOptions(no_device, std::nullopt);

  std::string unavailable;
  Generators::DeviceInterface* cuda = LoadCudaInterfaceWithoutAStream(unavailable);
  ASSERT_NE(cuda, nullptr) << unavailable;
  const auto gpu1 = CudaDeviceOptions("1");
  ASSERT_EQ(cuda->GetDeviceId(&gpu1), 1);

  try {
    cuda->GetDeviceId(&resolved);
    ADD_FAILURE() << "a legacy session without a device_id must be refused while the stream belongs to GPU 1";
  } catch (const std::runtime_error& error) {
    EXPECT_TRUE(Contains(error.what(), "conflicts")) << error.what();
  }

  cuda = LoadCudaInterfaceWithoutAStream(unavailable);
  ASSERT_NE(cuda, nullptr) << unavailable;
  EXPECT_EQ(cuda->GetDeviceId(&resolved), 0);
  OgaShutdown();
}

#if !defined(_WIN32)
class CudaAdjacentPluginTest : public testing::Test {
 protected:
  struct HardwareDevice {
    const OrtHardwareDevice* device;
    uint32_t id;
  };

  static inline const OrtApi* real_api{};
  static inline HardwareDevice hardware_devices[]{{nullptr, 0}, {nullptr, 1}};
  OrtApi test_api{};

  static const OrtHardwareDevice* ORT_API_CALL EpDeviceHardware(const OrtEpDevice* device) {
    const auto* hardware = real_api->EpDevice_Device(device);
    const auto cuda_id = Generators::GetCudaDeviceIdMetadata(device);
    if (cuda_id == "0" || cuda_id == "1") {
      hardware_devices[cuda_id == "1" ? 1 : 0].device = hardware;
    }
    return hardware;
  }

  static uint32_t ORT_API_CALL HardwareId(const OrtHardwareDevice* device) {
    for (const auto& hardware : hardware_devices) {
      if (device == hardware.device) return hardware.id;
    }
    return real_api->HardwareDevice_DeviceId(device);
  }

  static OrtStatus* ORT_API_CALL DisableProviderBridge(OrtCUDAProviderOptionsV2**) noexcept {
    return real_api->CreateStatus(ORT_FAIL, "Provider bridge disabled for adjacent-plugin regression");
  }

  void SetUp() override {
    if (const std::string reason = CudaUnavailableReason(2); !reason.empty()) GTEST_SKIP() << reason;
    (void)Generators::GetOrtEnv();
    real_api = Ort::api;
    test_api = *real_api;
    // Identical physical GPUs share PCI device IDs. Give just the hardware-filter view distinct
    // IDs while retaining the real plugin devices, CUDA metadata, streams, and append calls.
    test_api.EpDevice_Device = EpDeviceHardware;
    test_api.HardwareDevice_DeviceId = HardwareId;
    test_api.CreateCUDAProviderOptions = DisableProviderBridge;
    Ort::api = &test_api;
  }

  void TearDown() override {
    OgaShutdown();
    if (real_api) Ort::api = real_api;
    real_api = nullptr;
    for (auto& hardware : hardware_devices) hardware.device = nullptr;
  }
};

TEST_F(CudaAdjacentPluginTest, HardwareFilterExcludesGpuZeroBeforeStreamCreation) {
  if (const char* explicit_path = std::getenv("ORT_CUDA_PLUGIN_EP_PATH"); explicit_path && *explicit_path) {
    GTEST_SKIP() << "Adjacent-library discovery requires ORT_CUDA_PLUGIN_EP_PATH to be unset";
  }
  Dl_info info{};
  ASSERT_NE(dladdr(reinterpret_cast<const void*>(real_api->RegisterExecutionProviderLibrary), &info), 0);
  ASSERT_NE(info.dli_fname, nullptr);
  const auto candidate = std::filesystem::path(info.dli_fname).parent_path() / "libonnxruntime_providers_cuda.so";
  if (!std::filesystem::exists(candidate)) GTEST_SKIP() << "No CUDA plugin adjacent to the ORT runtime";
  ASSERT_TRUE(Generators::FindRegisteredEpDevices("CUDAExecutionProvider").empty());

  Generators::ProviderOptions options;
  options.name = "cuda";
  options.device_filtering_options.emplace();
  options.device_filtering_options->hardware_device_id = 1;
  auto session_options = OrtSessionOptions::Create();
  Generators::Config config;

  auto* device = Generators::CUDAExecutionProvider::AppendExecutionProvider(*session_options, options, config);

  ASSERT_NE(device, nullptr);
  EXPECT_EQ(device->GetDeviceId(nullptr), 1);
  EXPECT_EQ(device->GetMemoryInfo()->GetDeviceId(), 1);
  EXPECT_NE(device->GetCudaStream(), nullptr);
  const auto filtered = Generators::ApplyDeviceFiltering(
      options, Generators::FindRegisteredEpDevices("CUDAExecutionProvider"));
  ASSERT_EQ(filtered.size(), 1u);
  EXPECT_EQ(Generators::GetCudaDeviceIdMetadata(filtered.front()), "1");
}
#endif

// The CUDA add-on keeps its stream, its allocator pointer and its device interface in file-scope state. The qwen38 plane built
// into the add-on is never unloaded (a library that has launched kernels must not be unloaded and loaded again, see
// Qwen38Plane), so a process that has loaded a qwen38_flash_graph model keeps the add-on mapped across OgaShutdown() and the
// next init enters it again with the interface of the previous init still in it. The add-on has to release that state before it
// builds the new interface: a Debug build otherwise aborts on the asserts that guard it, a Release build leaks the stream of
// the previous init, and an interface that is destroyed after the new one was built releases the stream of the new one.
// A leaked stream cannot be seen from the outside (the streams of a process cannot be counted, and a destroyed one must not be
// queried), so a Release build catches the last case only, as the null stream checked below.
TEST(ReInitTests, TheCudaAddOnIsEnteredAgainWhenTheBuiltInQwen38PlaneKeepsItMapped) {
  // Without a usable GPU the first device operation (the stream) throws outside the setup guards below, so probe it up front.
  if (const std::string reason = CudaUnavailableReason(1); !reason.empty()) GTEST_SKIP() << reason;

  // A round is what loading a qwen38_flash_graph model and unloading it again does to the add-on. Only the first round may
  // find the setup unusable and skip: a later round that cannot enter the add-on again, or finds it unmapped, is the failure
  // this test is about.
  for (int round = 0; round < 3; ++round) {
    Generators::DeviceInterface* device = nullptr;
    try {
      device = Generators::GetDeviceInterface(Generators::DeviceType::CUDA);
      device->InitOrt(*Ort::api, Ort::Allocator::GetWithDefaultOptions());
    } catch (const std::exception& e) {
      OgaShutdown();
      if (round == 0) GTEST_SKIP() << "the CUDA add-on is not usable here: " << e.what();
      FAIL() << "the CUDA add-on could not be entered again in round " << round << ": " << e.what();
    }
    ASSERT_NE(device->GetCudaStream(), nullptr) << "round " << round;
    EXPECT_NO_THROW(device->Synchronize()) << "round " << round;

    try {
      Generators::Qwen38Plane plane{std::filesystem::path{}};  // what a qwen38_flash_graph model loads
    } catch (const std::exception& e) {
      OgaShutdown();
      if (round == 0) GTEST_SKIP() << "the CUDA add-on has no qwen38 plane: " << e.what();
      FAIL() << "the qwen38 plane could not be loaded again in round " << round << ": " << e.what();
    }

    OgaShutdown();
    ASSERT_TRUE(CudaAddOnIsMapped())
        << "OgaShutdown() unloaded the CUDA add-on in round " << round << " in spite of the built-in plane";
  }
}
#endif  // USE_CUDA

#if USE_DML
// Deterministic white-box guard for the DML branch of OrtGlobals::GetDeviceInterface. DML's
// interface (g_dml_device) is destroyed per-Model in Model::~Model via CloseDmlInterface(), so
// OrtGlobals must NOT cache the DML interface pointer -- a cached pointer would dangle after the
// first DML model is freed and be handed to the next DML model (use-after-free). This drives the
// same lifecycle directly (create -> GetDeviceInterface -> CloseDmlInterface -> recreate) with no
// ORT session/model, so it needs only a D3D12 device (not a DML-partitionable model). This mirrors
// the DML branch of OrtGlobals::GetDeviceInterface in generators.cpp.
TEST(ReInitTests, DmlInterfaceNotCachedAcrossReload) {
  // First "model load": create the DML interface. Skips if no DML/D3D12 device is available.
  try {
    Generators::InitDmlInterface(nullptr, nullptr);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "DML device not available: " << e.what();
  }

  // Populate the OrtGlobals DeviceInterface lookup for DML (this is where the buggy version would
  // memoize the pointer).
  Generators::DeviceInterface* dml_first = Generators::GetDeviceInterface(Generators::DeviceType::DML);
  ASSERT_NE(dml_first, nullptr);
  ASSERT_EQ(dml_first, Generators::GetDmlInterface());

  // First "model free": Model::~Model calls CloseDmlInterface(), destroying the interface.
  Generators::CloseDmlInterface();
  ASSERT_EQ(Generators::GetDmlInterface(), nullptr);

  // Second "model load": a fresh DML interface is created.
  Generators::InitDmlInterface(nullptr, nullptr);
  Generators::DeviceInterface* live = Generators::GetDmlInterface();
  ASSERT_NE(live, nullptr);

  // The guard: GetDeviceInterface(DML) must return the LIVE interface, not the freed first one.
  // Without the no-cache fix it returns the stale memoized pointer (which differs from the live
  // GetDmlInterface()), so this comparison fails.
  ASSERT_EQ(Generators::GetDeviceInterface(Generators::DeviceType::DML), live);

  Generators::CloseDmlInterface();
  OgaShutdown();  // reset process-global state for isolation
}
#endif  // USE_DML

int main(int argc, char** argv) {
  // Suppress telemetry before any Oga call, including across each re-initialization cycle.
  Generators::test::SuppressTelemetryForTests();

  ::testing::InitGoogleTest(&argc, argv);

  // --ep_dir <dir>: also register plugin EP libraries found under <dir> (recursive).
  std::filesystem::path ep_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ep_dir" && i + 1 < argc)
      ep_dir = argv[++i];
  }

  // Register any WinML-installed EP packages by default (Windows) so re-init is exercised with real
  // device interfaces where available, plus any --ep_dir. Discovery is filesystem/PowerShell only
  // (no genai env); the test registers these on genai's env at the start of each cycle.
#if defined(_WIN32)
  g_ep_registrar.DiscoverWinML();
#endif
  g_ep_registrar.DiscoverFromDirectory(ep_dir);

  const int result = RUN_ALL_TESTS();
  OgaShutdown();
  return result;
}

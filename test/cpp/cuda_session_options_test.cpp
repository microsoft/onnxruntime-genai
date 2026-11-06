// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include "ep/cuda/session_options.h"
#include "models/session_options.h"

namespace Generators::test {
namespace {

struct StreamDevice final : DeviceInterface {
  DeviceType GetType() const override { return DeviceType::CUDA; }
  void InitOrt(const OrtApi&, Ort::Allocator&) override { throw std::logic_error("Unexpected initialization"); }
  Ort::Allocator& GetAllocator() override { throw std::logic_error("Unexpected allocation"); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override { throw std::logic_error("Unexpected memory query"); }
  std::string GetExecutionProviderName() const override { return "cuda"; }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t) override { throw std::logic_error("Unexpected allocation"); }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void*, size_t) override { throw std::logic_error("Unexpected memory access"); }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams&) override { throw std::logic_error("Unexpected search"); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams&) override { throw std::logic_error("Unexpected search"); }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State&) override { throw std::logic_error("Unexpected cache"); }
  DeviceInterface& GetCpuFallbackDevice() override { return *this; }
  void Synchronize() override { throw std::logic_error("Unexpected synchronization"); }
  void* GetCudaStream() override { return stream; }

  int stream_storage{};
  void* stream{&stream_storage};
};

struct Calls {
  StreamDevice device;
  ProviderOptions plugin_options;
  std::vector<NamedString> session_config, bridge_options;
  void* bridge_stream{};
  int plugin_calls{}, bridge_calls{}, provider_releases{}, arena_releases{};
  bool plugin_registered{true}, plugin_throws{}, arena_alive{}, bridge_saw_arena{};
  int session_handle{}, provider_handle{}, arena_handle{};
};

Calls* calls{};

class CudaSessionOptionsTest : public testing::Test {
 protected:
  void SetUp() override {
    calls = &calls_;
    previous_api_ = Ort::api;
    api_.CreateSessionOptions = [](OrtSessionOptions** output) noexcept -> OrtStatus* {
      *output = reinterpret_cast<OrtSessionOptions*>(&calls->session_handle);
      return nullptr;
    };
    api_.ReleaseSessionOptions = [](OrtSessionOptions*) noexcept {};
    api_.AddSessionConfigEntry = [](OrtSessionOptions*, const char* key, const char* value) noexcept -> OrtStatus* {
      calls->session_config.emplace_back(key, value);
      return nullptr;
    };
    api_.CreateCUDAProviderOptions = [](OrtCUDAProviderOptionsV2** output) noexcept -> OrtStatus* {
      *output = reinterpret_cast<OrtCUDAProviderOptionsV2*>(&calls->provider_handle);
      return nullptr;
    };
    api_.ReleaseCUDAProviderOptions = [](OrtCUDAProviderOptionsV2*) noexcept { ++calls->provider_releases; };
    api_.UpdateCUDAProviderOptions = [](OrtCUDAProviderOptionsV2*, const char* const* keys,
                                        const char* const* values, size_t count) noexcept -> OrtStatus* {
      for (size_t i = 0; i < count; ++i)
        calls->bridge_options.emplace_back(keys[i], values[i]);
      return nullptr;
    };
    api_.UpdateCUDAProviderOptionsWithValue = [](OrtCUDAProviderOptionsV2*, const char* key,
                                                 void* value) noexcept -> OrtStatus* {
      if (std::string_view{key} == "user_compute_stream")
        calls->bridge_stream = value;
      else {
        EXPECT_STREQ(key, "default_memory_arena_cfg");
        EXPECT_TRUE(calls->arena_alive);
        EXPECT_EQ(value, &calls->arena_handle);
      }
      return nullptr;
    };
    api_.CreateArenaCfgV2 = [](const char* const* keys, const size_t* values, size_t count,
                               OrtArenaCfg** output) noexcept -> OrtStatus* {
      EXPECT_EQ(count, 5u);
      EXPECT_STREQ(keys[0], "max_mem");
      EXPECT_EQ(values[0], 4096u);
      calls->arena_alive = true;
      *output = reinterpret_cast<OrtArenaCfg*>(&calls->arena_handle);
      return nullptr;
    };
    api_.ReleaseArenaCfg = [](OrtArenaCfg*) noexcept {
      calls->arena_alive = false;
      ++calls->arena_releases;
    };
    api_.SessionOptionsAppendExecutionProvider_CUDA_V2 =
        [](OrtSessionOptions*, const OrtCUDAProviderOptionsV2*) noexcept -> OrtStatus* {
      ++calls->bridge_calls;
      calls->bridge_saw_arena = calls->arena_alive;
      return nullptr;
    };
    Ort::api = &api_;
    session_options_ = OrtSessionOptions::Create();
  }

  void TearDown() override {
    session_options_.reset();
    Ort::api = previous_api_;
    calls = nullptr;
  }

  DeviceInterface* Append(const ProviderOptions& options) {
    return CUDAExecutionProvider::AppendExecutionProvider(*session_options_, options, config_);
  }

  std::string StreamValue() const {
    return std::to_string(reinterpret_cast<uintptr_t>(calls_.device.stream));
  }

  Calls calls_;
  Config config_;
  OrtApi api_{};
  const OrtApi* previous_api_{};
  std::unique_ptr<OrtSessionOptions> session_options_;
};

TEST_F(CudaSessionOptionsTest, PluginReplacesStreamOptionsWithoutChangingCallerOptions) {
  ProviderOptions options{"cuda", {{"user_compute_stream", "17"}, {"has_user_compute_stream", "0"}, {"enable_cuda_graph", "1"}, {"user_compute_stream", "23"}, {"has_user_compute_stream", "0"}}, {}};
  options.device_filtering_options.emplace().hardware_device_id = 3;
  const auto original_options = options.options;
  EXPECT_EQ(Append(options), &calls_.device);
  EXPECT_EQ(calls_.plugin_calls, 1);
  EXPECT_EQ(calls_.bridge_calls, 0);
  EXPECT_EQ(calls_.plugin_options.name, options.name);
  ASSERT_TRUE(calls_.plugin_options.device_filtering_options);
  EXPECT_EQ(calls_.plugin_options.device_filtering_options->hardware_device_id, 3u);
  EXPECT_EQ(calls_.plugin_options.options, (std::vector<NamedString>{
                                               {"enable_cuda_graph", "1"}, {"user_compute_stream", StreamValue()}, {"has_user_compute_stream", "1"}}));
  EXPECT_EQ(options.options, original_options);
  EXPECT_EQ(calls_.session_config, (std::vector<NamedString>{{"user_compute_stream", StreamValue()}}));
}

TEST_F(CudaSessionOptionsTest, MissingPluginFallsBackToTypedStreamOptions) {
  calls_.plugin_registered = false;
  ProviderOptions options{"cuda", {{"enable_cuda_graph", "1"}, {"max_mem", "4096"}}, {}};
  const auto original_options = options.options;
  EXPECT_EQ(Append(options), &calls_.device);
  EXPECT_EQ(calls_.plugin_calls, 1);
  EXPECT_EQ(calls_.bridge_calls, 1);
  EXPECT_EQ(calls_.bridge_stream, calls_.device.stream);
  EXPECT_EQ(calls_.bridge_options, (std::vector<NamedString>{{"enable_cuda_graph", "1"}}));
  EXPECT_EQ(calls_.provider_releases, 1);
  EXPECT_TRUE(calls_.bridge_saw_arena);
  EXPECT_EQ(calls_.arena_releases, 1);
  EXPECT_FALSE(calls_.arena_alive);
  EXPECT_EQ(options.options, original_options);
}

TEST_F(CudaSessionOptionsTest, PluginErrorsAreNotHiddenByBridgeFallback) {
  calls_.plugin_throws = true;
  EXPECT_THROW(Append(ProviderOptions{"cuda", {}, {}}), std::runtime_error);
  EXPECT_EQ(calls_.plugin_calls, 1);
  EXPECT_EQ(calls_.bridge_calls, 0);
}

TEST_F(CudaSessionOptionsTest, InvalidArenaConfigurationDoesNotAppendAProvider) {
  calls_.plugin_registered = false;
  EXPECT_THROW(Append(ProviderOptions{"cuda", {{"max_mem", "-2"}}, {}}), std::out_of_range);
  EXPECT_EQ(calls_.bridge_calls, 0);
  EXPECT_EQ(calls_.provider_releases, 1);
}

TEST_F(CudaSessionOptionsTest, StreamConfigSupportsCustomKeysAndAbsentDevices) {
  CUDAExecutionProvider::AddCudaStreamConfig(*session_options_, &calls_.device, "custom_stream");
  CUDAExecutionProvider::AddCudaStreamConfig(*session_options_, nullptr);
  EXPECT_EQ(calls_.session_config, (std::vector<NamedString>{{"custom_stream", StreamValue()}}));
}

}  // namespace
}  // namespace Generators::test

namespace Generators {

DeviceInterface* GetDeviceInterface(DeviceType type) {
  EXPECT_EQ(type, DeviceType::CUDA);
  return &test::calls->device;
}

std::unique_ptr<PositionInputs> CreateStandardPositionInputs(State&, DeviceSpan<int32_t>,
                                                             const std::string&) {
  throw std::logic_error("Unexpected position inputs");
}

// Device enumeration is outside this CPU-only stream-forwarding target.
std::vector<const OrtEpDevice*> FindRegisteredEpDevices(const std::string& name) {
  EXPECT_EQ(name, "CUDAExecutionProvider");
  return {};
}

std::vector<const OrtEpDevice*> ApplyDeviceFiltering(const Config::ProviderOptions&,
                                                     const std::vector<const OrtEpDevice*>&) {
  throw std::logic_error("Unexpected device filtering");
}

bool HasDeviceFiltering(const Config::ProviderOptions&) {
  throw std::logic_error("Unexpected device filtering");
}

std::string GetCudaDeviceIdMetadata(const OrtEpDevice*) {
  throw std::logic_error("Unexpected CUDA device metadata query");
}

size_t SelectCudaDeviceIndex(const std::vector<std::string>&,
                             const std::optional<std::string>&, bool) {
  throw std::logic_error("Unexpected CUDA device selection");
}

bool AppendExecutionProviderV2(OrtSessionOptions&, const Config::ProviderOptions& options,
                               DeviceType type, const std::string& name) {
  EXPECT_EQ(type, DeviceType::CUDA);
  EXPECT_EQ(name, "CUDAExecutionProvider");
  ++test::calls->plugin_calls;
  test::calls->plugin_options = options;
  if (test::calls->plugin_throws)
    throw std::runtime_error("Injected plugin failure");
  return test::calls->plugin_registered;
}

}  // namespace Generators

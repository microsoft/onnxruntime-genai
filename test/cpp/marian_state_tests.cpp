// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "models/io/kv_cache.h"
#include "models/model.h"
#include "search.h"

namespace Generators {
namespace {

struct CopyStats {
  size_t largest_allocation{};
  size_t partial_copies{};
};

// Separate host/device storage reproduces stale CPU mirrors without requiring a GPU.
struct MirroredBuffer final : DeviceBuffer {
  MirroredBuffer(std::shared_ptr<DeviceBuffer> inner, CopyStats& stats)
      : inner_{std::move(inner)}, stats_{stats} {
    p_device_ = inner_->p_device_;
    size_in_bytes_ = inner_->size_in_bytes_;
  }

  const char* GetType() const override { return "test_mirrored"; }
  void AllocateCpu() override {
    if (cpu_.empty()) {
      cpu_.resize(size_in_bytes_);
      p_cpu_ = cpu_.data();
    }
  }
  void CopyDeviceToCpu() override {
    AllocateCpu();
    std::memcpy(p_cpu_, p_device_, size_in_bytes_);
  }
  void CopyCpuToDevice() override {
    std::memcpy(p_device_, p_cpu_, size_in_bytes_);
  }
  void CopyFrom(size_t dest, DeviceBuffer& source, size_t begin, size_t bytes) override {
    if (dest == 0 && begin == 0 && bytes == size_in_bytes_ && bytes == source.size_in_bytes_) {
      std::memcpy(p_device_, source.p_device_, bytes);
    } else {
      ++stats_.partial_copies;
      CopyThroughCpu(*this, dest, source, begin, bytes);
    }
  }
  void Zero() override { std::memset(p_device_, 0, size_in_bytes_); }

 private:
  std::shared_ptr<DeviceBuffer> inner_;
  CopyStats& stats_;
  std::vector<uint8_t> cpu_;
};

class MirroredDevice final : public DeviceInterface {
 public:
  DeviceType GetType() const override { return cpu_.GetType(); }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override { cpu_.InitOrt(api, allocator); }
  Ort::Allocator& GetAllocator() override { return cpu_.GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override { return cpu_.GetMemoryInfo(); }
  std::string GetExecutionProviderName() const override { return cpu_.GetExecutionProviderName(); }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    stats.largest_allocation = std::max(stats.largest_allocation, size);
    return std::make_shared<MirroredBuffer>(cpu_.AllocateBase(size), stats);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    return std::make_shared<MirroredBuffer>(cpu_.WrapMemoryBase(memory, size), stats);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return cpu_.CreateGreedy(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return cpu_.CreateBeam(params); }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override { return cpu_.CreateKeyValueCache(state); }
  void Synchronize() override { cpu_.Synchronize(); }

  CopyStats stats;

 private:
  DeviceInterface& cpu_{*GetDeviceInterface(DeviceType::CPU)};
};

auto LoadRecurrentModel() {
  return CreateModel(GetOrtEnv(), MODEL_PATH "marian-cache/recurrent");
}

DeviceSpan<int32_t> Tokens(DeviceInterface& device, std::span<const int32_t> values) {
  auto tokens = device.Allocate<int32_t>(values.size());
  tokens.CopyFromCpu(values);
  return tokens;
}

std::vector<float> CopyLogits(DeviceSpan<float> logits) {
  const auto values = logits.CopyDeviceToCpu();
  return {values.begin(), values.end()};
}

TEST(MarianStateTests, EncoderRewindReadsDeviceTokenRatherThanStaleCpuMirror) {
  auto model = LoadRecurrentModel();
  auto params = CreateGeneratorParams(*model);
  auto lengths = params->p_device->Allocate<int32_t>(1);
  auto state = model->CreateState(lengths, *params);
  const std::array<int32_t, 4> source{2, 3, 4, 5};
  auto input = Tokens(*params->p_device, source);
  state->Run(4, input);
  state->generation_started_ = true;
  const std::array<int32_t, 1> sampled{42};
  auto decoder = Tokens(*params->p_device, sampled);
  state->Run(5, decoder);
  state->RewindTo(2);

  MirroredDevice device;
  auto replay = Tokens(device, sampled);
  replay.Span()[0] = source[2];
  ASSERT_EQ(replay.CpuSpan()[0], 42);
  const auto actual = CopyLogits(state->Run(3, replay));
  auto reference = model->CreateState(lengths, *params);
  auto prefix = Tokens(*params->p_device, std::span{source}.first(3));
  EXPECT_EQ(actual, CopyLogits(reference->Run(3, prefix)));
}

TEST(MarianStateTests, CheckpointsCopyOnlyWholeIndependentStateBuffers) {
  auto model = LoadRecurrentModel();
  MirroredDevice device;
  model->p_device_inputs_ = &device;
  auto params = CreateGeneratorParams(*model);
  params->search.max_length = 64;
  auto lengths = params->p_device->Allocate<int32_t>(1);
  auto state = model->CreateState(lengths, *params);
  const std::array<int32_t, 4> source{2, 3, 4, 5};
  auto input = Tokens(*params->p_device, source);
  state->Run(4, input);
  state->generation_started_ = true;
  const std::array<int32_t, 1> token{42};
  auto next = Tokens(*params->p_device, token);
  for (int length = 5; length <= 8; ++length)
    state->Run(length, next);
  state->RewindTo(6);
  state->Run(7, next);
  const auto state_bytes = state->GetOutput(model->config_->model.decoder.outputs.rnn_states.c_str())
                               ->GetTensorTypeAndShapeInfo()
                               ->GetElementCount() *
                           sizeof(float);
  EXPECT_LE(device.stats.largest_allocation, state_bytes);
  EXPECT_EQ(device.stats.partial_copies, 0U);
}

TEST(MarianStateTests, BeamSearchDoesNotAllocateRewindCheckpoints) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "marian-cache/uncached");
  MirroredDevice device;
  model->p_device_inputs_ = &device;
  auto params = CreateGeneratorParams(*model);
  params->search.num_beams = 2;
  params->search.max_length = 64;
  auto lengths = params->p_device->Allocate<int32_t>(2);
  auto state = model->CreateState(lengths, *params);
  const std::array<int32_t, 4> source{2, 3, 4, 5};
  auto input = Tokens(*params->p_device, source);
  state->Run(4, input);
  state->generation_started_ = true;
  const std::array<int32_t, 2> tokens{42, 43};
  auto next = Tokens(*params->p_device, tokens);
  state->Run(5, next);
  EXPECT_EQ(device.stats.largest_allocation, (source.size() + 1) * sizeof(int32_t));
}

TEST(MarianStateTests, BeamPrefillPreservesEachRowAndRejectsOverflow) {
  Config config;
  config.model.vocab_size = 16;
  config.model.context_length = 6;
  config.model.eos_token_id = {0};
  config.search.batch_size = 2;
  config.search.num_beams = 2;
  config.search.max_length = 6;
  auto params = std::make_shared<GeneratorParams>(config);
  BeamSearch_Cpu search{*params};
  const std::array<int32_t, 4> first{2, 3, 4, 5};
  const std::array<int32_t, 2> second{6, 7};
  const std::array<int32_t, 4> third{8, 9, 10, 11};
  auto input = Tokens(*params->p_device, first);
  search.AppendTokens(input);
  input = Tokens(*params->p_device, second);
  search.AppendTokens(input);
  input = Tokens(*params->p_device, third);
  search.AppendTokens(input);
  const std::array<std::array<int32_t, 5>, 2> expected{{{2, 3, 6, 8, 9}, {4, 5, 7, 10, 11}}};
  const auto check_history = [&] {
    ASSERT_EQ(search.GetSequenceLength(), 5);
    for (size_t beam = 0; beam < 4; ++beam) {
      const auto sequence = search.sequences_.GetSequence(beam).CopyDeviceToCpu();
      EXPECT_TRUE(std::equal(sequence.begin(), sequence.end(), expected[beam / 2].begin()));
    }
  };
  check_history();
  EXPECT_THROW(search.AppendTokens(input), std::runtime_error);
  check_history();
}

}  // namespace
}  // namespace Generators

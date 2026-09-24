// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "models/io/kv_cache.h"
#include "models/io/position_inputs.h"
#include "models/model.h"
#include "search.h"

#if USE_CUDA
TEST(LogitsTests, Fp16PrefillDoesNotAllocateFullSequenceFp32Logits) {
  struct RecordingAllocator : OrtAllocator {
    explicit RecordingAllocator(OrtAllocator& delegate) : delegate_{delegate} {
      static_cast<OrtAllocator&>(*this) = delegate_;
      Alloc = [](OrtAllocator* self, size_t size) -> void* {
        auto& recorder = *static_cast<RecordingAllocator*>(self);
        recorder.sizes.push_back(size);
        return recorder.delegate_.Alloc(&recorder.delegate_, size);
      };
      Free = [](OrtAllocator* self, void* memory) {
        auto& recorder = *static_cast<RecordingAllocator*>(self);
        recorder.delegate_.Free(&recorder.delegate_, memory);
      };
      Info = [](const OrtAllocator* self) -> const OrtMemoryInfo* {
        auto& recorder = *static_cast<const RecordingAllocator*>(self);
        return recorder.delegate_.Info(&recorder.delegate_);
      };
    }

    OrtAllocator& delegate_;
    std::vector<size_t> sizes;
  };

  struct RecordingLogitsDevice : Generators::DeviceInterface {
    explicit RecordingLogitsDevice(Generators::DeviceInterface& delegate)
        : delegate_{delegate}, allocator_{delegate.GetAllocator()} {}

    Generators::DeviceType GetType() const override { return delegate_.GetType(); }
    void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override { delegate_.InitOrt(api, allocator); }
    Ort::Allocator& GetAllocator() override { return reinterpret_cast<Ort::Allocator&>(allocator_); }
    std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override { return delegate_.GetMemoryInfo(); }
    std::string GetExecutionProviderName() const override { return delegate_.GetExecutionProviderName(); }
    std::shared_ptr<Generators::DeviceBuffer> AllocateBase(size_t size) override { return delegate_.AllocateBase(size); }
    std::shared_ptr<Generators::DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
      return delegate_.WrapMemoryBase(memory, size);
    }
    std::unique_ptr<Generators::Search> CreateGreedy(const Generators::GeneratorParams& params) override {
      return delegate_.CreateGreedy(params);
    }
    std::unique_ptr<Generators::Search> CreateBeam(const Generators::GeneratorParams& params) override {
      return delegate_.CreateBeam(params);
    }
    std::unique_ptr<Generators::KeyValueCache> CreateKeyValueCache(Generators::State& state) override {
      return delegate_.CreateKeyValueCache(state);
    }
    Generators::DeviceInterface& GetCpuFallbackDevice() override { return delegate_.GetCpuFallbackDevice(); }
    std::unique_ptr<Generators::PositionInputs> CreatePositionInputs(
        Generators::State& state, Generators::DeviceSpan<int32_t> sequence_lengths,
        const std::string& attention_mask_name) override {
      return delegate_.CreatePositionInputs(state, sequence_lengths, attention_mask_name);
    }
    void Synchronize() override { delegate_.Synchronize(); }
    bool Cast(void* input, void* output, ONNXTensorElementDataType input_type,
              ONNXTensorElementDataType output_type, size_t count) override {
      return delegate_.Cast(input, output, input_type, output_type, count);
    }

    Generators::DeviceInterface& delegate_;
    RecordingAllocator allocator_;
  };

  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp16-cuda");
  ASSERT_FALSE(model->IsPruned());
  const auto& logits_name = model->config_->model.decoder.outputs.logits;
  ASSERT_EQ(model->session_info_.GetOutputDataType(logits_name), Ort::TypeToTensorType<Ort::Float16_t>);

  RecordingLogitsDevice recording_device{*model->p_device_logits_};
  struct RestoreLogitsDevice {
    Generators::Model& model;
    Generators::DeviceInterface* original;
    ~RestoreLogitsDevice() { model.p_device_logits_ = original; }
  } restore{*model, model->p_device_logits_};
  model->p_device_logits_ = &recording_device;
  auto params = Generators::CreateGeneratorParams(*model);
  params->search.batch_size = 2;
  params->search.max_length = 10;
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(std::vector<int32_t>{0, 0, 0, 52, 0, 0, 195, 731});
  EXPECT_EQ(generator->GetLogits().size(), 2u * 1000);

  const size_t last_token_fp32_bytes = 2 * 1000 * sizeof(float);
  const size_t full_sequence_fp32_bytes = 2 * 4 * 1000 * sizeof(float);
  EXPECT_GE(std::count(recording_device.allocator_.sizes.begin(), recording_device.allocator_.sizes.end(),
                       last_token_fp32_bytes),
            1);
  EXPECT_EQ(std::count(recording_device.allocator_.sizes.begin(), recording_device.allocator_.sizes.end(),
                       full_sequence_fp32_bytes),
            0);
}
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Each state allocates against the device of the session it drives, never the decoder's
// (see SessionCanAccess).

#include <array>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "models/model.h"
#include "models/io/embeddings.h"
#include "models/io/kv_cache.h"
#include "models/vision_stage_io.h"
#include "search.h"

namespace {

struct PlacementTestModel : Generators::Model {
  PlacementTestModel() : Model{std::make_unique<Generators::Config>()} {}

  std::unique_ptr<Generators::State> CreateState(
      Generators::DeviceSpan<int32_t>, const Generators::GeneratorParams&) const override {
    return nullptr;
  }
};

struct PlacementTestState : Generators::State {
  PlacementTestState(const Generators::GeneratorParams& params, const Generators::Model& model,
                     Generators::DeviceInterface* session_device)
      : State{params, model, session_device} {}

  Generators::DeviceSpan<float> Run(int, Generators::DeviceSpan<int32_t>&,
                                    Generators::DeviceSpan<int32_t>) override {
    return {};
  }
};

struct TestAcceleratorBuffer : Generators::DeviceBuffer {
  TestAcceleratorBuffer(void* data, size_t size) : host_copy_(size) {
    p_device_ = static_cast<uint8_t*>(data);
    size_in_bytes_ = size;
  }

  const char* GetType() const override { return "test_accelerator"; }
  void AllocateCpu() override { p_cpu_ = host_copy_.data(); }
  void CopyDeviceToCpu() override {
    AllocateCpu();
    std::memcpy(p_cpu_, p_device_, size_in_bytes_);
  }
  void CopyCpuToDevice() override { std::memcpy(p_device_, p_cpu_, size_in_bytes_); }
  void CopyFrom(size_t dest, Generators::DeviceBuffer& source, size_t begin, size_t bytes) override {
    Generators::CopyThroughCpu(*this, dest, source, begin, bytes);
  }
  void Zero() override { std::memset(p_device_, 0, size_in_bytes_); }

 private:
  std::vector<uint8_t> host_copy_;
};

struct TestAccelerator : Generators::DeviceInterface {
  Generators::DeviceType GetType() const override { return Generators::DeviceType::DML; }
  void InitOrt(const OrtApi&, Ort::Allocator&) override {}
  Ort::Allocator& GetAllocator() override { return Generators::GetDeviceInterface(Generators::DeviceType::CPU)->GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return OrtMemoryInfo::Create("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  }
  std::string GetExecutionProviderName() const override { return "DML"; }
  std::shared_ptr<Generators::DeviceBuffer> AllocateBase(size_t) override { return {}; }
  std::shared_ptr<Generators::DeviceBuffer> WrapMemoryBase(void* data, size_t size) override {
    return std::make_shared<TestAcceleratorBuffer>(data, size);
  }
  std::unique_ptr<Generators::Search> CreateGreedy(const Generators::GeneratorParams&) override { return {}; }
  std::unique_ptr<Generators::Search> CreateBeam(const Generators::GeneratorParams&) override { return {}; }
  std::unique_ptr<Generators::KeyValueCache> CreateKeyValueCache(Generators::State&) override { return {}; }
  void Synchronize() override {}
};

TEST(MultiModalDevicePlacementTests, VisionPipelineProvidersFollowStageOptions) {
  Generators::Config config{
      fs::path{std::string{MODEL_PATH "gemma4"}},
      R"({"model":{"vision":{"pipeline":[{"encoder":{"filename":"encoder.onnx","session_options":{"provider_options":[{"cuda":{}}]}}},{"projector":{"filename":"projector.onnx","session_options":{"provider_options":[{"DML":{}}]},"run_on_cpu":true}}]}}})"};
  ASSERT_EQ(config.model.vision.pipeline.size(), 2u);
  EXPECT_EQ(config.model.vision.pipeline[0].session_options->providers, (std::vector<std::string>{"cuda"}));
  EXPECT_EQ(config.model.vision.pipeline[1].session_options->providers, (std::vector<std::string>{"DML"}));
  EXPECT_TRUE(config.model.vision.pipeline[1].run_on_cpu);
}

TEST(MultiModalDevicePlacementTests, SessionCanAccessOnlyHostOrItsOwnMemory) {
  auto& cpu = *Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  auto& gpu = *Generators::GetDeviceInterface(Generators::DeviceType::WEBGPU);

  EXPECT_TRUE(Generators::SessionCanAccess(cpu, cpu));
  EXPECT_TRUE(Generators::SessionCanAccess(gpu, gpu));
  // Host memory is always reachable: ORT copies it to and from the session's own device.
  EXPECT_TRUE(Generators::SessionCanAccess(gpu, cpu));
  // A CPU-EP session has no data transfer for device memory, which is the case that corrupts
  // the heap when the buffer is bound anyway.
  EXPECT_FALSE(Generators::SessionCanAccess(cpu, gpu));
}

TEST(MultiModalDevicePlacementTests, StagesAcceleratorEncoderOutputForCpuProjector) {
  TestAccelerator encoder;
  auto& projector = *Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  auto memory_info = encoder.GetMemoryInfo();
  std::array<float, 4> pixels{1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<int64_t, 2> shape{2, 2};
  auto encoded = OrtValue::CreateTensor<float>(*memory_info, pixels, shape);

  auto staged = Generators::StageVisionEncoderOutput(*encoded, encoder, projector);
  ASSERT_NE(staged, nullptr);
  EXPECT_EQ(staged->GetTensorMemoryInfo().GetDeviceType(), OrtMemoryInfoDeviceType_CPU);
  EXPECT_EQ(std::vector<float>(staged->GetTensorData<float>(), staged->GetTensorData<float>() + pixels.size()),
            (std::vector<float>{pixels.begin(), pixels.end()}));
}

TEST(MultiModalDevicePlacementTests, ReusesCpuPlacedEncoderOutput) {
  TestAccelerator encoder;
  auto& cpu = *Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  auto encoded = OrtValue::CreateTensor<float>(cpu.GetAllocator(), std::array<int64_t, 1>{2});
  EXPECT_EQ(Generators::StageVisionEncoderOutput(*encoded, encoder, cpu), nullptr);
}

TEST(MultiModalDevicePlacementTests, RejectsDeviceOutputWithoutCopyInterface) {
  TestAccelerator encoder;
  auto& cpu = *Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  auto memory_info = encoder.GetMemoryInfo();
  std::array<float, 1> value{1.0f};
  auto encoded = OrtValue::CreateTensor<float>(*memory_info, value, std::array<int64_t, 1>{1});
  EXPECT_THROW(Generators::StageVisionEncoderOutput(*encoded, cpu, cpu), std::runtime_error);
}

TEST(MultiModalDevicePlacementTests, StateDefaultsToTheModelDevices) {
  auto model = std::make_shared<PlacementTestModel>();
  auto params = std::make_shared<Generators::GeneratorParams>(*model);
  // A CPU model has p_device_inputs_ == p_device_, which would hide a state that picks the wrong one.
  model->p_device_inputs_ = Generators::GetDeviceInterface(Generators::DeviceType::WEBGPU);

  PlacementTestState state{*params, *model, nullptr};

  EXPECT_EQ(state.p_session_device_, model->p_device_);
  EXPECT_EQ(state.p_session_device_inputs_, model->p_device_inputs_);
}

TEST(MultiModalDevicePlacementTests, StateOnAnotherDeviceKeepsItsOwnInputsDevice) {
  auto model = std::make_shared<PlacementTestModel>();
  auto params = std::make_shared<Generators::GeneratorParams>(*model);
  auto* elsewhere = Generators::GetDeviceInterface(Generators::DeviceType::WEBGPU);
  ASSERT_NE(elsewhere, model->p_device_);

  PlacementTestState state{*params, *model, elsewhere};

  EXPECT_EQ(state.p_session_device_, elsewhere);
  // p_device_inputs_ was picked for the *model's* device, so a session that runs somewhere else
  // must not allocate its inputs from it.
  EXPECT_EQ(state.p_session_device_inputs_, elsewhere);
}

// Uncaptured prefills and chunk views must preserve the persistent graph-capture decode buffer.
TEST(MultiModalDevicePlacementTests, EmbeddingsPreserveCapturedDecodeBufferAcrossPrefills) {
  Ort::InitApi();
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto params = Generators::CreateGeneratorParams(*model);
  params->use_graph_capture = true;
  PlacementTestState state{*params, *model, nullptr};
  const auto& name = model->config_->model.decoder.inputs.input_ids;

  for (int64_t hidden_size : {32, 128}) {
    Generators::Embeddings embeddings(state, Generators::Embeddings::Mode::Input, name, hidden_size);
    embeddings.Add();
    embeddings.UpdateSequenceLength(8);
    embeddings.UpdateSequenceLength(1);
    void* decode_buffer = embeddings.Get()->GetTensorMutableRawData();

    for (size_t prompt_length : {4, 19, 2}) {
      embeddings.UpdateSequenceLength(prompt_length);
      auto competing_buffer = OrtValue::CreateTensor(
          state.p_session_device_inputs_->GetAllocator(),
          std::array<int64_t, 3>{1, 1, hidden_size},
          model->session_info_.GetInputDataType(name));
      EXPECT_NE(competing_buffer->GetTensorMutableRawData(), decode_buffer);
      auto* prefill_tensor = embeddings.Get();
      embeddings.UseChunkView(1, prompt_length - 1);
      EXPECT_EQ(state.inputs_.back()->GetTensorTypeAndShapeInfo()->GetShape(),
                (std::vector<int64_t>{1, static_cast<int64_t>(prompt_length - 1), hidden_size}));
      embeddings.RestoreFullView();
      EXPECT_EQ(state.inputs_.back(), prefill_tensor);
      embeddings.UpdateSequenceLength(1);
      EXPECT_EQ(embeddings.Get()->GetTensorMutableRawData(), decode_buffer);
      EXPECT_EQ(state.inputs_.back(), embeddings.Get());
    }
  }
}

}  // namespace

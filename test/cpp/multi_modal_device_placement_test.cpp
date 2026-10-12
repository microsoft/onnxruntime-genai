// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Each state allocates against the device of the session it drives, never the decoder's
// (see SessionCanAccess).

#include <memory>

#include <gtest/gtest.h>

#include "models/model.h"
#include "models/io/embeddings.h"

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

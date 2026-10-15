// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "models/io/default_position_inputs.h"
#include "models/io/dynamic_attention_mask.h"
#include "models/io/static_attention_mask.h"
#include "telemetry_test_environment.h"

namespace Generators {
namespace {

// ONNX IR 8/opset 13: four Identity nodes with [batch, sequence] inputs.
// mask32/pos32 use INT32; mask64/pos64 use INT64. No external model assets.
constexpr uint8_t kMetadataGraph[] = {
    0x08, 0x08, 0x3a, 0xd1, 0x03, 0x0a, 0x1e, 0x0a, 0x06, 0x6d, 0x61, 0x73, 0x6b, 0x33, 0x32, 0x12,
    0x0a, 0x6d, 0x61, 0x73, 0x6b, 0x33, 0x32, 0x5f, 0x6f, 0x75, 0x74, 0x22, 0x08, 0x49, 0x64, 0x65,
    0x6e, 0x74, 0x69, 0x74, 0x79, 0x0a, 0x1c, 0x0a, 0x05, 0x70, 0x6f, 0x73, 0x33, 0x32, 0x12, 0x09,
    0x70, 0x6f, 0x73, 0x33, 0x32, 0x5f, 0x6f, 0x75, 0x74, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74,
    0x69, 0x74, 0x79, 0x0a, 0x1e, 0x0a, 0x06, 0x6d, 0x61, 0x73, 0x6b, 0x36, 0x34, 0x12, 0x0a, 0x6d,
    0x61, 0x73, 0x6b, 0x36, 0x34, 0x5f, 0x6f, 0x75, 0x74, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74,
    0x69, 0x74, 0x79, 0x0a, 0x1c, 0x0a, 0x05, 0x70, 0x6f, 0x73, 0x36, 0x34, 0x12, 0x09, 0x70, 0x6f,
    0x73, 0x36, 0x34, 0x5f, 0x6f, 0x75, 0x74, 0x22, 0x08, 0x49, 0x64, 0x65, 0x6e, 0x74, 0x69, 0x74,
    0x79, 0x12, 0x0f, 0x70, 0x6f, 0x73, 0x69, 0x74, 0x69, 0x6f, 0x6e, 0x5f, 0x69, 0x6e, 0x70, 0x75,
    0x74, 0x73, 0x5a, 0x25, 0x0a, 0x06, 0x6d, 0x61, 0x73, 0x6b, 0x33, 0x32, 0x12, 0x1b, 0x0a, 0x19,
    0x08, 0x06, 0x12, 0x15, 0x0a, 0x07, 0x12, 0x05, 0x62, 0x61, 0x74, 0x63, 0x68, 0x0a, 0x0a, 0x12,
    0x08, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x5a, 0x24, 0x0a, 0x05, 0x70, 0x6f, 0x73,
    0x33, 0x32, 0x12, 0x1b, 0x0a, 0x19, 0x08, 0x06, 0x12, 0x15, 0x0a, 0x07, 0x12, 0x05, 0x62, 0x61,
    0x74, 0x63, 0x68, 0x0a, 0x0a, 0x12, 0x08, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x5a,
    0x25, 0x0a, 0x06, 0x6d, 0x61, 0x73, 0x6b, 0x36, 0x34, 0x12, 0x1b, 0x0a, 0x19, 0x08, 0x07, 0x12,
    0x15, 0x0a, 0x07, 0x12, 0x05, 0x62, 0x61, 0x74, 0x63, 0x68, 0x0a, 0x0a, 0x12, 0x08, 0x73, 0x65,
    0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x5a, 0x24, 0x0a, 0x05, 0x70, 0x6f, 0x73, 0x36, 0x34, 0x12,
    0x1b, 0x0a, 0x19, 0x08, 0x07, 0x12, 0x15, 0x0a, 0x07, 0x12, 0x05, 0x62, 0x61, 0x74, 0x63, 0x68,
    0x0a, 0x0a, 0x12, 0x08, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x62, 0x29, 0x0a, 0x0a,
    0x6d, 0x61, 0x73, 0x6b, 0x33, 0x32, 0x5f, 0x6f, 0x75, 0x74, 0x12, 0x1b, 0x0a, 0x19, 0x08, 0x06,
    0x12, 0x15, 0x0a, 0x07, 0x12, 0x05, 0x62, 0x61, 0x74, 0x63, 0x68, 0x0a, 0x0a, 0x12, 0x08, 0x73,
    0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x62, 0x28, 0x0a, 0x09, 0x70, 0x6f, 0x73, 0x33, 0x32,
    0x5f, 0x6f, 0x75, 0x74, 0x12, 0x1b, 0x0a, 0x19, 0x08, 0x06, 0x12, 0x15, 0x0a, 0x07, 0x12, 0x05,
    0x62, 0x61, 0x74, 0x63, 0x68, 0x0a, 0x0a, 0x12, 0x08, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63,
    0x65, 0x62, 0x29, 0x0a, 0x0a, 0x6d, 0x61, 0x73, 0x6b, 0x36, 0x34, 0x5f, 0x6f, 0x75, 0x74, 0x12,
    0x1b, 0x0a, 0x19, 0x08, 0x07, 0x12, 0x15, 0x0a, 0x07, 0x12, 0x05, 0x62, 0x61, 0x74, 0x63, 0x68,
    0x0a, 0x0a, 0x12, 0x08, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x62, 0x28, 0x0a, 0x09,
    0x70, 0x6f, 0x73, 0x36, 0x34, 0x5f, 0x6f, 0x75, 0x74, 0x12, 0x1b, 0x0a, 0x19, 0x08, 0x07, 0x12,
    0x15, 0x0a, 0x07, 0x12, 0x05, 0x62, 0x61, 0x74, 0x63, 0x68, 0x0a, 0x0a, 0x12, 0x08, 0x73, 0x65,
    0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x0d,
};

std::unique_ptr<Config> MakeConfig(bool int64, int batch_size, int num_beams) {
  auto config = std::make_unique<Config>();
  config->model.type = "llama";
  config->model.context_length = 8;
  config->model.vocab_size = 16;
  config->model.pad_token_id = 0;
  config->model.decoder.inputs.attention_mask = int64 ? "mask64" : "mask32";
  config->model.decoder.inputs.position_ids = int64 ? "pos64" : "pos32";
  config->model.decoder.session_options.intra_op_num_threads = 1;
  config->search.max_length = 6;
  config->search.batch_size = batch_size;
  config->search.num_beams = num_beams;
  return config;
}

struct PositionTestModel : Model {
  PositionTestModel(bool int64, int batch_size, int num_beams)
      : Model{MakeConfig(int64, batch_size, num_beams)} {
    auto session = OrtSession::Create(GetOrtEnv(), kMetadataGraph, sizeof(kMetadataGraph), session_options_.get());
    session_info_.Add(*session);
  }

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const override {
    return nullptr;
  }
};

struct PositionTestState : State {
  using State::State;
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override { return {}; }
};

struct PositionInputsTest : testing::TestWithParam<std::tuple<bool, bool>> {
  void Initialize(int batch_size = 1, int num_beams = 1, bool has_positions = true, bool has_mask = true) {
    model = std::make_shared<PositionTestModel>(IsInt64(), batch_size, num_beams);
    if (!has_positions)
      model->config_->model.decoder.inputs.position_ids = "absent_positions";
    if (!has_mask)
      model->config_->model.decoder.inputs.attention_mask = "absent_mask";
    params = std::make_shared<GeneratorParams>(*model);
    state = std::make_unique<PositionTestState>(*params, *model);
    lengths = params->p_device->Allocate<int32_t>(batch_size * num_beams);
    const auto& name = model->config_->model.decoder.inputs.attention_mask;
    if (IsStatic())
      inputs = std::make_unique<DefaultPositionInputs>(*model, *state, lengths, name, 8);
    else
      inputs = std::make_unique<DefaultPositionInputs>(*model, *state, lengths, name);
    inputs->Add();
  }

  bool IsStatic() const { return std::get<0>(GetParam()); }
  bool IsInt64() const { return std::get<1>(GetParam()); }

  void Update(std::vector<int32_t> values, int total_length, int new_length) {
    auto tokens = params->p_device->Allocate<int32_t>(values.size());
    std::copy(values.begin(), values.end(), tokens.CpuSpan().begin());
    inputs->Update(tokens, total_length, new_length);
  }

  OrtValue& Mask() { return *state->GetInput(model->config_->model.decoder.inputs.attention_mask.c_str()); }

  std::vector<int64_t> Read(OrtValue& tensor) {
    const auto count = tensor.GetTensorTypeAndShapeInfo()->GetElementCount();
    if (IsInt64()) {
      auto* data = tensor.GetTensorData<int64_t>();
      return {data, data + count};
    }
    auto* data = tensor.GetTensorData<int32_t>();
    return {data, data + count};
  }

  void ExpectMask(const std::vector<std::vector<int64_t>>& rows) {
    const size_t width = IsStatic() ? 8 : rows.front().size();
    EXPECT_EQ(Mask().GetTensorTypeAndShapeInfo()->GetShape(),
              (std::vector<int64_t>{static_cast<int64_t>(rows.size()), static_cast<int64_t>(width)}));
    std::vector<int64_t> expected;
    for (auto row : rows) {
      row.resize(width, 0);
      expected.insert(expected.end(), row.begin(), row.end());
    }
    EXPECT_EQ(Read(Mask()), expected);
  }

  std::shared_ptr<PositionTestModel> model;
  std::shared_ptr<GeneratorParams> params;
  std::unique_ptr<PositionTestState> state;
  DeviceSpan<int32_t> lengths;
  std::unique_ptr<DefaultPositionInputs> inputs;
};

TEST_P(PositionInputsTest, PrefillDecodeRewindAndRestart) {
  Initialize();
  Update({2, 3, 4}, 3, 3);
  ExpectMask({{1, 1, 1}});
  EXPECT_EQ(Read(*state->GetInput(model->config_->model.decoder.inputs.position_ids.c_str())),
            (std::vector<int64_t>{0, 1, 2}));
  const auto* initial_mask = Mask().GetTensorRawData();
  for (int length = 4; length <= 6; ++length) {
    Update({5}, length, 1);
    ExpectMask({std::vector<int64_t>(length, 1)});
    EXPECT_EQ(Read(*state->GetInput(model->config_->model.decoder.inputs.position_ids.c_str())),
              (std::vector<int64_t>{length - 1}));
    if (IsStatic()) EXPECT_EQ(Mask().GetTensorRawData(), initial_mask);
  }
  inputs->RewindTo(2);
  if (IsStatic()) ExpectMask({{1, 1}});
  Update({6}, 3, 1);
  ExpectMask({{1, 1, 1}});
  inputs->RewindTo(0);
  Update({2, 3}, 2, 2);
  ExpectMask({{1, 1}});
}

TEST_P(PositionInputsTest, PreservesPaddingAndExpandsBeams) {
  Initialize(2, 2);
  Update({2, 3, 0, 4, 0, 0}, 3, 3);
  ExpectMask({{1, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 0}});
  Update({5, 5, 6, 6}, 4, 1);
  ExpectMask({{1, 1, 0, 1}, {1, 1, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}});
  EXPECT_THROW(inputs->RewindTo(2), std::runtime_error);
}

TEST_P(PositionInputsTest, FactoryRetainsGraphCaptureSelection) {
  Initialize();
  params->use_graph_capture = IsStatic();
  const auto type = IsInt64() ? Ort::TypeToTensorType<int64_t> : Ort::TypeToTensorType<int32_t>;
  auto selected = CreateAttentionMask(*model, *state, type);
  if (IsStatic()) {
    EXPECT_NE(dynamic_cast<StaticAttentionMask*>(selected.get()), nullptr);
  } else {
    EXPECT_NE(dynamic_cast<DynamicAttentionMask*>(selected.get()), nullptr);
  }

  PositionTestState default_state{*params, *model};
  const auto& name = model->config_->model.decoder.inputs.attention_mask;
  DefaultPositionInputs default_inputs{*model, default_state, lengths, name};
  default_inputs.Add();
  auto tokens = params->p_device->Allocate<int32_t>(3);
  std::fill(tokens.CpuSpan().begin(), tokens.CpuSpan().end(), 2);
  default_inputs.Update(tokens, 3, 3);
  EXPECT_EQ(default_state.GetInput(name.c_str())->GetTensorTypeAndShapeInfo()->GetShape()[1],
            IsStatic() ? params->search.max_length : 3);
}

TEST_P(PositionInputsTest, MaskWithoutPositionIds) {
  Initialize(2, 2, false);
  Update({2, 3, 0, 4, 0, 0}, 3, 3);
  ExpectMask({{1, 1, 0}, {1, 1, 0}, {1, 0, 0}, {1, 0, 0}});
  Update({5, 5, 6, 6}, 4, 1);
  ExpectMask({{1, 1, 0, 1}, {1, 1, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}});
  inputs->RewindTo(0);
  Update({2, 0, 3, 0}, 2, 2);
  ExpectMask({{1, 0}, {1, 0}, {1, 0}, {1, 0}});
}

TEST_P(PositionInputsTest, PositionIdsWithoutMask) {
  Initialize(1, 1, true, false);
  const auto& name = model->config_->model.decoder.inputs.position_ids;
  Update({2, 3, 4}, 3, 3);
  EXPECT_EQ(state->inputs_.size(), 1);
  EXPECT_EQ(Read(*state->GetInput(name.c_str())), (std::vector<int64_t>{0, 1, 2}));
  Update({5}, 4, 1);
  EXPECT_EQ(Read(*state->GetInput(name.c_str())), (std::vector<int64_t>{3}));
  inputs->RewindTo(0);
  Update({2, 3}, 2, 2);
  EXPECT_EQ(Read(*state->GetInput(name.c_str())), (std::vector<int64_t>{0, 1}));
}

INSTANTIATE_TEST_SUITE_P(MaskModesAndTypes, PositionInputsTest,
                        testing::Combine(testing::Bool(), testing::Bool()));

TEST(PositionInputsCapacityTest, UsesExplicitCapacityAndRejectsOverflow) {
  auto model = std::make_shared<PositionTestModel>(true, 1, 1);
  auto params = std::make_shared<GeneratorParams>(*model);
  PositionTestState state{*params, *model};
  auto lengths = params->p_device->Allocate<int32_t>(1);
  const auto& name = model->config_->model.decoder.inputs.attention_mask;
  EXPECT_THROW(DefaultPositionInputs(*model, state, lengths, name, 0), std::runtime_error);
  EXPECT_THROW(DefaultPositionInputs(*model, state, lengths, name, -1), std::runtime_error);
  DefaultPositionInputs inputs{*model, state, lengths, name, 8};
  inputs.Add();
  auto tokens = params->p_device->Allocate<int32_t>(9);
  std::fill(tokens.CpuSpan().begin(), tokens.CpuSpan().end(), 2);
  EXPECT_THROW(inputs.Update(tokens, 9, 9), std::runtime_error);
  inputs.Update(tokens.subspan(0, 3), 3, 3);
  EXPECT_EQ(state.GetInput(name.c_str())->GetTensorTypeAndShapeInfo()->GetShape()[1], 8);
  inputs.Update(tokens.subspan(0, 1), 8, 1);
  EXPECT_THROW(inputs.Update(tokens.subspan(0, 1), 9, 1), std::runtime_error);
  EXPECT_THROW(inputs.RewindTo(9), std::runtime_error);
}

}  // namespace
}  // namespace Generators

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}

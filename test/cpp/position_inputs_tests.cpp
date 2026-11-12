// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "models/io/default_position_inputs.h"
#include "models/io/logits.h"
#include "models/io/windowed_position_inputs.h"
#include "models/model.h"

namespace Generators {
namespace {

// ONNX IR 8/opset 13: four Identity nodes with [batch, sequence] inputs.
// mask32/pos32 use INT32; mask64/pos64 use INT64. No external model assets.
constexpr uint8_t kMetadataGraph[] = {
    0x08,
    0x08,
    0x3a,
    0xd1,
    0x03,
    0x0a,
    0x1e,
    0x0a,
    0x06,
    0x6d,
    0x61,
    0x73,
    0x6b,
    0x33,
    0x32,
    0x12,
    0x0a,
    0x6d,
    0x61,
    0x73,
    0x6b,
    0x33,
    0x32,
    0x5f,
    0x6f,
    0x75,
    0x74,
    0x22,
    0x08,
    0x49,
    0x64,
    0x65,
    0x6e,
    0x74,
    0x69,
    0x74,
    0x79,
    0x0a,
    0x1c,
    0x0a,
    0x05,
    0x70,
    0x6f,
    0x73,
    0x33,
    0x32,
    0x12,
    0x09,
    0x70,
    0x6f,
    0x73,
    0x33,
    0x32,
    0x5f,
    0x6f,
    0x75,
    0x74,
    0x22,
    0x08,
    0x49,
    0x64,
    0x65,
    0x6e,
    0x74,
    0x69,
    0x74,
    0x79,
    0x0a,
    0x1e,
    0x0a,
    0x06,
    0x6d,
    0x61,
    0x73,
    0x6b,
    0x36,
    0x34,
    0x12,
    0x0a,
    0x6d,
    0x61,
    0x73,
    0x6b,
    0x36,
    0x34,
    0x5f,
    0x6f,
    0x75,
    0x74,
    0x22,
    0x08,
    0x49,
    0x64,
    0x65,
    0x6e,
    0x74,
    0x69,
    0x74,
    0x79,
    0x0a,
    0x1c,
    0x0a,
    0x05,
    0x70,
    0x6f,
    0x73,
    0x36,
    0x34,
    0x12,
    0x09,
    0x70,
    0x6f,
    0x73,
    0x36,
    0x34,
    0x5f,
    0x6f,
    0x75,
    0x74,
    0x22,
    0x08,
    0x49,
    0x64,
    0x65,
    0x6e,
    0x74,
    0x69,
    0x74,
    0x79,
    0x12,
    0x0f,
    0x70,
    0x6f,
    0x73,
    0x69,
    0x74,
    0x69,
    0x6f,
    0x6e,
    0x5f,
    0x69,
    0x6e,
    0x70,
    0x75,
    0x74,
    0x73,
    0x5a,
    0x25,
    0x0a,
    0x06,
    0x6d,
    0x61,
    0x73,
    0x6b,
    0x33,
    0x32,
    0x12,
    0x1b,
    0x0a,
    0x19,
    0x08,
    0x06,
    0x12,
    0x15,
    0x0a,
    0x07,
    0x12,
    0x05,
    0x62,
    0x61,
    0x74,
    0x63,
    0x68,
    0x0a,
    0x0a,
    0x12,
    0x08,
    0x73,
    0x65,
    0x71,
    0x75,
    0x65,
    0x6e,
    0x63,
    0x65,
    0x5a,
    0x24,
    0x0a,
    0x05,
    0x70,
    0x6f,
    0x73,
    0x33,
    0x32,
    0x12,
    0x1b,
    0x0a,
    0x19,
    0x08,
    0x06,
    0x12,
    0x15,
    0x0a,
    0x07,
    0x12,
    0x05,
    0x62,
    0x61,
    0x74,
    0x63,
    0x68,
    0x0a,
    0x0a,
    0x12,
    0x08,
    0x73,
    0x65,
    0x71,
    0x75,
    0x65,
    0x6e,
    0x63,
    0x65,
    0x5a,
    0x25,
    0x0a,
    0x06,
    0x6d,
    0x61,
    0x73,
    0x6b,
    0x36,
    0x34,
    0x12,
    0x1b,
    0x0a,
    0x19,
    0x08,
    0x07,
    0x12,
    0x15,
    0x0a,
    0x07,
    0x12,
    0x05,
    0x62,
    0x61,
    0x74,
    0x63,
    0x68,
    0x0a,
    0x0a,
    0x12,
    0x08,
    0x73,
    0x65,
    0x71,
    0x75,
    0x65,
    0x6e,
    0x63,
    0x65,
    0x5a,
    0x24,
    0x0a,
    0x05,
    0x70,
    0x6f,
    0x73,
    0x36,
    0x34,
    0x12,
    0x1b,
    0x0a,
    0x19,
    0x08,
    0x07,
    0x12,
    0x15,
    0x0a,
    0x07,
    0x12,
    0x05,
    0x62,
    0x61,
    0x74,
    0x63,
    0x68,
    0x0a,
    0x0a,
    0x12,
    0x08,
    0x73,
    0x65,
    0x71,
    0x75,
    0x65,
    0x6e,
    0x63,
    0x65,
    0x62,
    0x29,
    0x0a,
    0x0a,
    0x6d,
    0x61,
    0x73,
    0x6b,
    0x33,
    0x32,
    0x5f,
    0x6f,
    0x75,
    0x74,
    0x12,
    0x1b,
    0x0a,
    0x19,
    0x08,
    0x06,
    0x12,
    0x15,
    0x0a,
    0x07,
    0x12,
    0x05,
    0x62,
    0x61,
    0x74,
    0x63,
    0x68,
    0x0a,
    0x0a,
    0x12,
    0x08,
    0x73,
    0x65,
    0x71,
    0x75,
    0x65,
    0x6e,
    0x63,
    0x65,
    0x62,
    0x28,
    0x0a,
    0x09,
    0x70,
    0x6f,
    0x73,
    0x33,
    0x32,
    0x5f,
    0x6f,
    0x75,
    0x74,
    0x12,
    0x1b,
    0x0a,
    0x19,
    0x08,
    0x06,
    0x12,
    0x15,
    0x0a,
    0x07,
    0x12,
    0x05,
    0x62,
    0x61,
    0x74,
    0x63,
    0x68,
    0x0a,
    0x0a,
    0x12,
    0x08,
    0x73,
    0x65,
    0x71,
    0x75,
    0x65,
    0x6e,
    0x63,
    0x65,
    0x62,
    0x29,
    0x0a,
    0x0a,
    0x6d,
    0x61,
    0x73,
    0x6b,
    0x36,
    0x34,
    0x5f,
    0x6f,
    0x75,
    0x74,
    0x12,
    0x1b,
    0x0a,
    0x19,
    0x08,
    0x07,
    0x12,
    0x15,
    0x0a,
    0x07,
    0x12,
    0x05,
    0x62,
    0x61,
    0x74,
    0x63,
    0x68,
    0x0a,
    0x0a,
    0x12,
    0x08,
    0x73,
    0x65,
    0x71,
    0x75,
    0x65,
    0x6e,
    0x63,
    0x65,
    0x62,
    0x28,
    0x0a,
    0x09,
    0x70,
    0x6f,
    0x73,
    0x36,
    0x34,
    0x5f,
    0x6f,
    0x75,
    0x74,
    0x12,
    0x1b,
    0x0a,
    0x19,
    0x08,
    0x07,
    0x12,
    0x15,
    0x0a,
    0x07,
    0x12,
    0x05,
    0x62,
    0x61,
    0x74,
    0x63,
    0x68,
    0x0a,
    0x0a,
    0x12,
    0x08,
    0x73,
    0x65,
    0x71,
    0x75,
    0x65,
    0x6e,
    0x63,
    0x65,
    0x42,
    0x04,
    0x0a,
    0x00,
    0x10,
    0x0d,
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
    lengths = params->p_device->Allocate<int32_t>(static_cast<size_t>(batch_size) * num_beams);
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

TEST_P(PositionInputsTest, AllPadBatchRowInitializesNextPositions) {
  Initialize(2, 1);
  const auto& pos_name = model->config_->model.decoder.inputs.position_ids;
  Update({2, 3, 0, 0, 0, 0}, 3, 3);
  ExpectMask({{1, 1, 0}, {0, 0, 0}});
  EXPECT_EQ(Read(*state->GetInput(pos_name.c_str())),
            (std::vector<int64_t>{0, 1, 0, 0, 0, 0}));
  Update({5, 6}, 4, 1);
  ExpectMask({{1, 1, 0, 1}, {0, 0, 0, 1}});
  EXPECT_EQ(Read(*state->GetInput(pos_name.c_str())),
            (std::vector<int64_t>{2, 1}));
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

struct WindowedPositionInputsTest : testing::TestWithParam<bool> {
  void Initialize(int context_length = 6, int window_size = 3) {
    inputs.reset();
    state.reset();
    params.reset();
    model.reset();
    model = std::make_shared<PositionTestModel>(GetParam(), 1, 1);
    model->config_->model.context_length = context_length;
    Config::Model::Decoder::SlidingWindow sw{};
    sw.window_size = window_size;
    sw.slide_inputs = true;
    model->config_->model.decoder.sliding_window = sw;
    params = std::make_shared<GeneratorParams>(*model);
    state = std::make_unique<PositionTestState>(*params, *model);
    inputs = std::make_unique<WindowedPositionInputs>(*state);
    inputs->Add();
  }

  void Update(std::vector<int32_t> values, int total_length, int new_length) {
    auto tokens = params->p_device->Allocate<int32_t>(values.size());
    std::copy(values.begin(), values.end(), tokens.CpuSpan().begin());
    inputs->Update(tokens, total_length, new_length);
  }

  std::vector<int64_t> ReadInput(const std::string& name) {
    auto& tensor = *state->GetInput(name.c_str());
    const auto count = tensor.GetTensorTypeAndShapeInfo()->GetElementCount();
    if (GetParam()) {
      const auto* data = tensor.GetTensorData<int64_t>();
      return {data, data + count};
    }
    const auto* data = tensor.GetTensorData<int32_t>();
    return {data, data + count};
  }

  std::shared_ptr<PositionTestModel> model;
  std::shared_ptr<GeneratorParams> params;
  std::unique_ptr<PositionTestState> state;
  std::unique_ptr<WindowedPositionInputs> inputs;
};

TEST_P(WindowedPositionInputsTest, MultiWindowPrefillAndDecode) {
  Initialize(6, 3);
  const auto& mask_name = model->config_->model.decoder.inputs.attention_mask;
  const auto& pos_name = model->config_->model.decoder.inputs.position_ids;

  Update({0, 2, 3, 4, 5, 6}, 6, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 0, 0, 0, 1, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{0, 0, 1}));

  Update({0, 2, 3, 4, 5, 6}, 6, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 1, 1, 1, 1, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{2, 3, 4}));

  Update({7}, 7, 1);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{1, 1, 1, 1, 1, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{5}));
}

TEST_P(WindowedPositionInputsTest, AllPadFirstWindowDoesNotOverflow) {
  Initialize(6, 3);
  const auto& mask_name = model->config_->model.decoder.inputs.attention_mask;
  const auto& pos_name = model->config_->model.decoder.inputs.position_ids;

  Update({0, 0, 0, 4, 5, 6}, 6, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{0, 0, 0}));

  Update({0, 0, 0, 4, 5, 6}, 6, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 0, 0, 1, 1, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{0, 1, 2}));

  Update({7}, 7, 1);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 0, 1, 1, 1, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{3}));
}

TEST_P(WindowedPositionInputsTest, MultipleAllPadWindowsPreserveZeroMaskAndStartPosition) {
  Initialize(6, 3);
  const auto& mask_name = model->config_->model.decoder.inputs.attention_mask;
  const auto& pos_name = model->config_->model.decoder.inputs.position_ids;

  Update({0, 0, 0, 0, 0, 0}, 6, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{0, 0, 0}));

  Update({0, 0, 0, 0, 0, 0}, 6, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{0, 0, 0}));

  Update({7}, 7, 1);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 0, 0, 0, 0, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{0}));
}

TEST_P(WindowedPositionInputsTest, PartiallyPaddedSecondWindowPreservesMaskAndPositions) {
  Initialize(6, 3);
  const auto& mask_name = model->config_->model.decoder.inputs.attention_mask;
  const auto& pos_name = model->config_->model.decoder.inputs.position_ids;

  Update({2, 3, 4, 5, 0, 0}, 6, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{0, 0, 0, 1, 1, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{0, 1, 2}));

  Update({2, 3, 4, 5, 0, 0}, 6, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{1, 1, 1, 1, 0, 0}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{3, 0, 0}));

  Update({7}, 7, 1);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{1, 1, 1, 0, 0, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{4}));
}

TEST_P(WindowedPositionInputsTest, ContextLengthEqualToWindowSizeDoesNotUnderflow) {
  Initialize(3, 3);
  const auto& mask_name = model->config_->model.decoder.inputs.attention_mask;
  const auto& pos_name = model->config_->model.decoder.inputs.position_ids;

  Update({2, 3, 4}, 3, 3);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{1, 1, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{0, 1, 2}));

  Update({5}, 4, 1);
  EXPECT_EQ(ReadInput(mask_name), (std::vector<int64_t>{1, 1, 1}));
  EXPECT_EQ(ReadInput(pos_name), (std::vector<int64_t>{3}));
}

TEST_P(WindowedPositionInputsTest, RejectsInvalidWindowDimensionsAndShortInput) {
  EXPECT_THROW(Initialize(6, 0), std::runtime_error);
  EXPECT_THROW(Initialize(6, -2), std::runtime_error);
  EXPECT_THROW(Initialize(2, 3), std::runtime_error);

  Initialize(6, 3);
  EXPECT_THROW(Update({2, 3}, 2, 2), std::runtime_error);
  EXPECT_THROW(Update({2, 3, 4, 5}, 4, 3), std::runtime_error);
  Update({2, 3, 4, 5, 6, 7}, 6, 3);
  EXPECT_THROW(Update({2, 3, 4}, 6, 3), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(IntTypes, WindowedPositionInputsTest, testing::Bool());

TEST(LogitsAllPadTest, SlicesValidBoundsOnAllPadSequences) {
  auto base_model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
  const size_t vocab_size = static_cast<size_t>(base_model->config_->model.vocab_size);

  // Single-sequence all-pad prefill uses the final prompt token slot.
  {
    auto params = std::make_shared<GeneratorParams>(*base_model);
    PositionTestState state{*params, *base_model};
    Logits logits{state};
    logits.Add();

    auto tokens = params->p_device->Allocate<int32_t>(3);
    std::fill(tokens.CpuSpan().begin(), tokens.CpuSpan().end(), base_model->config_->model.pad_token_id);
    logits.Update(tokens, 3);

    auto* raw_logits = state.outputs_[0]->GetTensorMutableData<float>();
    for (size_t pos = 0; pos < 3; ++pos) {
      for (size_t v = 0; v < vocab_size; ++v) {
        raw_logits[pos * vocab_size + v] = static_cast<float>(pos * 100 + v);
      }
    }

    auto sliced = logits.Get();
    auto sliced_cpu = sliced.CopyDeviceToCpu();
    ASSERT_EQ(sliced_cpu.size(), vocab_size);
    EXPECT_FLOAT_EQ(sliced_cpu[0], 200.0f);
    EXPECT_FLOAT_EQ(sliced_cpu[vocab_size - 1], static_cast<float>(200 + vocab_size - 1));
  }

  // Batched prefill with an all-pad batch row clamps that row to slot 0 without underflowing.
  {
    auto params = std::make_shared<GeneratorParams>(*base_model);
    params->search.batch_size = 2;
    PositionTestState state{*params, *base_model};
    Logits logits{state};
    logits.Add();

    const int32_t pad = base_model->config_->model.pad_token_id;
    std::array<int32_t, 6> batch_tokens{2, 3, pad, pad, pad, pad};
    auto tokens = params->p_device->Allocate<int32_t>(batch_tokens.size());
    std::copy(batch_tokens.begin(), batch_tokens.end(), tokens.CpuSpan().begin());
    logits.Update(tokens, 3);

    auto* raw_logits = state.outputs_[0]->GetTensorMutableData<float>();
    for (size_t b = 0; b < 2; ++b) {
      for (size_t pos = 0; pos < 3; ++pos) {
        for (size_t v = 0; v < vocab_size; ++v) {
          raw_logits[(b * 3 + pos) * vocab_size + v] = static_cast<float>(b * 1000 + pos * 100 + v);
        }
      }
    }

    auto sliced = logits.Get();
    auto sliced_cpu = sliced.CopyDeviceToCpu();
    ASSERT_EQ(sliced_cpu.size(), 2 * vocab_size);
    // Batch 0 last non-pad token is pos=1; Batch 1 is all-pad and clamps to pos=0.
    EXPECT_FLOAT_EQ(sliced_cpu[0], 100.0f);
    EXPECT_FLOAT_EQ(sliced_cpu[vocab_size], 1000.0f);
  }

  // Left-aligned sliding_window only right-pads when slide_inputs is true.
  for (bool slide_inputs : {false, true}) {
    auto model_copy = CreateModel(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    Config::Model::Decoder::SlidingWindow sw{};
    sw.window_size = 3;
    sw.alignment = "left";
    sw.slide_inputs = slide_inputs;
    model_copy->config_->model.decoder.sliding_window = sw;

    auto params = std::make_shared<GeneratorParams>(*model_copy);
    PositionTestState state{*params, *model_copy};
    Logits logits{state};
    logits.Add();

    auto tokens = params->p_device->Allocate<int32_t>(3);
    std::fill(tokens.CpuSpan().begin(), tokens.CpuSpan().end(), model_copy->config_->model.pad_token_id);
    logits.Update(tokens, 3);

    auto* raw_logits = state.outputs_[0]->GetTensorMutableData<float>();
    for (size_t pos = 0; pos < 3; ++pos) {
      for (size_t v = 0; v < vocab_size; ++v) {
        raw_logits[pos * vocab_size + v] = static_cast<float>(pos * 100 + v);
      }
    }

    auto sliced = logits.Get();
    auto sliced_cpu = sliced.CopyDeviceToCpu();
    ASSERT_EQ(sliced_cpu.size(), vocab_size);
    EXPECT_FLOAT_EQ(sliced_cpu[0], slide_inputs ? 0.0f : 200.0f);
  }
}

}  // namespace
}  // namespace Generators

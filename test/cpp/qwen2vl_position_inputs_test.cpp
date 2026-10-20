// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <array>
#include <vector>

#include "models/io/default_position_inputs.h"
#include "models/io/position_inputs.h"
#include "models/io/qwen_vl_position_inputs.h"
#include "models/io/windowed_position_inputs.h"
#include "test_utils.h"

namespace Generators::test {
namespace {

template <typename Fn>
std::string CaptureThrowMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

struct PositionInputsTestState : State {
  using State::State;

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override {
    return {};
  }
};

class StandardPositionInputsTest : public testing::Test {
 protected:
  void SetUp() override {
    model_ = CreateModel(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    params_ = std::make_shared<GeneratorParams>(*model_);
    state_ = std::make_unique<PositionInputsTestState>(*params_, *model_);
    sequence_lengths_ = model_->p_device_->Allocate<int32_t>(params_->search.batch_size);
  }

  void Use3DPositionIds() {
    session_ = model_->CreateSession(GetOrtEnv(), "../../qwen3-vl/dummy_text.onnx", model_->session_options_.get());
    model_->session_info_ = SessionInfo{};
    model_->session_info_.Add(*session_);
    ASSERT_EQ(model_->session_info_.GetInputShape("position_ids").size(), 3u);
  }

  void UseRotaryPositionHistory() {
    session_ = model_->CreateSession(GetOrtEnv(), "../synthetic-rotary-history/positions.onnx",
                                     model_->session_options_.get());
    model_->session_info_ = SessionInfo{};
    model_->session_info_.Add(*session_);
    model_->config_->model.type = "qwen4_exp_text";
    model_->config_->model.decoder.inputs.rotary_position_ids = "rotary_position_ids";
    model_->config_->model.vision.spatial_merge_size = 1;
    model_->config_->model.image_token_id = 7;
    model_->config_->model.vision_start_token_id = 6;
    model_->config_->model.pad_token_id = 0;
    model_->config_->model.context_length = 32;
    params_->search.max_length = 16;
  }

  DeviceSpan<int32_t> Tokens(std::span<const int32_t> values) {
    auto tokens = model_->p_device_->Allocate<int32_t>(values.size());
    std::copy(values.begin(), values.end(), tokens.CpuSpan().begin());
    tokens.CopyCpuToDevice();
    return tokens;
  }

  std::unique_ptr<PositionInputs> CreatePositionInputs() {
    return CreateStandardPositionInputs(*state_, sequence_lengths_, model_->config_->model.decoder.inputs.attention_mask);
  }

  static constexpr std::array<const char*, 5> qwen_model_types_{
      "qwen2_5_vl", "qwen3_vl", "qwen3_5", "qwen3_5_text", "qwen3_5_moe_text"};
  std::shared_ptr<Model> model_;
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<PositionInputsTestState> state_;
  DeviceSpan<int32_t> sequence_lengths_;
  std::unique_ptr<OrtSession> session_;
};

}  // namespace

TEST_F(StandardPositionInputsTest, QwenVLWith3DPositionIdsUsesMrope) {
  Use3DPositionIds();
  for (const auto* model_type : qwen_model_types_) {
    SCOPED_TRACE(model_type);
    model_->config_->model.type = model_type;
    auto inputs = CreatePositionInputs();
    EXPECT_NE(dynamic_cast<Qwen2VLPositionInputs*>(inputs.get()), nullptr);
  }
}

TEST_F(StandardPositionInputsTest, RotaryHistoryRetainsImageCoordinatesThroughDecodeAndReset) {
  UseRotaryPositionHistory();
  auto positions = CreatePositionInputs();
  auto* qwen = dynamic_cast<Qwen2VLPositionInputs*>(positions.get());
  ASSERT_NE(qwen, nullptr);
  auto grid = OrtValue::CreateTensor<int64_t>(model_->allocator_cpu_, std::array<int64_t, 2>{1, 3});
  const std::array<int64_t, 3> dimensions{1, 2, 2};
  std::copy(dimensions.begin(), dimensions.end(), grid->GetTensorMutableData<int64_t>());
  qwen->SetGridTensors(std::make_shared<Tensor>(std::move(grid)), {}, {});
  positions->Add();
  auto prompt = Tokens(std::array<int32_t, 7>{5, 6, 7, 7, 7, 7, 5});
  positions->Update(prompt, 7, 7);
  auto* history = positions->GetRotaryPositionIds();
  ASSERT_NE(history, nullptr);
  EXPECT_EQ(history->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{3, 1, 16}));
  const auto* values = history->GetTensorData<int64_t>();
  EXPECT_EQ(values[2], 2);
  EXPECT_EQ(values[16 + 4], 3);
  EXPECT_EQ(values[32 + 3], 3);
  EXPECT_EQ(values[7], 5);
  EXPECT_EQ(values[16 + 7], 5);
  EXPECT_EQ(values[32 + 7], 5);
  auto decode = Tokens(std::array<int32_t, 1>{9});
  positions->Update(decode, 8, 1);
  EXPECT_EQ(state_->GetInput("position_ids")->GetTensorData<int64_t>()[0], 5);
  positions->RewindTo(7);
  positions->Update(decode, 8, 1);
  EXPECT_EQ(state_->GetInput("position_ids")->GetTensorData<int64_t>()[0], 5);
  EXPECT_EQ(values[16 + 4], 3);
  positions->RewindTo(0);
  qwen->SetGridTensors({}, {}, {});
  auto next_prompt = Tokens(std::array<int32_t, 3>{5, 5, 5});
  positions->Update(next_prompt, 3, 3);
  for (size_t plane = 0; plane < 3; ++plane)
    for (size_t token = 0; token < 16; ++token)
      EXPECT_EQ(values[plane * 16 + token], static_cast<int64_t>(token));
}

TEST_F(StandardPositionInputsTest, MtpCoordinatesComeFromMainHistoryNotTextOnlyTokenIndices) {
  UseRotaryPositionHistory();
  auto positions = CreatePositionInputs();
  positions->Add();
  auto source = OrtValue::CreateTensor<int64_t>(model_->allocator_cpu_, std::array<int64_t, 3>{3, 1, 16});
  RotaryPositionHistory history{1, 16};
  history.Update(std::array<int64_t, 9>{0, 1, 1, 0, 2, 2, 0, 1, 4}, 0, 3);
  std::copy(history.Values().begin(), history.Values().end(), source->GetTensorMutableData<int64_t>());
  positions->SetRotaryPositionIds(source.get());
  auto draft = Tokens(std::array<int32_t, 1>{9});
  positions->Update(draft, 4, 1);
  const auto* current = state_->GetInput("position_ids")->GetTensorData<int64_t>();
  EXPECT_EQ(current[0], 5);
  EXPECT_EQ(current[1], 5);
  EXPECT_EQ(current[2], 5);
  const auto* retained = positions->GetRotaryPositionIds()->GetTensorData<int64_t>();
  EXPECT_EQ(retained[16 + 1], 2);
  EXPECT_EQ(retained[32 + 2], 4);
  positions->RewindTo(2);
  positions->Update(draft, 3, 1);
  current = state_->GetInput("position_ids")->GetTensorData<int64_t>();
  EXPECT_EQ(current[0], 1);
  EXPECT_EQ(current[1], 2);
  EXPECT_EQ(current[2], 4);
}

TEST_F(StandardPositionInputsTest, CapturedMtpPositionsKeepPerLengthBuffersAcrossUpdates) {
  UseRotaryPositionHistory();
  params_->use_graph_capture = true;
  params_->max_graph_capture_length = 3;
  auto positions = CreatePositionInputs();
  positions->Add();
  auto source = OrtValue::CreateTensor<int64_t>(model_->allocator_cpu_, std::array<int64_t, 3>{3, 1, 16});
  for (size_t index = 0; index < 48; ++index)
    source->GetTensorMutableData<int64_t>()[index] = index % 16;
  positions->SetRotaryPositionIds(source.get());
  std::array<void*, 3> buffers{};
  int length = 0;
  for (int count = 1; count <= 3; ++count) {
    length += count;
    auto tokens = Tokens(std::vector<int32_t>(count, 5));
    positions->Update(tokens, length, count);
    buffers[count - 1] = state_->GetInput("position_ids")->GetTensorMutableRawData();
  }
  auto prefill = Tokens(std::array<int32_t, 4>{5, 5, 5, 5});
  positions->Update(prefill, 10, 4);
  length = 10;
  for (int count = 1; count <= 3; ++count) {
    length += count;
    auto tokens = Tokens(std::vector<int32_t>(count, 5));
    positions->Update(tokens, length, count);
    auto* current = state_->GetInput("position_ids");
    EXPECT_EQ(current->GetTensorMutableRawData(), buffers[count - 1]);
    EXPECT_EQ(current->GetTensorData<int64_t>()[0], length - count);
  }
}

TEST(Qwen2VLProcessorTest, CreatesProcessorForQwen4ExpThroughThePublicModelApi) {
  auto config = CreateConfig(GetOrtEnv(), MODEL_PATH "qwen3-vl");
  config->model.type = "qwen4_exp";
  auto model = CreateModel(GetOrtEnv(), std::move(config));
  EXPECT_NE(model->CreateMultiModalProcessor(), nullptr);
}

TEST_F(StandardPositionInputsTest, QwenVLWith2DPositionIdsUsesDefault) {
  ASSERT_EQ(model_->session_info_.GetInputShape("position_ids").size(), 2u);
  for (const auto* model_type : qwen_model_types_) {
    SCOPED_TRACE(model_type);
    model_->config_->model.type = model_type;
    auto inputs = CreatePositionInputs();
    EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
  }
}

TEST_F(StandardPositionInputsTest, QwenVLWithoutConfiguredPositionIdsUsesDefault) {
  model_->config_->model.decoder.inputs.position_ids = "missing_position_ids";
  ASSERT_FALSE(model_->session_info_.HasInput("missing_position_ids"));
  for (const auto* model_type : qwen_model_types_) {
    SCOPED_TRACE(model_type);
    model_->config_->model.type = model_type;
    auto inputs = CreatePositionInputs();
    EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
  }
}

TEST_F(StandardPositionInputsTest, QwenVLWithEmptyPositionIdsNameUsesDefault) {
  model_->config_->model.type = "qwen3_5_text";
  model_->config_->model.decoder.inputs.position_ids.clear();
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, UsesConfiguredPositionIdsNameForRankCheck) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_5_text";
  model_->config_->model.decoder.inputs.position_ids = "attention_mask";
  ASSERT_EQ(model_->session_info_.GetInputShape("attention_mask").size(), 2u);
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, NonQwenModelWith3DPositionIdsUsesDefault) {
  Use3DPositionIds();
  model_->config_->model.type = "llama";
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<DefaultPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, QwenVLWith2DPositionIdsUsesSlidingWindowWhenEnabled) {
  model_->config_->model.type = "qwen3_5_text";
  auto& sliding_window = model_->config_->model.decoder.sliding_window.emplace();
  sliding_window.window_size = 8;
  sliding_window.slide_inputs = true;
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<WindowedPositionInputs*>(inputs.get()), nullptr);
}

TEST_F(StandardPositionInputsTest, QwenVLWith3DPositionIdsPrefersMropeOverSlidingWindow) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_5_text";
  auto& sliding_window = model_->config_->model.decoder.sliding_window.emplace();
  sliding_window.window_size = 8;
  sliding_window.slide_inputs = true;
  auto inputs = CreatePositionInputs();
  EXPECT_NE(dynamic_cast<Qwen2VLPositionInputs*>(inputs.get()), nullptr);
}

TEST(Qwen2VLPositionInputsTest, RejectsNegativeGridDimensions) {
  const std::vector<int64_t> grid{1, -1, 2};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw");
  });
  EXPECT_NE(message.find("non-negative"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsExcessiveGridDimensions) {
  const std::vector<int64_t> grid{1, 1, 20000};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "video_grid_thw");
  });
  EXPECT_NE(message.find("<= 16384"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsVisionLenExceedingSequenceLength) {
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLVisionLengthFitsSequence(8, 8, 8, 5, 100);
  });
  EXPECT_NE(message.find("positions available in sequence"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, RejectsGridWithIncorrectElementCount) {
  const std::vector<int64_t> grid{1, 2};
  const std::string message = CaptureThrowMessage([&] {
    ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw");
  });
  EXPECT_NE(message.find("divisible by 3"), std::string::npos) << message;
}

TEST(Qwen2VLPositionInputsTest, AcceptsValidGridDimensions) {
  const std::vector<int64_t> grid{1, 8, 8, 2, 4, 4};
  EXPECT_NO_THROW(ValidateQwen2VLGridTensorValues(grid.data(), grid.size(), "image_grid_thw"));
  EXPECT_NO_THROW(ValidateQwen2VLVisionLengthFitsSequence(2, 2, 2, 10, 100));
}

}  // namespace Generators::test

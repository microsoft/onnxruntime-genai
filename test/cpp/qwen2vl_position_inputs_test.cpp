// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <algorithm>
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

TEST_F(StandardPositionInputsTest, CapturedMropeInputsStayStableAcrossDecodeAndRewind) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_5_text";
  model_->config_->model.pad_token_id = 0;
  params_->use_graph_capture = true;
  params_->max_graph_capture_length = 2;
  params_->search.max_length = 16;
  auto inputs = CreatePositionInputs();
  inputs->Add();
  auto tokens = model_->p_device_->Allocate<int32_t>(3);
  std::copy_n(std::array<int32_t, 3>{0, 10, 20}.data(), 3, tokens.CpuSpan().data());
  inputs->Update(tokens, 3, 3);

  auto* mask = state_->GetInput("attention_mask");
  ASSERT_EQ(mask->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{1, 16}));
  const void* mask_address = mask->GetTensorRawData();
  EXPECT_EQ(mask->GetTensorData<int64_t>()[0], 0);
  EXPECT_EQ(mask->GetTensorData<int64_t>()[2], 1);
  EXPECT_EQ(mask->GetTensorData<int64_t>()[3], 0);

  auto next_token = tokens.subspan(1, 1);
  inputs->Update(next_token, 4, 1);
  const void* position_address = state_->GetInput("position_ids")->GetTensorRawData();
  auto expect_inputs = [&](int sequence_length, int first_position, int valid_length) {
    auto* positions = state_->GetInput("position_ids");
    EXPECT_EQ(positions->GetTensorRawData(), position_address);
    EXPECT_EQ(positions->GetTensorTypeAndShapeInfo()->GetShape(),
              (std::vector<int64_t>{3, 1, sequence_length}));
    for (int dim = 0; dim < 3; ++dim) {
      for (int s = 0; s < sequence_length; ++s) {
        EXPECT_EQ(positions->GetTensorData<int64_t>()[dim * sequence_length + s], first_position + s);
      }
    }
    auto* current_mask = state_->GetInput("attention_mask");
    EXPECT_EQ(current_mask->GetTensorRawData(), mask_address);
    ASSERT_EQ(current_mask->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{1, 16}));
    for (int s = 0; s < 16; ++s) {
      EXPECT_EQ(current_mask->GetTensorData<int64_t>()[s], s > 0 && s < valid_length ? 1 : 0);
    }
  };
  expect_inputs(1, 2, 4);
  inputs->Update(tokens.subspan(1, 2), 6, 2);
  expect_inputs(2, 3, 6);
  inputs->Update(next_token, 7, 1);
  expect_inputs(1, 5, 7);
  inputs->RewindTo(4);
  inputs->Update(next_token, 5, 1);
  expect_inputs(1, 3, 5);

  inputs->RewindTo(0);
  inputs->Update(tokens, 3, 3);
  inputs->Update(next_token, 4, 1);
  expect_inputs(1, 2, 4);

  // A short new prompt can itself be captured; it must reuse the same allocation.
  inputs->RewindTo(0);
  inputs->Update(next_token, 1, 1);
  EXPECT_EQ(state_->GetInput("position_ids")->GetTensorRawData(), position_address);
  EXPECT_EQ(state_->GetInput("attention_mask")->GetTensorRawData(), mask_address);
  EXPECT_EQ(state_->GetInput("position_ids")->GetTensorData<int64_t>()[0], 0);
  EXPECT_EQ(state_->GetInput("attention_mask")->GetTensorData<int64_t>()[0], 1);
  EXPECT_EQ(state_->GetInput("attention_mask")->GetTensorData<int64_t>()[1], 0);
  EXPECT_THROW(inputs->RewindTo(17), std::runtime_error);
}

TEST_F(StandardPositionInputsTest, MropeFullRewindReinitializesPromptPositionsAndPadding) {
  Use3DPositionIds();
  model_->config_->model.type = "qwen3_5_text";
  model_->config_->model.pad_token_id = 0;
  params_->search.max_length = 16;
  for (bool capture : {false, true}) {
    SCOPED_TRACE(capture);
    params_->use_graph_capture = capture;
    state_ = std::make_unique<PositionInputsTestState>(*params_, *model_);
    auto inputs = CreatePositionInputs();
    inputs->Add();
    auto tokens = model_->p_device_->Allocate<int32_t>(3);
    std::fill_n(tokens.CpuSpan().data(), 3, 10);
    inputs->Update(tokens, 3, 3);
    inputs->Update(tokens.subspan(0, 1), 4, 1);
    inputs->RewindTo(0);
    tokens.CpuSpan()[0] = 0;
    inputs->Update(tokens, 3, 3);
    const auto* positions = state_->GetInput("position_ids")->GetTensorData<int64_t>();
    for (int dim = 0; dim < 3; ++dim) {
      EXPECT_EQ(positions[dim * 3], 0);
      EXPECT_EQ(positions[dim * 3 + 1], 0);
      EXPECT_EQ(positions[dim * 3 + 2], 1);
    }
    const auto* mask = state_->GetInput("attention_mask")->GetTensorData<int64_t>();
    EXPECT_EQ(mask[0], 0);
    EXPECT_EQ(mask[1], 1);
    EXPECT_EQ(mask[2], 1);
  }
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

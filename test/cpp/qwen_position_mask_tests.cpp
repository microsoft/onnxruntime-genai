// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <gtest/gtest.h>

#include "models/io/qwen_vl_position_inputs.h"
#include "telemetry_test_environment.h"

namespace Generators::test {
namespace {

struct PositionTestModel : Model {
  explicit PositionTestModel(const char* dtype) : Model{std::make_unique<Config>()} {
    config_->model.type = "qwen3_5_moe_text";
    config_->model.pad_token_id = 0;
    auto path = std::filesystem::path{QWEN_POSITION_TEST_MODEL_DIR} / (std::string{dtype} + ".onnx");
    env = OrtEnv::Create();
    session = OrtSession::Create(*env, path.c_str(), session_options_.get());
    session_info_.Add(*session);
  }
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const override { return nullptr; }
  std::unique_ptr<OrtEnv> env;
  std::unique_ptr<OrtSession> session;
};

struct PositionTestState : State {
  PositionTestState(const GeneratorParams& params, const Model& model) : State{params, model} {}
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override { return {}; }
};

template <typename T>
void CheckMaskAndPositions(bool capture, const char* model_type = "qwen3_5_moe_text") {
  auto model_owner = std::make_shared<PositionTestModel>(sizeof(T) == 4 ? "int32" : "int64");
  auto& model = *model_owner;
  model.config_->model.type = model_type;
  auto params = std::make_shared<GeneratorParams>(model);
  params->search.batch_size = 2;
  params->search.max_length = 8;
  params->search.past_present_share_buffer = capture;
  // Exercise static tensor management on CPU without requiring a CUDA device.
  params->use_graph_capture = capture;
  params->max_graph_capture_length = 2;
  PositionTestState state{*params, model};
  auto& cpu = *GetDeviceInterface(DeviceType::CPU);
  std::vector<int32_t> lengths(2), prompt{0, 5, 6, 7, 8, 9}, next{10, 11};
  auto inputs = CreateStandardPositionInputs(state, cpu.WrapMemory(std::span{lengths}), "attention_mask");
  inputs->Add();
  inputs->Update(cpu.WrapMemory(std::span{prompt}), 3, 3);
  auto mask = [&] { return state.GetInput("attention_mask"); };
  auto check = [&](std::vector<T> expected, int64_t width) {
    EXPECT_EQ(mask()->GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{2, width}));
    EXPECT_EQ(std::vector<T>(mask()->GetTensorData<T>(), mask()->GetTensorData<T>() + expected.size()), expected);
  };
  const void* mask_address = mask()->GetTensorRawData();
  if (capture)
    check({0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0}, 8);
  else
    check({0, 1, 1, 1, 1, 1}, 3);

  inputs->Update(cpu.WrapMemory(std::span{next}), 4, 1);
  if (capture) {
    check({0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0}, 8);
    EXPECT_EQ(mask_address, mask()->GetTensorRawData());
  } else {
    check({1, 1, 1, 1, 1, 1, 1, 1}, 4);
  }
  const void* position_address = state.GetInput("position_ids")->GetTensorRawData();
  inputs->Update(cpu.WrapMemory(std::span{next}), 6, 2);
  const auto* positions = state.GetInput("position_ids")->GetTensorData<T>();
  EXPECT_EQ(std::vector<T>(positions, positions + 12), (std::vector<T>{3, 4, 4, 5, 3, 4, 4, 5, 3, 4, 4, 5}));
  if (capture) {
    EXPECT_EQ(position_address, state.GetInput("position_ids")->GetTensorRawData());
    check({0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0}, 8);
    inputs->RewindTo(4);
    check({0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0}, 8);
    inputs->Update(cpu.WrapMemory(std::span{next}), 5, 1);
    EXPECT_EQ(position_address, state.GetInput("position_ids")->GetTensorRawData());
    check({0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0}, 8);
    EXPECT_THROW(inputs->RewindTo(9), std::runtime_error);
    EXPECT_THROW(inputs->Update(cpu.WrapMemory(std::span{next}), 9, 1), std::runtime_error);
  }
  inputs->RewindTo(0);
  inputs->Update(cpu.WrapMemory(std::span{prompt}), 3, 3);
  if (capture)
    check({0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0}, 8);
  else
    check({0, 1, 1, 1, 1, 1}, 3);
}

TEST(QwenPositionMask, StaticInt32) { CheckMaskAndPositions<int32_t>(true); }
TEST(QwenPositionMask, StaticInt64) { CheckMaskAndPositions<int64_t>(true); }
TEST(QwenPositionMask, DynamicInt32) { CheckMaskAndPositions<int32_t>(false); }
TEST(QwenPositionMask, DynamicInt64) { CheckMaskAndPositions<int64_t>(false); }
TEST(QwenPositionMask, DenseStaticInt32) { CheckMaskAndPositions<int32_t>(true, "qwen3_5_text"); }
TEST(QwenPositionMask, DenseStaticInt64) { CheckMaskAndPositions<int64_t>(true, "qwen3_5_text"); }
TEST(QwenPositionMask, DenseDynamicInt32) { CheckMaskAndPositions<int32_t>(false, "qwen3_5_text"); }
TEST(QwenPositionMask, DenseDynamicInt64) { CheckMaskAndPositions<int64_t>(false, "qwen3_5_text"); }

}  // namespace
}  // namespace Generators::test

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  Generators::Shutdown();
  return result;
}

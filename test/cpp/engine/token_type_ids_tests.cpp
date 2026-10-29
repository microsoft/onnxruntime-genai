// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "models/io/token_type_ids.h"
#include "test_utils.h"

namespace Generators::test {
namespace {

struct TokenTypeIdsTestState : State {
  using State::State;

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override {
    return {};
  }
};

class TokenTypeIdsLifecycleTest
    : public testing::TestWithParam<std::tuple<ONNXTensorElementDataType, int>> {
 protected:
  void SetUp() override {
    model_ = CreateModel(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    type_ = std::get<0>(GetParam());
    batch_size_ = std::get<1>(GetParam());
    // Borrow declared input metadata without running the decoder through these aliases.
    model_->config_->model.decoder.inputs.token_type_ids =
        type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ? "input_ids" : "attention_mask";
    ASSERT_EQ(model_->session_info_.GetInputDataType(model_->config_->model.decoder.inputs.token_type_ids), type_);
    params_ = std::make_shared<GeneratorParams>(*model_);
    params_->search.batch_size = batch_size_;
    params_->search.num_beams = 1;
    state_ = std::make_unique<TokenTypeIdsTestState>(*params_, *model_);
  }

  void ExpectTensor(int64_t sequence_length, const std::vector<int32_t>& expected) {
    ASSERT_EQ(state_->inputs_.size(), 1u);
    ASSERT_EQ(state_->input_names_.size(), 1u);
    EXPECT_STREQ(state_->input_names_[0], model_->config_->model.decoder.inputs.token_type_ids.c_str());
    const auto* tensor = state_->inputs_[0];
    ASSERT_NE(tensor, nullptr);
    const auto info = tensor->GetTensorTypeAndShapeInfo();
    ASSERT_EQ(info->GetElementType(), type_);
    ASSERT_EQ(info->GetShape(), (std::vector<int64_t>{batch_size_, sequence_length}));
    ASSERT_EQ(info->GetElementCount(), expected.size());
    if (expected.empty()) return;
    if (type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      const auto* data = tensor->GetTensorData<int32_t>();
      EXPECT_EQ(std::vector<int32_t>(data, data + expected.size()), expected);
    } else {
      const auto* data = tensor->GetTensorData<int64_t>();
      EXPECT_EQ(std::vector<int64_t>(data, data + expected.size()),
                std::vector<int64_t>(expected.begin(), expected.end()));
    }
  }

  uintptr_t BoundAddress() const {
    return reinterpret_cast<uintptr_t>(state_->inputs_[0]);
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<TokenTypeIdsTestState> state_;
  ONNXTensorElementDataType type_;
  int batch_size_;
};

TEST_P(TokenTypeIdsLifecycleTest, ReusesStableShapeAndRefreshesMarkers) {
  TokenTypeIds token_type_ids{*state_};
  token_type_ids.Add();
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(0, {}));
  const auto empty_address = BoundAddress();
  token_type_ids.Update(0);
  EXPECT_EQ(BoundAddress(), empty_address);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(0, {}));

  std::vector<int32_t> prompt;
  std::vector<int32_t> restaged;
  for (int batch = 0; batch < batch_size_; ++batch) {
    prompt.insert(prompt.end(), {0, 1, 1, 0});
    restaged.insert(restaged.end(), {1, 0, 0, 1});
  }
  token_type_ids.SetPending(prompt);
  token_type_ids.Update(4);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(4, prompt));
  const auto prompt_address = BoundAddress();
  token_type_ids.Update(4);
  EXPECT_EQ(BoundAddress(), prompt_address);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(4, std::vector<int32_t>(batch_size_ * 4, 0)));

  const auto cleared_address = BoundAddress();
  token_type_ids.SetPending(restaged);
  token_type_ids.Update(4);
  EXPECT_EQ(BoundAddress(), cleared_address);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(4, restaged));

  token_type_ids.Update(1);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(1, std::vector<int32_t>(batch_size_, 0)));
  const auto decode_address = BoundAddress();
  token_type_ids.SetPending(std::vector<int32_t>(batch_size_, 1));
  token_type_ids.Update(1);
  EXPECT_EQ(BoundAddress(), decode_address);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(1, std::vector<int32_t>(batch_size_, 1)));
  const auto staged_decode_address = BoundAddress();
  token_type_ids.Update(1);
  EXPECT_EQ(BoundAddress(), staged_decode_address);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(1, std::vector<int32_t>(batch_size_, 0)));

  token_type_ids.Update(3);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(3, std::vector<int32_t>(batch_size_ * 3, 0)));
  const auto resized_address = BoundAddress();
  token_type_ids.SetPending(prompt);
  EXPECT_THROW(token_type_ids.Update(3), std::runtime_error);
  token_type_ids.Update(3);
  EXPECT_EQ(BoundAddress(), resized_address);
  ASSERT_NO_FATAL_FAILURE(ExpectTensor(3, std::vector<int32_t>(batch_size_ * 3, 0)));
}

INSTANTIATE_TEST_SUITE_P(DecoderTypesAndBatches, TokenTypeIdsLifecycleTest,
                         testing::Combine(testing::Values(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                                                          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64),
                                          testing::Values(1, 2)));

TEST(PendingTokenTypeIdsTest, AppliesMarkersOnceThenLeavesTextZeros) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto token_type_ids = OrtValue::CreateTensor<int32_t>(allocator, std::array<int64_t, 2>{1, 4});
  std::vector<int32_t> pending{0, 1, 1, 0};

  UpdatePendingTokenTypeIds(*token_type_ids, pending);
  EXPECT_TRUE(pending.empty());
  EXPECT_EQ(std::vector<int32_t>(token_type_ids->GetTensorData<int32_t>(),
                                 token_type_ids->GetTensorData<int32_t>() + 4),
            (std::vector<int32_t>{0, 1, 1, 0}));

  UpdatePendingTokenTypeIds(*token_type_ids, pending);
  EXPECT_EQ(std::vector<int32_t>(token_type_ids->GetTensorData<int32_t>(),
                                 token_type_ids->GetTensorData<int32_t>() + 4),
            (std::vector<int32_t>{0, 0, 0, 0}));
}

TEST(PendingTokenTypeIdsTest, ConsumesMarkersAndThrowsOnLengthMismatch) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto token_type_ids = OrtValue::CreateTensor<int32_t>(allocator, std::array<int64_t, 2>{1, 3});
  std::vector<int32_t> pending{0, 1, 1, 0};

  EXPECT_THROW(UpdatePendingTokenTypeIds(*token_type_ids, pending), std::runtime_error);
  EXPECT_TRUE(pending.empty());
}

TEST(PendingTokenTypeIdsTest, SupportsInt64DecoderInputs) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto token_type_ids = OrtValue::CreateTensor<int64_t>(allocator, std::array<int64_t, 2>{1, 2});
  std::vector<int32_t> pending{1, 0};

  UpdatePendingTokenTypeIds(*token_type_ids, pending);
  EXPECT_EQ(std::vector<int64_t>(token_type_ids->GetTensorData<int64_t>(),
                                 token_type_ids->GetTensorData<int64_t>() + 2),
            (std::vector<int64_t>{1, 0}));
}

TEST(TokenTypeIdsReadTest, SupportsInt32AndInt64Sources) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto int32_ids = OrtValue::CreateTensor<int32_t>(allocator, std::array<int64_t, 1>{3});
  auto int64_ids = OrtValue::CreateTensor<int64_t>(allocator, std::array<int64_t, 1>{3});
  const std::array<int32_t, 3> int32_values{0, 1, 0};
  const std::array<int64_t, 3> int64_values{0, 1, 0};
  std::copy(int32_values.begin(), int32_values.end(), int32_ids->GetTensorMutableData<int32_t>());
  std::copy(int64_values.begin(), int64_values.end(), int64_ids->GetTensorMutableData<int64_t>());

  EXPECT_EQ(ReadTokenTypeIds(*int32_ids), (std::vector<int32_t>{0, 1, 0}));
  EXPECT_EQ(ReadTokenTypeIds(*int64_ids), (std::vector<int32_t>{0, 1, 0}));
}

TEST(TokenTypeIdsReadTest, SupportsEmptySource) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto token_type_ids = OrtValue::CreateTensor<int32_t>(allocator, std::array<int64_t, 1>{0});

  EXPECT_TRUE(ReadTokenTypeIds(*token_type_ids).empty());
}

TEST(TokenTypeIdsReadTest, RejectsUnsupportedSourceTypes) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto uint8_ids = OrtValue::CreateTensor<uint8_t>(allocator, std::array<int64_t, 1>{1});
  auto float_ids = OrtValue::CreateTensor<float>(allocator, std::array<int64_t, 1>{1});

  EXPECT_THROW(ReadTokenTypeIds(*uint8_ids), std::runtime_error);
  EXPECT_THROW(ReadTokenTypeIds(*float_ids), std::runtime_error);
}

TEST(TokenTypeIdsReadTest, RejectsInt64ValuesOutsideInt32Range) {
  GetOrtEnv();
  auto& allocator = Ort::Allocator::GetWithDefaultOptions();
  auto token_type_ids = OrtValue::CreateTensor<int64_t>(allocator, std::array<int64_t, 1>{1});
  token_type_ids->GetTensorMutableData<int64_t>()[0] =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;

  EXPECT_THROW(ReadTokenTypeIds(*token_type_ids), std::runtime_error);
}

}  // namespace
}  // namespace Generators::test

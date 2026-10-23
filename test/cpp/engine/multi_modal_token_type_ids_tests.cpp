// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "models/multi_modal.h"

namespace Generators::test {
namespace {

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

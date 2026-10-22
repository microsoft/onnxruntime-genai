// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
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

}  // namespace
}  // namespace Generators::test

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "engine/decoders/varlen_decoder_io.h"
#include "engine/paged_key_value_cache.h"

namespace Generators {
namespace test {
namespace {

TEST(VarlenDecoderIOTest, PackedHybridPositionIdsAcceptTokenVectorOrMropeMatrix) {
  EXPECT_NO_THROW(ValidatePackedPositionIdsInput(
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      std::array<int64_t, 1>{-1},
      std::array<const char*, 1>{"num_tokens"}));
  EXPECT_NO_THROW(ValidatePackedPositionIdsInput(
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      std::array<int64_t, 2>{3, -1},
      std::array<const char*, 2>{nullptr, "num_tokens"}));
  EXPECT_THROW(ValidatePackedPositionIdsInput(
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                   std::array<int64_t, 1>{-1},
                   std::array<const char*, 1>{"batch_size"}),
               std::runtime_error);
  EXPECT_THROW(ValidatePackedPositionIdsInput(
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                   std::array<int64_t, 2>{2, -1}),
               std::runtime_error);
  EXPECT_THROW(ValidatePackedPositionIdsInput(
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                   std::array<int64_t, 2>{3, 1}),
               std::runtime_error);
  EXPECT_THROW(ValidatePackedPositionIdsInput(
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
                   std::array<int64_t, 1>{-1}),
               std::runtime_error);
  EXPECT_THROW(ValidatePackedPositionIdsInput(
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                   std::array<int64_t, 1>{1}),
               std::runtime_error);
  EXPECT_THROW(ValidatePackedPositionIdsInput(
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                   std::array<int64_t, 1>{0}),
               std::runtime_error);
}

TEST(VarlenDecoderIOTest, EagerMetadataUsesExactStepBounds) {
  StepPlan plan;
  RequestStepPlan first;
  first.unprocessed_token_count = 3;
  first.target_cache_slots = 259;
  RequestStepPlan second;
  second.unprocessed_token_count = 1;
  second.target_cache_slots = 513;
  plan.requests = {first, second};

  const auto metadata = GetAttentionMetadataForPlan(plan);

  EXPECT_EQ(metadata.max_query_len_bound, 3);
  EXPECT_EQ(metadata.max_kv_len_bound, 513);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 513);
}

TEST(VarlenDecoderIOTest, MinimumGraphBucketHasOnlyTrivialLowerBound) {
  const auto metadata = GetAttentionMetadataForGraph(/*block_table_columns=*/8, /*block_size=*/128);

  EXPECT_EQ(metadata.max_query_len_bound, 1);
  EXPECT_EQ(metadata.max_kv_len_bound, 1024);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 1);
}

TEST(VarlenDecoderIOTest, PowerOfTwoGraphBucketUsesPrecedingBoundary) {
  const auto metadata = GetAttentionMetadataForGraph(/*block_table_columns=*/16, /*block_size=*/128);

  EXPECT_EQ(metadata.max_kv_len_bound, 2048);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 1025);
}

TEST(VarlenDecoderIOTest, TruncatedFinalGraphBucketUsesPrecedingPowerOfTwoBoundary) {
  const auto metadata = GetAttentionMetadataForGraph(/*block_table_columns=*/100, /*block_size=*/128);

  EXPECT_EQ(metadata.max_kv_len_bound, 12800);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 8193);
}

TEST(VarlenDecoderIOTest, FirstTruncatedGraphBucketUsesMinimumBoundary) {
  const auto metadata = GetAttentionMetadataForGraph(/*block_table_columns=*/9, /*block_size=*/256);

  EXPECT_EQ(metadata.max_kv_len_bound, 2304);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 2049);
}

TEST(VarlenDecoderIOTest, GraphCapacityBelowMinimumBucketHasTrivialLowerBound) {
  const auto metadata = GetAttentionMetadataForGraph(/*block_table_columns=*/4, /*block_size=*/256);

  EXPECT_EQ(metadata.max_kv_len_bound, 1024);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 1);
}

TEST(VarlenDecoderIOTest, GraphBucketingAndMetadataBoundsStayConsistent) {
  constexpr size_t block_size = 256;
  constexpr size_t max_columns = 100;
  for (size_t blocks = 1; blocks <= max_columns; ++blocks) {
    const size_t columns = GetGraphBlockTableColumns(blocks, max_columns);
    const auto metadata = GetAttentionMetadataForGraph(columns, block_size);
    const size_t minimum_tokens_for_blocks = (blocks - 1) * block_size + 1;

    EXPECT_GE(static_cast<size_t>(metadata.max_kv_len_bound), minimum_tokens_for_blocks);
    EXPECT_LE(static_cast<size_t>(metadata.max_kv_len_lower_bound), minimum_tokens_for_blocks);
  }
}

TEST(VarlenDecoderIOTest, GraphStepFallsBackWhenReservationExceedsLiveKvLength) {
  StepPlan plan;
  RequestStepPlan request;
  request.unprocessed_token_count = 1;
  request.target_cache_slots = 513;
  plan.requests = {request};

  const auto metadata =
      GetAttentionMetadataForGraphStep(plan, /*block_table_columns=*/16, /*block_size=*/128);

  EXPECT_EQ(metadata.max_query_len_bound, 1);
  EXPECT_EQ(metadata.max_kv_len_bound, 2048);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 1);
}

TEST(VarlenDecoderIOTest, GraphStepRejectsInsufficientUpperBounds) {
  StepPlan plan;
  RequestStepPlan request;
  request.unprocessed_token_count = 2;
  request.target_cache_slots = 2049;
  plan.requests = {request};

  EXPECT_THROW(
      GetAttentionMetadataForGraphStep(plan, /*block_table_columns=*/16, /*block_size=*/128),
      std::runtime_error);
}

TEST(VarlenDecoderIOTest, PacksMetadataInOperatorContractOrder) {
  AttentionMetadataValues metadata;
  metadata.max_query_len_bound = 3;
  metadata.max_kv_len_bound = 513;
  metadata.max_kv_len_lower_bound = 257;
  const auto packed = PackAttentionMetadata(metadata);

  static_assert(packed.size() == kAttentionMetadataElementCount);
  EXPECT_EQ(packed[0], 3);
  EXPECT_EQ(packed[1], 513);
  EXPECT_EQ(packed[2], 257);
}

TEST(VarlenDecoderIOTest, RejectsZeroSizedGraphBounds) {
  EXPECT_THROW(GetAttentionMetadataForGraph(/*block_table_columns=*/0, /*block_size=*/128),
               std::runtime_error);
  EXPECT_THROW(GetAttentionMetadataForGraph(/*block_table_columns=*/8, /*block_size=*/0),
               std::runtime_error);
}

TEST(VarlenDecoderIOTest, RejectsZeroGraphBlockTableCapacity) {
  EXPECT_THROW(GetGraphBlockTableColumns(/*max_blocks=*/1, /*max_columns=*/0),
               std::runtime_error);
}

TEST(VarlenDecoderIOTest, RejectsGraphBoundsOutsideInt32Range) {
  const size_t overflowing_columns =
      static_cast<size_t>(std::numeric_limits<int32_t>::max()) / 128 + 1;

  EXPECT_THROW(GetAttentionMetadataForGraph(overflowing_columns, /*block_size=*/128),
               std::runtime_error);
}

}  // namespace
}  // namespace test
}  // namespace Generators

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
  const auto metadata = GetAttentionMetadataForGraph(/*max_query_len=*/1, /*block_table_columns=*/8, /*block_size=*/128);

  EXPECT_EQ(metadata.max_query_len_bound, 1);
  EXPECT_EQ(metadata.max_kv_len_bound, 1024);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 1);
}

TEST(VarlenDecoderIOTest, PowerOfTwoGraphBucketUsesPrecedingBoundary) {
  const auto metadata = GetAttentionMetadataForGraph(/*max_query_len=*/1, /*block_table_columns=*/16, /*block_size=*/128);

  EXPECT_EQ(metadata.max_kv_len_bound, 2048);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 1025);
}

TEST(VarlenDecoderIOTest, TruncatedFinalGraphBucketUsesPrecedingPowerOfTwoBoundary) {
  const auto metadata = GetAttentionMetadataForGraph(/*max_query_len=*/1, /*block_table_columns=*/100, /*block_size=*/128);

  EXPECT_EQ(metadata.max_kv_len_bound, 12800);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 8193);
}

TEST(VarlenDecoderIOTest, FirstTruncatedGraphBucketUsesMinimumBoundary) {
  const auto metadata = GetAttentionMetadataForGraph(/*max_query_len=*/1, /*block_table_columns=*/9, /*block_size=*/256);

  EXPECT_EQ(metadata.max_kv_len_bound, 2304);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 2049);
}

TEST(VarlenDecoderIOTest, GraphCapacityBelowMinimumBucketHasTrivialLowerBound) {
  const auto metadata = GetAttentionMetadataForGraph(/*max_query_len=*/1, /*block_table_columns=*/4, /*block_size=*/256);

  EXPECT_EQ(metadata.max_kv_len_bound, 1024);
  EXPECT_EQ(metadata.max_kv_len_lower_bound, 1);
}

TEST(VarlenDecoderIOTest, GraphBucketingAndMetadataBoundsStayConsistent) {
  constexpr size_t block_size = 256;
  constexpr size_t max_columns = 100;
  for (size_t blocks = 1; blocks <= max_columns; ++blocks) {
    const size_t columns = GetGraphBlockTableColumns(blocks, max_columns);
    const auto metadata = GetAttentionMetadataForGraph(/*max_query_len=*/1, columns, block_size);
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

TEST(VarlenDecoderIOTest, GraphBucketCarriesTheVerifiedBlockQueryLength) {
  const auto metadata =
      GetAttentionMetadataForGraph(/*max_query_len=*/8, /*block_table_columns=*/8, /*block_size=*/128);

  EXPECT_EQ(metadata.max_query_len_bound, 8);
  EXPECT_EQ(metadata.max_kv_len_bound, 1024);
}

TEST(VarlenDecoderIOTest, GraphStepTakesItsQueryBoundFromTheStep) {
  StepPlan plan;
  RequestStepPlan first;
  first.unprocessed_token_count = 4;
  first.target_cache_slots = 600;
  RequestStepPlan second;
  second.unprocessed_token_count = 4;
  second.target_cache_slots = 700;
  plan.requests = {first, second};

  const auto metadata =
      GetAttentionMetadataForGraphStep(plan, /*block_table_columns=*/16, /*block_size=*/128);

  EXPECT_EQ(metadata.max_query_len_bound, 4);
  EXPECT_EQ(metadata.max_kv_len_bound, 2048);
}

TEST(VarlenDecoderIOTest, GraphIdSeparatesEveryCapturedShape) {
  GraphAnnotationIds ids;
  const auto id = [&](size_t batch, size_t tokens, size_t columns, size_t binding) {
    const auto key = DecodeGraphKey(batch, tokens, columns, binding);
    if (!key) return -1;
    int last = -1;
    // A shape only earns an id once it has proved it recurs.
    for (size_t sighting = 0; sighting < GraphAnnotationIds::kSightingsBeforeCapture; ++sighting) {
      last = ids.Id(*key);
    }
    return last;
  };

  const int single = id(1, 1, 8, 0);
  const int block = id(1, 8, 8, 0);
  const int wider_batch = id(2, 8, 8, 0);
  const int wider_table = id(1, 8, 16, 0);
  // The persistent state bank flips on every commit, which moves the addresses the graph recorded.
  const int flipped_bank = id(1, 8, 8, 2);

  // A shape that shares a graph with a different shape would replay the wrong launch dimensions.
  EXPECT_NE(single, block);
  EXPECT_NE(single, wider_batch);
  EXPECT_NE(single, wider_table);
  EXPECT_NE(block, wider_batch);
  EXPECT_NE(block, wider_table);
  EXPECT_NE(wider_batch, wider_table);
  EXPECT_NE(block, flipped_bank);
  EXPECT_GT(single, 0);
  EXPECT_GT(flipped_bank, 0);
  EXPECT_EQ(ids.size(), 5u);
}

TEST(VarlenDecoderIOTest, GraphIdIsStableForARepeatedShape) {
  GraphAnnotationIds ids;
  const auto key = DecodeGraphKey(/*batch_size=*/2, /*tokens_per_request=*/8,
                                  /*block_table_columns=*/16, /*state_binding_key=*/3);
  ASSERT_TRUE(key.has_value());

  int assigned = -1;
  for (size_t sighting = 0; sighting < GraphAnnotationIds::kSightingsBeforeCapture; ++sighting) {
    assigned = ids.Id(*key);
  }

  // The EP only captures on the second occurrence of an id, so a shape must map to one id forever.
  ASSERT_GT(assigned, 0);
  EXPECT_EQ(ids.Id(*key), assigned);
  EXPECT_EQ(ids.Id(*key), assigned);
  EXPECT_EQ(ids.size(), 1u);
}

TEST(VarlenDecoderIOTest, GraphIdWithholdsAnIdUntilAShapeRecurs) {
  GraphAnnotationIds ids;
  const auto key = DecodeGraphKey(1, 1, 8, 0);
  ASSERT_TRUE(key.has_value());

  // Capturing costs a stalled step and permanent memory, so a shape seen once must not claim either.
  for (size_t sighting = 1; sighting < GraphAnnotationIds::kSightingsBeforeCapture; ++sighting) {
    EXPECT_EQ(ids.Id(*key), -1);
  }
  EXPECT_GT(ids.Id(*key), 0);
  EXPECT_EQ(ids.size(), 1u);
}

TEST(VarlenDecoderIOTest, GraphIdDoesNotSpendItsBudgetOnOneOffShapes) {
  GraphAnnotationIds ids;

  // A workload that walks through many distinct shapes once each must leave the budget intact for
  // the steady-state shape that follows it.
  for (size_t shape = 1; shape <= 4 * GraphAnnotationIds::kMaxCapturedShapes; ++shape) {
    EXPECT_EQ(ids.Id(*DecodeGraphKey(1, 1, 8, shape)), -1);
  }
  EXPECT_EQ(ids.size(), 0u);

  const auto steady = DecodeGraphKey(1, 8, 8, 0);
  int assigned = -1;
  for (size_t sighting = 0; sighting < GraphAnnotationIds::kSightingsBeforeCapture; ++sighting) {
    assigned = ids.Id(*steady);
  }
  EXPECT_GT(assigned, 0);
}

TEST(VarlenDecoderIOTest, GraphKeyBucketsNeighbouringBlockTableWidths) {
  const auto narrow = DecodeGraphKey(1, 1, /*block_table_columns=*/9, 0);
  const auto wide = DecodeGraphKey(1, 1, /*block_table_columns=*/16, 0);
  const auto next_bucket = DecodeGraphKey(1, 1, /*block_table_columns=*/17, 0);

  ASSERT_TRUE(narrow.has_value());
  EXPECT_EQ(*narrow, *wide);
  EXPECT_NE(*wide, *next_bucket);
}

TEST(VarlenDecoderIOTest, GraphKeyRejectsShapesItCannotCapture) {
  EXPECT_FALSE(DecodeGraphKey(/*batch_size=*/0, 1, 8, 0).has_value());
  EXPECT_FALSE(DecodeGraphKey(1, /*tokens_per_request=*/0, 8, 0).has_value());
  EXPECT_FALSE(DecodeGraphKey(1, 1, /*block_table_columns=*/0, 0).has_value());
}

TEST(VarlenDecoderIOTest, GraphIdStopsHandingOutIdsPastItsBudget) {
  GraphAnnotationIds ids;
  const auto claim = [&](const GraphAnnotationIds::Key& key) {
    int assigned = -1;
    for (size_t sighting = 0; sighting < GraphAnnotationIds::kSightingsBeforeCapture; ++sighting) {
      assigned = ids.Id(key);
    }
    return assigned;
  };

  for (size_t shape = 1; shape <= GraphAnnotationIds::kMaxCapturedShapes; ++shape) {
    EXPECT_GT(claim(*DecodeGraphKey(shape, 1, 8, 0)), 0);
  }

  // Past the budget a new shape runs eagerly instead of growing capture memory without bound.
  EXPECT_EQ(claim(*DecodeGraphKey(GraphAnnotationIds::kMaxCapturedShapes + 1, 1, 8, 0)), -1);
  EXPECT_EQ(ids.size(), GraphAnnotationIds::kMaxCapturedShapes);
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
  EXPECT_THROW(GetAttentionMetadataForGraph(/*max_query_len=*/1, /*block_table_columns=*/0, /*block_size=*/128),
               std::runtime_error);
  EXPECT_THROW(GetAttentionMetadataForGraph(/*max_query_len=*/1, /*block_table_columns=*/8, /*block_size=*/0),
               std::runtime_error);
  EXPECT_THROW(GetAttentionMetadataForGraph(/*max_query_len=*/0, /*block_table_columns=*/8, /*block_size=*/128),
               std::runtime_error);
}

TEST(VarlenDecoderIOTest, RejectsZeroGraphBlockTableCapacity) {
  EXPECT_THROW(GetGraphBlockTableColumns(/*max_blocks=*/1, /*max_columns=*/0),
               std::runtime_error);
}

TEST(VarlenDecoderIOTest, RejectsGraphBoundsOutsideInt32Range) {
  const size_t overflowing_columns =
      static_cast<size_t>(std::numeric_limits<int32_t>::max()) / 128 + 1;

  EXPECT_THROW(GetAttentionMetadataForGraph(/*max_query_len=*/1, overflowing_columns, /*block_size=*/128),
               std::runtime_error);
}

}  // namespace
}  // namespace test
}  // namespace Generators

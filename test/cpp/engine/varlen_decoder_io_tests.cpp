// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "engine/decoders/varlen_decoder_io.h"
#include "engine/paged_key_value_cache.h"
#include "engine/step_plan.h"
#include "engine_test_helpers.h"

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

TEST(VarlenDecoderIOTest, GraphStepFallsBackForMixedKvLengthsBelowTheBucketBoundary) {
  const AttentionMetadataValues exact_metadata{
      /*max_query_len_bound=*/4,
      /*max_kv_len_bound=*/1800,
      /*max_kv_len_lower_bound=*/700};

  const auto metadata = GetAttentionMetadataForGraphStep(
      exact_metadata, /*block_table_columns=*/16, /*block_size=*/128);

  EXPECT_EQ(metadata.max_query_len_bound, 4);
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

TEST(VarlenDecoderIOTest, SelectsOnlyConsumedLogitsRowsInRequestOrder) {
  StepPlan plan;
  RequestStepPlan verified;
  verified.packed_token_offset = 0;
  verified.logits_row_index = 3;
  verified.draft_token_count = 3;
  RequestStepPlan decode;
  decode.packed_token_offset = 4;
  decode.logits_row_index = 4;
  RequestStepPlan prefill;
  prefill.packed_token_offset = 5;
  prefill.logits_row_index = 14;
  plan.requests = {verified, decode, prefill};
  plan.token_count = 15;

  EXPECT_EQ(GetSelectedLogitsIndices(plan),
            (std::vector<size_t>{0, 1, 2, 3, 4, 14}));
}

TEST(VarlenDecoderIOTest, RejectsVerificationRowsOutsideTheRequestRange) {
  StepPlan plan;
  RequestStepPlan invalid;
  invalid.packed_token_offset = 4;
  invalid.logits_row_index = 5;
  invalid.draft_token_count = 2;
  plan.requests = {invalid};
  plan.token_count = 6;

  EXPECT_THROW(GetSelectedLogitsIndices(plan), std::runtime_error);
}

TEST(VarlenDecoderIOTest, RejectsSelectedRowsPastThePackedStep) {
  StepPlan plan;
  RequestStepPlan invalid;
  invalid.packed_token_offset = 0;
  invalid.logits_row_index = 2;
  plan.requests = {invalid};
  plan.token_count = 2;

  EXPECT_THROW(GetSelectedLogitsIndices(plan), std::runtime_error);
}

TEST(VarlenDecoderIOTest, SelectedLogitsModelGetsAPersistentIndicesBuffer) {
  auto selected = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedSelectedLogitsModel());
  auto per_token = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedPerTokenModel());
  ASSERT_TRUE(selected);
  ASSERT_TRUE(per_token);

  EXPECT_TRUE(DecoderLogitsArePerToken(*selected));
  EXPECT_TRUE(DecoderLogitsAreSelected(*selected));
  EXPECT_FALSE(DecoderLogitsAreSelected(*per_token));

  VarlenGraphBuffers selected_buffers{*selected, PackedPositionIdPlanes(*selected), kMaxDraftTokensPerStep + 1};
  VarlenGraphBuffers per_token_buffers{*per_token, PackedPositionIdPlanes(*per_token), kMaxDraftTokensPerStep + 1};
  ASSERT_NE(selected_buffers.logits_indices, nullptr);
  EXPECT_EQ(selected_buffers.logits_indices->GetShape(),
            (std::vector<int64_t>{static_cast<int64_t>(selected_buffers.max_token_rows)}));
  EXPECT_EQ(per_token_buffers.logits_indices, nullptr);
}

TEST(VarlenDecoderIOTest, RejectsLogitsIndicesOnAPerRequestLogitsModel) {
  auto model = LoadSyntheticPagedModel();
  // Any real session input makes the model look like it accepts selected rows.
  model->config_->model.decoder.inputs.logits_indices = "attention_metadata";

  EXPECT_THROW(DecoderLogitsAreSelected(*model), std::runtime_error);
  EXPECT_THROW(VarlenGraphBufferBytes(*model, PackedPositionIdPlanes(*model), 1), std::runtime_error);
}

TEST(VarlenDecoderIOTest, GraphBufferBytesIgnoreAWidthTheModelCannotUse) {
  // A model whose logits carry one row per request can never verify drafts, so a wider capture
  // window must not enlarge the buffers the engine has to reserve for it.
  auto model = LoadSyntheticPagedModel();
  const size_t planes = PackedPositionIdPlanes(*model);

  EXPECT_EQ(VarlenGraphBufferBytes(*model, planes, /*max_query_tokens_per_request=*/1),
            VarlenGraphBufferBytes(*model, planes, kMaxDraftTokensPerStep + 1));
}

TEST(VarlenDecoderIOTest, GraphBufferBytesGrowWithTheVerifiedBlock) {
  auto model = LoadSyntheticPagedPerTokenModel();
  const size_t planes = PackedPositionIdPlanes(*model);

  const size_t single_token = VarlenGraphBufferBytes(*model, planes, 1);
  const size_t verified_block = VarlenGraphBufferBytes(*model, planes, kMaxDraftTokensPerStep + 1);

  // The packed rows a verify step needs are what the engine has to take off the cache budget.
  EXPECT_GT(verified_block, single_token);
}

TEST(VarlenDecoderIOTest, GraphBuffersRejectStepsWiderThanTheyWereSizedFor) {
  auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedPerTokenModel());
  ASSERT_TRUE(model);
  const size_t max_batch_size = model->config_->engine.dynamic_batching->max_batch_size;
  VarlenGraphBuffers buffers{*model, PackedPositionIdPlanes(*model), kMaxDraftTokensPerStep + 1};

  EXPECT_TRUE(buffers.Fits(max_batch_size, 1));
  EXPECT_TRUE(buffers.Fits(1, kMaxDraftTokensPerStep + 1));
  // A step the buffers cannot hold runs eagerly rather than overflowing a static view mid-step.
  EXPECT_FALSE(buffers.Fits(max_batch_size + 1, 1));
  EXPECT_FALSE(buffers.Fits(1, kMaxDraftTokensPerStep + 2));
}

TEST(VarlenDecoderIOTest, GraphIdSeparatesEveryCapturedShape) {
  GraphAnnotationIds ids;
  const auto id = [&](size_t batch, size_t tokens, size_t columns, size_t binding) {
    const auto key = DecodeGraphKey(batch, tokens, columns, binding);
    return key ? ids.Id(*key) : -1;
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

  int assigned = ids.Id(*key);

  // The EP only captures on the second occurrence of an id, so a shape must map to one id forever.
  ASSERT_GT(assigned, 0);
  EXPECT_EQ(ids.Id(*key), assigned);
  EXPECT_EQ(ids.Id(*key), assigned);
  EXPECT_EQ(ids.size(), 1u);
}

TEST(VarlenDecoderIOTest, GraphKeyBucketsNeighbouringBlockTableWidths) {
  // Widths inside one power-of-two bucket map to one key. A single cache never offers both 9 and 16
  // columns, so this only ever merges a truncated final bucket with the boundary above it.
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

  for (size_t shape = 1; shape <= GraphAnnotationIds::kMaxCapturedShapes; ++shape) {
    EXPECT_GT(ids.Id(*DecodeGraphKey(shape, 1, 8, 0)), 0);
  }

  // Past the budget a new shape runs eagerly instead of growing capture memory without bound.
  EXPECT_EQ(ids.Id(*DecodeGraphKey(GraphAnnotationIds::kMaxCapturedShapes + 1, 1, 8, 0)), -1);
  EXPECT_EQ(ids.size(), GraphAnnotationIds::kMaxCapturedShapes);
}

TEST(VarlenDecoderIOTest, GraphIdsNeverCollideAcrossLiveAllocators) {
  // Two Engines may share one Model and therefore one session, and the session is what stores the
  // captured graphs. A shared id would replay one decoder's graph against the other's buffers.
  GraphAnnotationIds first;
  GraphAnnotationIds second;
  const auto key = DecodeGraphKey(/*batch_size=*/1, 1, 8, 0);
  ASSERT_TRUE(key.has_value());

  EXPECT_NE(first.Id(*key), second.Id(*key));
}

TEST(VarlenDecoderIOTest, GraphIdsAreNotReusedAfterAnAllocatorIsDestroyed) {
  // Destroying an Engine and building another over the same Model must not hand the new decoder an
  // id whose graph the session may still hold.
  const auto key = DecodeGraphKey(/*batch_size=*/2, 1, 8, 0);
  ASSERT_TRUE(key.has_value());
  int retired = 0;
  {
    GraphAnnotationIds ids;
    retired = ids.Id(*key);
  }
  ASSERT_GT(retired, 0);

  GraphAnnotationIds fresh;
  EXPECT_NE(fresh.Id(*key), retired);
}

TEST(VarlenDecoderIOTest, GraphIdsReportEveryAssignedIdForRelease) {
  GraphAnnotationIds ids;
  std::vector<int> handed_out;
  for (size_t batch = 1; batch <= 3; ++batch) {
    handed_out.push_back(ids.Id(*DecodeGraphKey(batch, 1, 8, 0)));
  }

  // The decoder releases these before freeing the buffers the graphs recorded, so missing one would
  // leave the session replaying against freed memory.
  auto assigned = ids.AssignedIds();
  std::sort(assigned.begin(), assigned.end());
  std::sort(handed_out.begin(), handed_out.end());
  EXPECT_EQ(assigned, handed_out);
}

TEST(VarlenDecoderIOTest, ClearingRetiredGraphIdsRestoresTheOwnerBudget) {
  GraphAnnotationIds ids;
  std::vector<int> retired;
  for (size_t shape = 1; shape <= GraphAnnotationIds::kMaxCapturedShapes; ++shape) {
    retired.push_back(ids.Id(*DecodeGraphKey(shape, 1, 8, 0)));
  }

  std::vector<int> visited;
  ids.ForEachAssignedId([&visited](int annotation_id) { visited.push_back(annotation_id); });
  std::sort(retired.begin(), retired.end());
  std::sort(visited.begin(), visited.end());
  EXPECT_EQ(visited, retired);

  ids.Clear();
  EXPECT_EQ(ids.size(), 0u);
  const int next = ids.Id(*DecodeGraphKey(1, 1, 8, 0));
  EXPECT_GT(next, 0);
  EXPECT_FALSE(std::binary_search(retired.begin(), retired.end(), next));
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

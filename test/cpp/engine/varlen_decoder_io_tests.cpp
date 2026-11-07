// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "engine/decoders/varlen_decoder_io.h"
#include "engine/decoders/simple_decoder.h"
#include "engine/paged_key_value_cache.h"
#include "engine/step_plan.h"
#include "engine_test_doubles.h"
#include "engine_test_helpers.h"
#include "models/decoder_only.h"
#include "models/io/rotary_position_history.h"
#include "models/preprocessing/genai_tokenizer.h"
#include "scoped-environment-variable.h"

namespace Generators {
namespace test {
namespace {

TEST(RotaryPositionHistoryTest, RetainsPromptCoordinatesAndForecastsGeneratedTokens) {
  RotaryPositionHistory history{1, 8};
  const std::array<int64_t, 9> prompt{0, 1, 1, 0, 2, 2, 0, 1, 4};
  history.Update(prompt, 0, 3);
  const auto values = history.Values();
  EXPECT_EQ(values[1], 1);
  EXPECT_EQ(values[8 + 1], 2);
  EXPECT_EQ(values[16 + 2], 4);
  for (size_t plane = 0; plane < 3; ++plane) {
    for (size_t token = 3; token < 8; ++token)
      EXPECT_EQ(values[plane * 8 + token], static_cast<int64_t>(token + 2));
  }
  const std::array<int64_t, 3> decode{5, 5, 5};
  history.Update(decode, 3, 1);
  EXPECT_EQ(values[16 + 2], 4);
  EXPECT_EQ(values[8 + 4], 6);
  history.Forecast(2);
  EXPECT_EQ(values[2], 3);
  EXPECT_EQ(values[8 + 2], 3);
  EXPECT_EQ(values[16 + 2], 3);
  history.Forecast(0);
  for (size_t plane = 0; plane < 3; ++plane)
    EXPECT_EQ(values[plane * 8 + 7], 7);
}

TEST(RotaryPositionHistoryTest, ExpandsEachPromptRowAcrossItsBeams) {
  RotaryPositionHistory history{4, 4};
  const std::array<int64_t, 12> prompt{0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6};
  history.Update(prompt, 0, 2, 2);
  for (size_t plane = 0; plane < 3; ++plane) {
    EXPECT_EQ(history.Values()[(plane * 4 + 0) * 4 + 1], prompt[plane * 4 + 1]);
    EXPECT_EQ(history.Values()[(plane * 4 + 1) * 4 + 1], prompt[plane * 4 + 1]);
    EXPECT_EQ(history.Values()[(plane * 4 + 2) * 4 + 1], prompt[plane * 4 + 3]);
    EXPECT_EQ(history.Values()[(plane * 4 + 3) * 4 + 1], prompt[plane * 4 + 3]);
  }
}

TEST(RotaryPositionHistoryTest, MtpCopiesBothRowsAndFutureCoordinatesWithoutAliasing) {
  RotaryPositionHistory main{2, 8};
  RotaryPositionHistory head{2, 6};
  const std::array<int64_t, 12> prompt{0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6};
  main.Update(prompt, 0, 2);
  head.CopyFrom(main.Values(), 8);
  for (size_t row = 0; row < 6; ++row)
    for (size_t token = 0; token < 6; ++token)
      EXPECT_EQ(head.Values()[row * 6 + token], main.Values()[row * 8 + token]);
  const auto old = head.Values()[0];
  main.Values()[0] = 42;
  EXPECT_EQ(head.Values()[0], old);
}

TEST(RotaryPositionHistoryTest, RejectsMalformedOrOutOfCapacityCoordinates) {
  EXPECT_THROW((RotaryPositionHistory{0, 4}), std::runtime_error);
  RotaryPositionHistory history{1, 4};
  EXPECT_THROW(history.Update(std::array<int64_t, 3>{0, 0, 0}, 4, 1), std::runtime_error);
  EXPECT_THROW(history.Update(std::array<int64_t, 3>{0, -1, 0}, 0, 1), std::runtime_error);
  EXPECT_THROW(history.Update(std::array<int64_t, 3>{0, 0, 0}, 0, 1, 2), std::runtime_error);
  EXPECT_THROW(history.CopyFrom(std::array<int64_t, 3>{0, 0, 0}, 1), std::runtime_error);
  EXPECT_THROW(history.Forecast(5), std::runtime_error);
}

TEST(RotaryPositionHistoryTest, PadsUnusedForecastRowsWithinTheRotaryTable) {
  RotaryPositionHistory history{1, 8, 7};
  history.Update(std::array<int64_t, 3>{4, 5, 6}, 0, 1);
  for (size_t plane = 0; plane < 3; ++plane)
    for (size_t token = 1; token < 8; ++token)
      EXPECT_EQ(history.Values()[plane * 8 + token], 7);
  EXPECT_THROW(history.Update(std::array<int64_t, 3>{8, 8, 8}, 1, 1), std::runtime_error);
}

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

TEST(VarlenDecoderIOTest, PackedHiddenStatesUseExportedInputWidth) {
  EXPECT_NO_THROW(ValidatePackedHiddenStatesInputShape(
      std::array<int64_t, 2>{3, 10240}, 3,
      std::array<int64_t, 2>{-1, 10240}));
  EXPECT_THROW(ValidatePackedHiddenStatesInputShape(
                   std::array<int64_t, 2>{3, 2560}, 3,
                   std::array<int64_t, 2>{-1, 10240}),
               std::runtime_error);
  EXPECT_THROW(ValidatePackedHiddenStatesInputShape(
                   std::array<int64_t, 2>{2, 10240}, 3,
                   std::array<int64_t, 2>{-1, 10240}),
               std::runtime_error);
}

TEST(VarlenDecoderIOTest, PacksThreePlanePositionsInScheduledRequestOrder) {
  auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedPerTokenModel());
  auto cache = std::shared_ptr<CacheManager>{CacheManager::Create(model)};
  auto engine = MakeDoublesEngine(model, 2, EosToken(*model)).engine;
  auto first = CreateRequestWithPrompt(engine, std::array<int32_t, 7>{2, 3, 4, 5, 6, 7, 8});
  auto second = CreateRequestWithPrompt(engine, std::array<int32_t, 5>{9, 8, 7, 6, 5});
  for (const auto& request : {first, second}) {
    request->Schedule();
    RequestStepPlan prefix;
    prefix.request = request;
    prefix.request_id = request.get();
    prefix.sequence_length_before = request->CurrentSequenceLength();
    prefix.unprocessed_token_count = request == first ? 5 : 1;
    prefix.target_cache_slots = prefix.unprocessed_token_count;
    PrepareRequestStep(model, prefix);
    request->CommitStep(prefix, {});
  }
  for (bool reversed : {true, false}) {
    StepPlan plan;
    const std::array<std::shared_ptr<Request>, 2> order =
        reversed ? std::array{second, first} : std::array{first, second};
    for (const auto& request : order) {
      RequestStepPlan entry;
      entry.request = request;
      entry.request_id = request.get();
      entry.sequence_length_before = request->CurrentSequenceLength();
      entry.unprocessed_token_count = request == first ? 2 : 3;
      entry.packed_token_offset = plan.token_count;
      entry.logits_row_index = plan.token_count + entry.unprocessed_token_count - 1;
      entry.target_cache_slots = request->ProcessedSequenceLength() + entry.unprocessed_token_count;
      plan.token_count += entry.unprocessed_token_count;
      plan.requests.push_back(entry);
    }
    ScheduledRequests scheduled{plan, model, nullptr, nullptr};
    ExecutionContext context{&plan};
    VarlenDecoderIO io{model, scheduled, cache, &context, nullptr, 3};
    const auto position = std::find_if(io.input_names_.begin(), io.input_names_.end(), [&](const char* name) {
      return model->config_->model.decoder.inputs.position_ids == name;
    });
    ASSERT_NE(position, io.input_names_.end());
    auto& tensor = *io.inputs_[position - io.input_names_.begin()];
    EXPECT_EQ(tensor.GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{3, 5}));
    const std::array<int64_t, 5> expected = reversed ? std::array<int64_t, 5>{1, 2, 3, 5, 6}
                                                     : std::array<int64_t, 5>{5, 6, 1, 2, 3};
    const auto* data = tensor.GetTensorData<int64_t>();
    for (size_t plane = 0; plane < 3; ++plane) {
      EXPECT_TRUE(std::equal(expected.begin(), expected.end(), data + plane * expected.size()));
    }
    VarlenGraphBuffers buffers{*model, 3, 4};
    context.block_table_columns = 8;
    context.graph_query_len_bound = 4;
    VarlenDecoderIO captured_io{model, scheduled, cache, &context, &buffers, 3};
    EXPECT_EQ(buffers.input_ids->GetShape(), (std::vector<int64_t>{5}));
    EXPECT_EQ(buffers.cumulative_sequence_lengths->GetShape(), (std::vector<int64_t>{3}));
    EXPECT_EQ(buffers.logits->GetShape()[0], 5);
    const auto cumulative = buffers.cumulative_sequence_lengths->GetDeviceSpan<int32_t>().CopyDeviceToCpu();
    EXPECT_EQ(cumulative[0], 0);
    EXPECT_EQ(cumulative[1], reversed ? 3 : 2);
    EXPECT_EQ(cumulative[2], 5);
    const auto attention = std::find_if(
        captured_io.input_names_.begin(), captured_io.input_names_.end(),
        [&](const char* name) { return model->config_->model.decoder.inputs.attention_metadata == name; });
    ASSERT_NE(attention, captured_io.input_names_.end());
    EXPECT_EQ(captured_io.inputs_[attention - captured_io.input_names_.begin()]->GetTensorData<int32_t>()[0], 4);
  }
}

TEST(VarlenDecoderIOTest, RotaryHistoryBoundsMediaCoordinatesAndOverwritesUnusedCaptureRows) {
  auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedPerTokenModel());
  auto session = model->CreateSession(GetOrtEnv(), "../synthetic-rotary-history/positions.onnx",
                                      model->session_options_.get());
  model->session_info_.Add(*session);
  model->config_->model.decoder.inputs.rotary_position_ids = "rotary_position_ids";
  model->config_->model.context_length = 32;
  auto cache = std::shared_ptr<CacheManager>{CacheManager::Create(model)};
  auto engine = MakeDoublesEngine(model, 2, EosToken(*model)).engine;
  auto first = std::make_shared<MediaItem>();
  first->grid = {1, 4, 6};
  first->rows = 6;
  auto second = std::make_shared<MediaItem>();
  second->grid = {1, 4, 4};
  second->rows = 4;
  const std::array<MediaSpan, 2> media{{{1, first}, {8, second}}};
  const std::array<std::array<int64_t, 11>, 3> expected{{
      {0, 1, 1, 1, 1, 1, 1, 4, 5, 5, 5},
      {0, 1, 1, 1, 2, 2, 2, 4, 5, 5, 6},
      {0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 5},
  }};
  for (const bool persistent : {false, true}) {
    auto buffers = persistent ? std::make_unique<VarlenGraphBuffers>(*model, 0, 3) : nullptr;
    for (const size_t past : {8u, 0u}) {
      auto request = CreateRequestWithPrompt(engine, std::array<int32_t, 13>{2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4});
      request->ReserveMedia(2);
      request->AppendMedia(media);
      if (past != 0) {
        request->StagePrefixAdoption(past);
        request->CommitPrefixAdoption();
      }
      StepPlan plan;
      RequestStepPlan entry;
      entry.request = request;
      entry.request_id = request.get();
      entry.sequence_length_before = request->CurrentSequenceLength();
      entry.unprocessed_token_count = 3;
      entry.target_cache_slots = past + 3;
      entry.logits_row_index = 2;
      plan.token_count = 3;
      plan.requests.push_back(entry);
      ScheduledRequests scheduled{plan, model, nullptr, nullptr};
      ExecutionContext context{&plan};
      context.block_table_columns = 4;
      VarlenDecoderIO io{model, scheduled, cache, &context, buffers.get()};
      const auto input = std::find_if(io.input_names_.begin(), io.input_names_.end(),
                                      [](const char* name) { return std::string_view{name} == "rotary_position_ids"; });
      ASSERT_NE(input, io.input_names_.end());
      const auto& tensor = *io.inputs_[input - io.input_names_.begin()];
      EXPECT_EQ(tensor.GetTensorTypeAndShapeInfo()->GetShape(), (std::vector<int64_t>{3, 1, 32}));
      const auto* values = tensor.GetTensorData<int64_t>();
      for (size_t plane = 0; plane < 3; ++plane)
        for (size_t row = 0; row < 32; ++row)
          EXPECT_EQ(values[plane * 32 + row], row < past + 3 ? expected[plane][row] : static_cast<int64_t>(row));
      request->Close();
    }
  }
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

TEST(VarlenDecoderIOTest, VariableWidthCaptureUsesExactPackedRowsAndBucketedQueryBound) {
  auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedPerTokenModel());
  ASSERT_TRUE(model);
  model->config_->engine.dynamic_batching->max_batch_size = 2;
  model->config_->engine.dynamic_batching->max_scheduled_tokens = 5;
  VarlenGraphBuffers buffers{*model, PackedPositionIdPlanes(*model), 4};
  // A 1+4 token batch fits even though padding both requests to width four would not.
  EXPECT_TRUE(buffers.FitsPacked(2, 4, 5));
  EXPECT_FALSE(buffers.Fits(2, 4));
  EXPECT_FALSE(buffers.FitsPacked(2, 4, 6));
  EXPECT_FALSE(buffers.FitsPacked(2, 4, 1));
  EXPECT_EQ(GetGraphQueryTokenBound(3, 7), 4u);
  EXPECT_EQ(GetGraphQueryTokenBound(5, 7), 7u);
  EXPECT_EQ(GetGraphQueryTokenBound(0, 7), 0u);
  EXPECT_EQ(GetGraphQueryTokenBound(8, 7), 0u);
}

TEST(VarlenDecoderIOTest, VariableWidthGraphKeysSeparatePackedGeometryAndQueryBounds) {
  GraphAnnotationIds ids;
  const auto key = [&](size_t query_bound, size_t tokens) {
    return *DecodeGraphKey(2, query_bound, 8, 0, 0, nullptr, false, 0, tokens);
  };
  const int mixed = ids.Id(key(4, 5));
  EXPECT_EQ(ids.Id(key(4, 5)), mixed);
  EXPECT_NE(ids.Id(key(4, 6)), mixed);
  EXPECT_NE(ids.Id(key(8, 5)), mixed);
  EXPECT_NE(ids.Id(*DecodeGraphKey(2, 4, 8, 0)), mixed);
}

TEST(VarlenGraphReplayTest, ChangingWidthsPreserveTokensAndCaptureGeometry) {
  const char* model_path = std::getenv("GENAI_VARLEN_GRAPH_MODEL");
  if (!model_path || !*model_path)
    GTEST_SKIP() << "Set GENAI_VARLEN_GRAPH_MODEL to a compatible CUDA paged model.";
  const std::string model_directory{model_path};

  struct ReplayStep {
    std::vector<size_t> widths;
    GraphAnnotationIds::Key key;
  };
  struct ObservingExecutor : ModelExecutor {
    ObservingExecutor(std::unique_ptr<ModelExecutor> executor, const OrtSession* session,
                      std::vector<ReplayStep>& steps)
        : executor_{std::move(executor)}, session_{session}, steps_{steps} {}

    bool SupportsDraftVerification() const override {
      return executor_->SupportsDraftVerification();
    }

    void Decode(ScheduledRequests& requests, ExecutionContext& context) override {
      executor_->Decode(requests, context);
      if (context.graph_query_len_bound == 0 || !context.plan ||
          context.plan->requests.size() != 2 || context.plan->token_count != 6)
        return;
      EXPECT_EQ(context.graph_query_len_bound, 4u);
      const auto key = SimpleDecoder::GraphKeyForContext(
          2, context.graph_query_len_bound, context, session_);
      ASSERT_TRUE(key);
      ReplayStep step{{}, *key};
      for (const auto& entry : context.plan->requests)
        step.widths.push_back(entry.unprocessed_token_count);
      steps_.push_back(std::move(step));
    }

    std::unique_ptr<ModelExecutor> executor_;
    const OrtSession* session_;
    std::vector<ReplayStep>& steps_;
  };

  using Tokens = std::array<std::vector<int32_t>, 2>;
  std::vector<ReplayStep> steps;
  const auto run = [&](bool enabled, const Tokens* oracle) {
    ScopedEnvironmentVariable flag{"ORTGENAI_ENGINE_VARLEN_GRAPH", enabled ? "1" : "0"};
    auto config = CreateConfig(GetOrtEnv(), model_directory.c_str());
    config->speculative.max_draft_tokens = 3;
    if (!config->engine.dynamic_batching)
      throw std::runtime_error("Replay test requires an ordinary paged Engine model.");
    config->engine.dynamic_batching->max_batch_size = 2;
    config->engine.dynamic_batching->max_scheduled_tokens = 8;
    config->engine.dynamic_batching->num_blocks = 64;
    config->engine.dynamic_batching->prefix_caching = false;
    SetProviderOption(*config, "cuda", "enable_cuda_graph", "1");
    auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(
        CreateModel(GetOrtEnv(), std::move(config)));
    if (!model)
      throw std::runtime_error("Replay test requires a decoder-only model.");
    if (model->p_device_->GetType() != DeviceType::CUDA)
      throw std::runtime_error("Replay test requires CUDA execution.");
    auto dependencies = Engine::CreateDependencies(model);
    dependencies.model_executor = std::make_unique<ObservingExecutor>(
        std::move(dependencies.model_executor), model->session_decoder_.get(), steps);
    auto engine = std::make_shared<Engine>(model, std::move(dependencies));
    Tokenizer tokenizer{*model->config_};
    const std::array<const char*, 2> prompts{
        "Write a Python function that merges sorted integer lists without modifying either input.",
        "Explain binary search and its time complexity."};
    std::array<std::shared_ptr<Request>, 2> requests;
    TurnOptions turn;
    turn.do_sample = false;
    turn.min_generated_tokens = 64;
    turn.max_generated_tokens = 64;
    for (size_t row = 0; row < requests.size(); ++row)
      requests[row] = CreateRequestWithPrompt(engine, tokenizer.Encode(prompts[row]), turn);
    Tokens tokens;
    size_t draft_round = 0;
    while (engine->HasPendingRequests()) {
      if (oracle && requests[0]->status_ == RequestStatus::Active &&
          requests[1]->status_ == RequestStatus::Active &&
          !requests[0]->IsPrefill() && !requests[1]->IsPrefill()) {
        const std::array<size_t, 2> widths = draft_round++ % 2 == 0
                                                 ? std::array<size_t, 2>{2, 4}
                                                 : std::array<size_t, 2>{3, 3};
        for (size_t row = 0; row < requests.size(); ++row) {
          const size_t offset = tokens[row].size();
          if (offset >= (*oracle)[row].size())
            throw std::runtime_error("Replay test exhausted its greedy reference.");
          const size_t count = std::min(widths[row] - 1, (*oracle)[row].size() - offset - 1);
          if (count != 0)
            requests[row]->SetDraftTokens(std::span<const int32_t>{(*oracle)[row]}.subspan(offset, count));
        }
      }
      std::array<EngineEvent, 8> events;
      const size_t count = engine->Run(events);
      for (size_t index = 0; index < count; ++index) {
        const auto& event = events[index];
        if (event.flags & EngineEventFlagFailed)
          throw std::runtime_error("Replay test request failed.");
        if (event.flags & EngineEventFlagToken) {
          const auto request = std::find(requests.begin(), requests.end(), event.request);
          if (request == requests.end())
            throw std::runtime_error("Replay test received an unknown request.");
          tokens[request - requests.begin()].push_back(event.token);
        }
      }
    }
    for (const auto& request : requests)
      engine->CloseRequest(request);
    return tokens;
  };
  const auto oracle = run(false, nullptr);
  for (const auto& tokens : oracle)
    ASSERT_EQ(tokens.size(), 64u);
  EXPECT_EQ(run(false, &oracle), oracle);
  steps.clear();
  EXPECT_EQ(run(true, &oracle), oracle);
  ASSERT_GE(steps.size(), 8u);
  size_t mixed_steps = 0;
  size_t uniform_steps = 0;
  for (const auto& step : steps) {
    EXPECT_EQ(step.key, steps.front().key);
    if (step.widths == std::vector<size_t>{2, 4})
      ++mixed_steps;
    else if (step.widths == std::vector<size_t>{3, 3})
      ++uniform_steps;
    else
      ADD_FAILURE() << "Unexpected captured query distribution.";
  }
  EXPECT_GE(mixed_steps, 3u);
  EXPECT_GE(uniform_steps, 3u);
}

TEST(VarlenDecoderIOTest, GraphBuffersRejectInvalidPackedRowCounts) {
  auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedPerTokenModel());
  ASSERT_TRUE(model);
  const size_t planes = PackedPositionIdPlanes(*model);
  const size_t width = kMaxDraftTokensPerStep + 1;
  for (const size_t batch : std::array<size_t, 3>{
           0, static_cast<size_t>(std::numeric_limits<int64_t>::max()), std::numeric_limits<size_t>::max()}) {
    model->config_->engine.dynamic_batching->max_batch_size = batch;
    EXPECT_THROW(VarlenGraphBufferBytes(*model, planes, width), std::runtime_error);
    EXPECT_THROW((VarlenGraphBuffers{*model, planes, width}), std::runtime_error);
  }
  model->config_->engine.dynamic_batching->max_batch_size =
      static_cast<size_t>(std::numeric_limits<int64_t>::max()) / width;
  model->config_->engine.dynamic_batching->max_scheduled_tokens = 0;
  EXPECT_THROW(VarlenGraphBufferBytes(*model, 3, width), std::runtime_error);
  EXPECT_THROW((VarlenGraphBuffers{*model, 3, width}), std::runtime_error);
}

TEST(VarlenDecoderIOTest, GraphBuffersRejectPackedRowOverflowBeforeScheduledTokenCap) {
  auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(LoadSyntheticPagedPerTokenModel());
  ASSERT_TRUE(model);
  const size_t planes = PackedPositionIdPlanes(*model);
  const size_t width = static_cast<size_t>(std::numeric_limits<int64_t>::max()) / 2 + 1;
  model->config_->engine.dynamic_batching->max_batch_size = 2;
  model->config_->engine.dynamic_batching->max_scheduled_tokens = 2;

  EXPECT_NO_THROW(VarlenGraphBufferBytes(*model, planes, width - 1));
  EXPECT_NO_THROW((VarlenGraphBuffers{*model, planes, width - 1}));
  EXPECT_THROW(VarlenGraphBufferBytes(*model, planes, width), std::runtime_error);
  EXPECT_THROW((VarlenGraphBuffers{*model, planes, width}), std::runtime_error);
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

TEST(VarlenDecoderIOTest, GraphIdSeparatesMtpScratchBanksFromCommittedState) {
  GraphAnnotationIds ids;
  const auto committed = DecodeGraphKey(1, 1, 8, 3);
  const auto first_bank = DecodeGraphKey(1, 1, 8, 3, 1);
  const auto second_bank = DecodeGraphKey(1, 1, 8, 3, 2);
  ASSERT_TRUE(committed.has_value());
  ASSERT_TRUE(first_bank.has_value());
  ASSERT_TRUE(second_bank.has_value());
  const int committed_id = ids.Id(*committed);
  const int first_id = ids.Id(*first_bank);
  const int second_id = ids.Id(*second_bank);
  EXPECT_NE(committed_id, first_id);
  EXPECT_NE(committed_id, second_id);
  EXPECT_NE(first_id, second_id);
  EXPECT_EQ(ids.Id(*DecodeGraphKey(1, 1, 8, 3, 1)), first_id);
  EXPECT_EQ(ids.Id(*DecodeGraphKey(1, 1, 8, 3, 2)), second_id);
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

TEST(VarlenDecoderIOTest, PacksSelectedOnlySparseMetadataInOperatorContractOrder) {
  AttentionMetadataValues metadata;
  metadata.max_query_len_bound = 3;
  metadata.max_kv_len_bound = 513;
  metadata.max_kv_len_lower_bound = 257;
  const auto packed = PackSparseAttentionMetadata(metadata);

  static_assert(packed.size() == kSparseAttentionMetadataElementCount);
  EXPECT_EQ(packed[0], 3);
  EXPECT_EQ(packed[1], 0);
  EXPECT_EQ(packed[2], 513);
  EXPECT_EQ(packed[3], 0);
  EXPECT_EQ(packed[4], 513);
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

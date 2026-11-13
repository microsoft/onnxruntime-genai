// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <gtest/gtest.h>
#include <numeric>

#include "engine_test_helpers.h"
#include "models/io/recurrent_state.h"
#include "models/io/indexer_cache.h"
#include "models/io/ple_state.h"

namespace Generators::test {
namespace {

struct RecurrentReviewState : State {
  using State::State;
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override { return {}; }
};

class RecurrentStateReview : public testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadDummyDecoderModel();
    auto& decoder = model_->config_->model.decoder;
    decoder.inputs.past_conv_names = "past.%d.conv";
    decoder.inputs.past_recurrent_names = "past.%d.recurrent";
    decoder.outputs.present_conv_names = "present.%d.conv";
    decoder.outputs.present_recurrent_names = "present.%d.recurrent";
    params_ = CreateGeneratorParams(*model_);
  }

  void UseSession(const char* filename, bool compact = false, bool keep_inputs = false) {
    session_ = model_->CreateSession(GetOrtEnv(),
                                     std::string{"../synthetic-rotary-history/"} + filename, model_->session_options_.get());
    if (!keep_inputs) model_->session_info_ = SessionInfo{};
    model_->session_info_.Add(*session_);
    if (compact) {
      model_->config_->model.decoder.state_update_capacity = 3;
      model_->config_->model.decoder.outputs.state_update_recurrent_capsule_names = "state_update.%d.recurrent_capsule";
    }
    state_ = std::make_unique<RecurrentReviewState>(*params_, *model_);
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<OrtSession> session_;
  std::unique_ptr<RecurrentReviewState> state_;
};

TEST_F(RecurrentStateReview, LegacyRankFiveWindowsCropBothConvolutionAndRecurrentState) {
  UseSession("windowed.onnx");
  auto recurrent = CreateRecurrentState(*state_, false);
  ASSERT_NE(recurrent, nullptr);
  ASSERT_TRUE(recurrent->IsWindowed());
  EXPECT_FALSE(recurrent->CropRequiresSnapshot());
  EXPECT_EQ(recurrent->StateWindow(), 4);
  recurrent->Add();
  auto* conv = state_->outputs_[0]->GetTensorMutableData<float>();
  auto* state = state_->outputs_[1]->GetTensorMutableData<float>();
  std::iota(conv, conv + 8, 0.0f);
  std::iota(state, state + 16, 100.0f);
  recurrent->SetForwardLength(3);
  recurrent->CropToPosition(1);
  EXPECT_EQ(conv[6], 4.0f);
  EXPECT_EQ(conv[7], 5.0f);
  for (size_t index = 0; index < 4; ++index)
    EXPECT_EQ(state[12 + index], 108.0f + static_cast<float>(index));
  recurrent->SetForwardLength(8);
  EXPECT_THROW(recurrent->CropToPosition(0), std::runtime_error);
  EXPECT_EQ(conv[6], 4.0f);
  EXPECT_EQ(state[12], 108.0f);
}

TEST_F(RecurrentStateReview, RejectsMismatchedLegacyWindowWidths) {
  UseSession("mismatched.onnx");
  EXPECT_THROW(CreateRecurrentState(*state_, false), std::runtime_error);
}

TEST_F(RecurrentStateReview, PleResetRestoresCanonicalBufferOrderAndVariant) {
  UseSession("ple.onnx");
  auto& decoder = model_->config_->model.decoder;
  decoder.ple_token_pad_id = 7;
  decoder.inputs.past_ple_token_names = "past.%d.ple_tokens";
  decoder.inputs.past_ple_conv_names = "past.%d.ple_conv";
  decoder.outputs.present_ple_token_names = "present.%d.ple_tokens";
  decoder.outputs.present_ple_conv_names = "present.%d.ple_conv";
  auto ple = CreatePleState(*state_);
  ASSERT_NE(ple, nullptr);
  ple->Add();
  auto* input = state_->inputs_[0];
  auto* output = state_->outputs_[0];
  ple->Update();
  EXPECT_EQ(ple->GraphCaptureVariant(), 1);
  ple->RewindTo(0);
  EXPECT_EQ(ple->GraphCaptureVariant(), 0);
  EXPECT_EQ(state_->inputs_[0], input);
  EXPECT_EQ(state_->outputs_[0], output);
  ple->Update();
  EXPECT_EQ(ple->GraphCaptureVariant(), 1);
}

TEST_F(RecurrentStateReview, CompactCropRequiresSnapshotBeforeAnyConvolutionMutation) {
  UseSession("compact.onnx", true);
  auto recurrent = CreateRecurrentState(*state_, false);
  ASSERT_NE(recurrent, nullptr);
  ASSERT_TRUE(recurrent->IsWindowed());
  EXPECT_TRUE(recurrent->CropRequiresSnapshot());
  recurrent->Add();
  recurrent->SetForwardLength(2);
  auto* conv = state_->outputs_[0]->GetTensorMutableData<float>();
  std::iota(conv, conv + 8, 0.0f);
  EXPECT_THROW(recurrent->CropToPosition(0), std::runtime_error);
  EXPECT_EQ(conv[6], 6.0f);
  EXPECT_EQ(conv[7], 7.0f);
  EXPECT_FALSE(recurrent->StateUpdateCaptureEnabled());
  recurrent->Snapshot(0);
  EXPECT_TRUE(recurrent->StateUpdateCaptureEnabled());
  EXPECT_NO_THROW(recurrent->ValidateCropPosition(0));
  recurrent->RestoreSnapshot();
  EXPECT_FALSE(recurrent->StateUpdateCaptureEnabled());
}

TEST_F(RecurrentStateReview, IndexerRollbackUsesCompressionMetadataAndRejectsOtherPositions) {
  for (int explicit_ratio : {0, 8}) {
    UseSession("indexer.onnx", false, true);
    auto& decoder = model_->config_->model.decoder;
    decoder.inputs.past_indexer_names = "past.%d.indexer_key";
    decoder.outputs.present_indexer_names = "present.%d.indexer_key";
    decoder.indexer_compress_ratio = explicit_ratio;
    Config::Model::Decoder::StateGroup group;
    group.kind = Config::Model::Decoder::StateGroupKind::FixedIndexer;
    group.layer_ids = {0};
    group.state_update.emplace();
    group.state_update->compress_ratio = 8;
    decoder.state_groups = std::vector<Config::Model::Decoder::StateGroup>{group};
    auto indexer = CreateIndexerCache(*state_);
    ASSERT_NE(indexer, nullptr);
    EXPECT_TRUE(indexer->UsesHostSequenceLength());
    indexer->Add();
    auto* cache = state_->inputs_.front()->GetTensorMutableData<float>();
    std::iota(cache, cache + 64 * 4, 0.0f);
    indexer->Snapshot(7);
    cache[0] = 500;
    cache[15 * 4] = 600;
    indexer->RewindTo(7);
    EXPECT_EQ(cache[0], 0.0f);
    EXPECT_EQ(cache[15 * 4], 60.0f);
    EXPECT_THROW(indexer->RewindTo(6), std::runtime_error);
    EXPECT_EQ(cache[15 * 4], 60.0f);
    indexer->RewindTo(0);
    for (size_t entry = 0; entry < 64 * 4; ++entry)
      EXPECT_EQ(cache[entry], 0.0f);
    decoder.indexer_compress_ratio = 4;
    EXPECT_THROW(CreateIndexerCache(*state_), std::runtime_error);
  }
}

TEST_F(RecurrentStateReview, GrowingIndexerWithoutSequenceLengthKeepsSeparateBuffers) {
  params_->search.past_present_share_buffer = true;
  UseSession("indexer-growing.onnx", false, true);
  auto& decoder = model_->config_->model.decoder;
  decoder.inputs.past_indexer_names = "past.%d.indexer_key";
  decoder.outputs.present_indexer_names = "present.%d.indexer_key";
  auto indexer = CreateIndexerCache(*state_);
  ASSERT_NE(indexer, nullptr);
  EXPECT_FALSE(indexer->UsesHostSequenceLength());
  const size_t output_index = state_->outputs_.size();
  indexer->Add();
  ASSERT_EQ(state_->outputs_.size(), output_index + 1);
  EXPECT_EQ(state_->outputs_[output_index], nullptr);
}

}  // namespace
}  // namespace Generators::test

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "generator/mtp_generator.h"
#include "models/model.h"
#include "search.h"
#include "tensor.h"

namespace Generators::test {
namespace {

// Metadata-only ONNX graph: Identity(hidden_states) -> main_hidden and a constant logits
// output, with input_ids declared. The recording State below supplies all forward results.
constexpr char kMetadataGraph[] =
    "\x08\x0a\x3a\x88\x02\x0a\x26\x0a\x0d\x68\x69\x64\x64\x65\x6e\x5f\x73\x74\x61\x74\x65\x73\x12\x0b"
    "\x6d\x61\x69\x6e\x5f\x68\x69\x64\x64\x65\x6e\x22\x08\x49\x64\x65\x6e\x74\x69\x74\x79\x0a\x1a\x0a"
    "\x06\x73\x63\x6f\x72\x65\x73\x12\x06\x6c\x6f\x67\x69\x74\x73\x22\x08\x49\x64\x65\x6e\x74\x69\x74"
    "\x79\x12\x08\x6d\x74\x70\x2d\x74\x65\x73\x74\x2a\x32\x08\x01\x08\x01\x08\x08\x10\x01\x22\x20\x00"
    "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    "\x00\x00\x00\x00\x00\x00\x00\x42\x06\x73\x63\x6f\x72\x65\x73\x5a\x1c\x0a\x09\x69\x6e\x70\x75\x74"
    "\x5f\x69\x64\x73\x12\x0f\x0a\x0d\x08\x06\x12\x09\x0a\x02\x08\x01\x0a\x03\x12\x01\x53\x5a\x24\x0a"
    "\x0d\x68\x69\x64\x64\x65\x6e\x5f\x73\x74\x61\x74\x65\x73\x12\x13\x0a\x11\x08\x01\x12\x0d\x0a\x02"
    "\x08\x01\x0a\x03\x12\x01\x53\x0a\x02\x08\x02\x62\x22\x0a\x0b\x6d\x61\x69\x6e\x5f\x68\x69\x64\x64"
    "\x65\x6e\x12\x13\x0a\x11\x08\x01\x12\x0d\x0a\x02\x08\x01\x0a\x03\x12\x01\x53\x0a\x02\x08\x02\x62"
    "\x1c\x0a\x06\x6c\x6f\x67\x69\x74\x73\x12\x12\x0a\x10\x08\x01\x12\x0c\x0a\x02\x08\x01\x0a\x02\x08"
    "\x01\x0a\x02\x08\x08\x42\x04\x0a\x00\x10\x15";

struct RecordingMtpState : State {
  RecordingMtpState(const GeneratorParams& params, const Model& model, bool head, bool crop, bool accept)
      : State(params, model), head_{head}, crop_{crop}, accept_{accept}, logits_{model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}, hidden_{model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT}, positions_{model.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64} {
    positions_.CreateTensor(std::array<int64_t, 3>{3, 1, 32});
    std::fill_n(positions_.GetMutableData<int64_t>(), 96, 42);
  }

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>& tokens, DeviceSpan<int32_t>) override {
    const int64_t count = static_cast<int64_t>(tokens.size());
    last_forward_start_ = length_;
    forwards.push_back(count);
    if (head_) {
      EXPECT_NE(rotary_history, nullptr);
      if (rotary_history)
        EXPECT_EQ(rotary_history->GetTensorData<int64_t>()[0], 42);
    }
    logits_.CreateTensor(std::array<int64_t, 3>{1, count, 8});
    hidden_.CreateTensor(std::array<int64_t, 3>{1, count, 2});
    auto* scores = logits_.GetMutableData<float>();
    std::fill_n(scores, count * 8, -100.0f);
    for (int64_t row = 0; row < count; ++row) {
      // Deliberately make wide verification disagree with single-token decode. The head
      // always proposes token 7, so every round rejects its first draft.
      const int token = head_ ? 7 : (length_ == 0 ? 1 : (count == 1 ? 4 : (accept_ ? 7 : 2)));
      scores[row * 8 + token] = 100.0f;
    }
    std::fill_n(hidden_.GetMutableData<float>(), count * 2, static_cast<float>(forwards.size()));
    length_ += static_cast<size_t>(count);
    return logits_.GetDeviceSpan<float>().subspan(static_cast<size_t>(count - 1) * 8, 8);
  }

  void SnapshotState(size_t position) override {
    ++snapshots;
    snapshot_position_ = position;
  }
  void ValidateRewindTo(size_t position) const override {
    if (unsupported_rewind_ && position == *unsupported_rewind_)
      throw std::runtime_error("unsupported recording-state rewind position");
  }
  void RewindTo(size_t position) override {
    if (position > 0) EXPECT_EQ(snapshot_position_, position);
    rewinds.push_back(position);
    length_ = position;
  }
  bool HasCroppableRecurrentState() const override { return crop_; }
  bool CropRequiresSnapshot() const override { return crop_requires_snapshot_; }
  int64_t RecurrentStateWindow() const override { return crop_ ? 4 : 1; }
  void CropToAccepted(size_t length, size_t) override {
    if (crop_requires_snapshot_) {
      EXPECT_GT(snapshots, 0);
      EXPECT_EQ(snapshot_position_, last_forward_start_);
    }
    ++crops;
    length_ = length;
  }
  void SetHiddenStates(OrtValue*) override {}
  OrtValue* GetRotaryPositionIds() override { return positions_.GetOrtTensor(); }
  void SetRotaryPositionIds(OrtValue* positions) override { rotary_history = positions; }
  OrtValue* GetOutput(const char* name) override {
    return std::strcmp(name, "logits") == 0 ? logits_.GetOrtTensor() : hidden_.GetOrtTensor();
  }

  bool head_;
  bool crop_;
  bool accept_;
  bool crop_requires_snapshot_{};
  Tensor logits_, hidden_, positions_;
  OrtValue* rotary_history{};
  size_t length_{};
  size_t last_forward_start_{};
  size_t snapshot_position_{};
  std::optional<size_t> unsupported_rewind_;
  int snapshots{};
  int crops{};
  std::vector<int64_t> forwards;
  std::vector<size_t> rewinds;
};

std::unique_ptr<Config> RecordingMtpConfig() {
  auto config = std::make_unique<Config>();
  config->model.type = "mtp-test";
  config->model.vocab_size = 8;
  config->model.context_length = 32;
  config->model.eos_token_id = {6};
  config->model.decoder.hidden_size = 2;
  config->model.decoder.inputs.hidden_states = "hidden_states";
  config->model.mtp.main_hidden_states = "main_hidden";
  config->model.mtp.outputs.hidden_states = "main_hidden";
  config->search.max_length = 16;
  config->search.chunk_size = 0;
  return config;
}

struct RecordingMtpModel : Model {
  RecordingMtpModel(bool head, bool crop, bool accept = false)
      : Model(RecordingMtpConfig()), head_{head}, crop_{crop}, accept_{accept} {
    auto session = OrtSession::Create(GetOrtEnv(), kMetadataGraph, sizeof(kMetadataGraph) - 1,
                                      session_options_.get());
    session_info_.Add(*session);
  }
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t>, const GeneratorParams& params) const override {
    auto state = std::make_unique<RecordingMtpState>(params, *this, head_, crop_, accept_);
    recording_state = state.get();
    return state;
  }

  bool head_, crop_, accept_;
  mutable RecordingMtpState* recording_state{};
};

TEST(MtpGeneratorReview, SingleRejectionReplaysDecodeEvenWithWindow) {
  for (bool crop : {false, true}) {
    SCOPED_TRACE(crop);
    auto main = std::make_shared<RecordingMtpModel>(false, crop);
    auto head = std::make_shared<RecordingMtpModel>(true, false);
    auto params = std::make_shared<GeneratorParams>(*main);
    params->speculative.max_draft_tokens = 1;
    MtpGenerator generator{*main, *head, *params};
    const std::array<int32_t, 1> prompt{0};
    generator.AppendTokens(prompt);
    generator.GenerateNextToken();

    EXPECT_EQ(main->recording_state->snapshots, 1);
    EXPECT_EQ(main->recording_state->crops, 0);
    EXPECT_EQ(main->recording_state->rewinds, (std::vector<size_t>{1}));
    EXPECT_EQ(main->recording_state->forwards, (std::vector<int64_t>{1, 2, 1}));
    EXPECT_EQ(generator.GetSpeculativeStats().target_forward_passes, 2);
    generator.GenerateNextToken();
    EXPECT_EQ(generator.GetSequence(), (std::vector<int32_t>{0, 1, 4}));
  }
}

TEST(MtpGeneratorReview, MultiRejectionPreservesFallbackSnapshotAndHeadRollback) {
  for (bool sampling : {false, true}) {
    SCOPED_TRACE(sampling);
    auto main = std::make_shared<RecordingMtpModel>(false, false);
    auto head = std::make_shared<RecordingMtpModel>(true, false);
    auto params = std::make_shared<GeneratorParams>(*main);
    params->speculative.max_draft_tokens = 2;
    params->search.do_sample = sampling;
    params->search.top_k = 1;
    params->search.random_seed = 123;
    MtpGenerator generator{*main, *head, *params};
    const std::array<int32_t, 1> prompt{0};
    generator.AppendTokens(prompt);
    generator.GenerateNextToken();

    EXPECT_EQ(main->recording_state->snapshots, 1);
    EXPECT_EQ(main->recording_state->forwards, (std::vector<int64_t>{1, 3, 1}));
    EXPECT_EQ(head->recording_state->snapshots, 1);
    // Greedy defers the head rewind to the following round; sampling performs it immediately.
    generator.GenerateNextToken();
    EXPECT_FALSE(head->recording_state->rewinds.empty());
    EXPECT_EQ(head->recording_state->rewinds.front(), 1u);
  }
}

TEST(MtpGeneratorReview, SingleAcceptanceKeepsBatchedHeadLookahead) {
  auto main = std::make_shared<RecordingMtpModel>(false, true, true);
  auto head = std::make_shared<RecordingMtpModel>(true, false);
  auto params = std::make_shared<GeneratorParams>(*main);
  params->speculative.max_draft_tokens = 1;
  MtpGenerator generator{*main, *head, *params};
  const std::array<int32_t, 1> prompt{0};
  generator.AppendTokens(prompt);
  generator.GenerateNextToken();

  EXPECT_EQ(main->recording_state->forwards, (std::vector<int64_t>{1, 2}));
  EXPECT_TRUE(main->recording_state->rewinds.empty());
  EXPECT_EQ(head->recording_state->forwards, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(head->recording_state->snapshots, 0);
  EXPECT_EQ(generator.GetSpeculativeStats().draft_tokens_accepted, 1u);
  generator.GenerateNextToken();
  EXPECT_EQ(generator.GetSequence(), (std::vector<int32_t>{0, 1, 7}));
}

TEST(MtpGeneratorReview, SampleSingleDoesNotSnapshotUnmodifiedHeadState) {
  auto main = std::make_shared<RecordingMtpModel>(false, false);
  auto head = std::make_shared<RecordingMtpModel>(true, false);
  auto params = std::make_shared<GeneratorParams>(*main);
  params->speculative.max_draft_tokens = 1;
  params->search.do_sample = true;
  params->search.top_k = 1;
  params->search.random_seed = 123;
  MtpGenerator generator{*main, *head, *params};
  const std::array<int32_t, 1> prompt{0};
  generator.AppendTokens(prompt);
  generator.GenerateNextToken();

  EXPECT_EQ(main->recording_state->snapshots, 1);
  EXPECT_EQ(head->recording_state->snapshots, 0);
  EXPECT_TRUE(head->recording_state->rewinds.empty());
  EXPECT_EQ(main->recording_state->forwards, (std::vector<int64_t>{1, 2, 1}));
}

TEST(MtpGeneratorReview, DirectCommitSnapshotsOnlyForCompactCropBaseline) {
  for (bool sampling : {false, true}) {
    for (bool compact : {false, true}) {
      for (bool accept : {false, true}) {
        SCOPED_TRACE(sampling);
        SCOPED_TRACE(compact);
        SCOPED_TRACE(accept);
        auto main = std::make_shared<RecordingMtpModel>(false, true, accept);
        auto head = std::make_shared<RecordingMtpModel>(true, false);
        auto params = std::make_shared<GeneratorParams>(*main);
        params->speculative.max_draft_tokens = 2;
        params->search.do_sample = sampling;
        params->search.top_k = 1;
        params->search.random_seed = 123;
        MtpGenerator generator{*main, *head, *params};
        main->recording_state->crop_requires_snapshot_ = compact;
        const std::array<int32_t, 1> prompt{0};
        generator.AppendTokens(prompt);
        generator.GenerateNextToken();

        EXPECT_EQ(main->recording_state->snapshots, compact ? 1 : 0);
        EXPECT_EQ(main->recording_state->crops, accept ? 0 : 1);
        EXPECT_TRUE(main->recording_state->rewinds.empty());
        EXPECT_EQ(main->recording_state->forwards, (std::vector<int64_t>{1, 3}));
        EXPECT_EQ(generator.GetSpeculativeStats().target_forward_passes, 1u);
      }
    }
  }
}

TEST(MtpGeneratorReview, UnsupportedRewindLeavesSearchAndStateUnchanged) {
  auto model = std::make_shared<RecordingMtpModel>(false, false);
  auto params = std::make_shared<GeneratorParams>(*model);
  auto generator = CreateGenerator(*model, *params);
  const std::array<int32_t, 1> first{0};
  generator->AppendTokens(first);
  generator->SnapshotState();
  const std::array<int32_t, 2> suffix{1, 2};
  generator->AppendTokens(suffix);
  auto& state = *model->recording_state;
  state.unsupported_rewind_ = 2;
  const auto sequence = [&] {
    auto tokens = generator->GetSequence(0).CopyDeviceToCpu();
    return std::vector<int32_t>(tokens.begin(), tokens.end());
  };

  ASSERT_EQ(sequence(), (std::vector<int32_t>{0, 1, 2}));
  EXPECT_THROW(generator->RewindToLength(2), std::runtime_error);
  EXPECT_EQ(sequence(), (std::vector<int32_t>{0, 1, 2}));
  EXPECT_EQ(generator->search_->GetSequenceLength(), 3u);
  EXPECT_EQ(state.length_, 3u);
  EXPECT_TRUE(state.rewinds.empty());
  EXPECT_EQ(state.forwards, (std::vector<int64_t>{1, 2}));
  EXPECT_NO_THROW(generator->GetLogits());
  EXPECT_EQ(state.forwards, (std::vector<int64_t>{1, 2}));

  EXPECT_NO_THROW(generator->RewindToLength(1));
  EXPECT_EQ(sequence(), (std::vector<int32_t>{0}));
  EXPECT_EQ(generator->search_->GetSequenceLength(), 1u);
  EXPECT_EQ(state.length_, 1u);
  EXPECT_EQ(state.rewinds, (std::vector<size_t>{1}));
  const std::array<int32_t, 1> continuation{3};
  EXPECT_NO_THROW(generator->AppendTokens(continuation));
  EXPECT_EQ(sequence(), (std::vector<int32_t>{0, 3}));
  EXPECT_EQ(state.length_, 2u);
}

}  // namespace
}  // namespace Generators::test

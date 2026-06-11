// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "search.h"
#include "constrained_logits_processor.h"
#include "models/model.h"
#include "mtp_generator.h"

namespace Generators {

namespace {
// Greedy argmax over a contiguous vocab row of fp32 logits on the CPU.
int32_t ArgmaxRow(const float* row, int vocab_size) {
  int32_t best = 0;
  float best_val = row[0];
  for (int i = 1; i < vocab_size; ++i) {
    if (row[i] > best_val) {
      best_val = row[i];
      best = i;
    }
  }
  return best;
}
}  // namespace

MtpGenerator::MtpGenerator(const Model& main_model, const Model& mtp_model, const GeneratorParams& params)
    : main_model_{main_model}, mtp_model_{mtp_model} {
  // MTP runs both a 1-token decode and a 2-token verify on the main model. Allow CUDA graph
  // capture of both shapes (each captured under its own annotation id with pre-sized static
  // buffers). Harmless for the MTP head, which only ever runs a single token per step.
  const_cast<GeneratorParams&>(params).max_graph_capture_length = 2;

  main_ = CreateGenerator(main_model_, params);
  mtp_ = CreateGenerator(mtp_model_, params);

  hidden_size_ = main_model_.config_->model.decoder.hidden_size;
  vocab_size_ = main_model_.config_->model.vocab_size;
  max_length_ = params.search.max_length;

  // Reusable [1, 1, hidden] device buffer for the on-device hidden-state handoff.
  hidden_slice_ = std::make_shared<Tensor>(
      main_model_.p_device_inputs_,
      main_model_.session_info_.GetOutputDataType(main_model_.config_->model.decoder.outputs.hidden_states));
  const std::array<int64_t, 3> slice_shape{1, 1, hidden_size_};
  hidden_slice_->CreateTensor(slice_shape);
}

void MtpGenerator::ExtractHiddenPosition(OrtValue* hidden, int position) {
  // hidden is [1, S, H] on the model device; copy row `position` into hidden_slice_ ([1,1,H]).
  auto src = ByteWrapTensor(*main_model_.p_device_, *hidden);
  const size_t row_bytes = hidden_slice_->GetByteSpan().size();
  auto src_row = src.subspan(static_cast<size_t>(position) * row_bytes, row_bytes);
  hidden_slice_->GetByteSpan().CopyFrom(src_row);
}

int32_t MtpGenerator::ArgmaxLogitsRow(int row) {
  // Cast the main model's raw logits output ([1, S, V], io dtype) to fp32 and argmax row `row`.
  OrtValue* raw = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.logits.c_str());
  Cast(*raw, logits_fp32_, *main_model_.p_device_, Ort::TypeToTensorType<float>);
  // Keep the wrapping span alive while we read its CPU copy (it owns the pinned host buffer).
  auto span = ByteWrapTensor(*main_model_.p_device_, *logits_fp32_);
  auto cpu = span.CopyDeviceToCpu();
  const float* data = reinterpret_cast<const float*>(cpu.data());
  return ArgmaxRow(data + static_cast<size_t>(row) * vocab_size_, vocab_size_);
}

int32_t MtpGenerator::DraftNextToken(OrtValue* /*unused*/, int32_t token) {
  // hidden_slice_ already holds the hidden state paired with `token`. Feed (hidden, token) to the
  // MTP head; its KV cache accumulates, so this is an O(1) incremental draft step. Returns the
  // MTP head's predicted next-next token (greedy argmax of its last-position logits).
  mtp_->SetHiddenStates(hidden_slice_);
  std::array<int32_t, 1> tok{token};
  mtp_->AppendTokens(cpu_span<const int32_t>(tok));
  auto logits_span = mtp_->GetLogits();              // fp32, last token, [1, V]
  auto logits = logits_span.CopyDeviceToCpu();
  return ArgmaxRow(logits.data(), vocab_size_);
}

void MtpGenerator::AppendTokens(cpu_span<const int32_t> input_ids) {
  main_->AppendTokens(input_ids);
  length_ = input_ids.size();
  for (auto t : input_ids) sequence_.push_back(t);

  OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
  const int last = static_cast<int>(input_ids.size()) - 1;
  ExtractHiddenPosition(hidden, last);             // h for the token we are about to predict
  next_token_ = ArgmaxLogitsRow(last);             // token predicted for position length_
  primed_ = true;
}

void MtpGenerator::GenerateNextToken() {
  if (!primed_) throw std::runtime_error("MtpGenerator: AppendTokens must be called before GenerateNextToken");
  if (done_) return;

  // Commit the token predicted for position length_.
  const int32_t t = next_token_;
  sequence_.push_back(t);
  if (contains(main_model_.config_->model.eos_token_id, t) || sequence_.size() >= static_cast<size_t>(max_length_)) {
    done_ = true;
    return;
  }

  // 1. Draft the next token with the MTP head (hidden_slice_ holds h paired with t).
  const int32_t d = DraftNextToken(nullptr, t);

  // 2. Snapshot the recurrent state at length L, then verify [t, d] in a single main forward.
  main_->SnapshotState();
  std::array<int32_t, 2> verify{t, d};
  main_->AppendTokens(cpu_span<const int32_t>(verify));
  ++forwards_;

  const int32_t m = ArgmaxLogitsRow(0);  // main model's real token after t
  ++trials_;

  if (d == m) {
    // 2a. Accept: t and d are both correct. Commit d and harvest the free prediction at row 1.
    ++accepts_;
    sequence_.push_back(d);
    if (sequence_.size() >= static_cast<size_t>(max_length_)) {
      done_ = true;
      return;
    }
    OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
    // Feed the accepted pair (hidden@L, d) to the MTP head so its KV stays aligned.
    ExtractHiddenPosition(hidden, 0);
    DraftNextToken(nullptr, d);
    // Next token to commit is argmax(logits@L+1); its hidden is row 1.
    next_token_ = ArgmaxLogitsRow(1);
    ExtractHiddenPosition(hidden, 1);
    length_ += 2;
  } else {
    // 2b. Reject: roll back the speculative forward (restore recurrent state + crop KV to L),
    //     then re-run the single correct token t.
    main_->RewindToLength(length_);
    std::array<int32_t, 1> rerun{t};
    main_->AppendTokens(cpu_span<const int32_t>(rerun));
    ++forwards_;
    OrtValue* hidden = main_->state_->GetOutput(main_model_.config_->model.decoder.outputs.hidden_states.c_str());
    next_token_ = ArgmaxLogitsRow(0);
    ExtractHiddenPosition(hidden, 0);
    length_ += 1;
  }
}

bool MtpGenerator::IsDone() const {
  return done_;
}

}  // namespace Generators

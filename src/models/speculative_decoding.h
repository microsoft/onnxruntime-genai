// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <vector>
#include "model.h"
#include "decoder_only.h"

namespace Generators {

struct Generator;

// Composes two DecoderOnly_Model instances each with its own cloned Config
// reused verbatim for session loading, KV cache, and logits I/O.
struct SpeculativeDecodingModel : Model {
  SpeculativeDecodingModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  const DecoderOnly_Model& target_model() const { return *target_model_; }
  const DecoderOnly_Model& draft_model() const { return *draft_model_; }

 private:
  std::shared_ptr<DecoderOnly_Model> target_model_;
  std::shared_ptr<DecoderOnly_Model> draft_model_;
};

// 2 inner States + the cross-round "pending draft probs"
// carry-over. The per-round propose -> verify -> accept loop lives in
// SpeculativeDecodingStrategy; Run() here only handles prefill.
struct SpeculativeDecodingState : State {
  SpeculativeDecodingState(const SpeculativeDecodingModel& model,
                           DeviceSpan<int32_t> sequence_lengths,
                           const GeneratorParams& params);

  // Prefill path: runs both inner states, saves draft pending probs, returns target logits.
  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  void RewindTo(size_t index) override;
  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;
  void SetActiveAdapter(Adapters* adapters, const std::string& adapter_name) override;
  void SetRunOption(const char* key, const char* value) override;
  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  // Accessors for the strategy layer.
  const SpeculativeDecodingModel& spec_model() const { return model_; }
  State& target_state() { return *target_state_; }
  State& draft_state() { return *draft_state_; }

  const std::vector<float>& draft_pending_logits() const { return draft_pending_logits_; }
  void set_draft_pending_logits(std::vector<float> logits) {
    draft_pending_logits_ = std::move(logits);
    draft_pending_valid_ = true;
  }
  // Copy count floats into the pending-logits buffer, reusing its existing capacity (no per-round
  // reallocation of a vocab-sized vector). Mirrors the assign() the prefill Run() uses.
  void assign_draft_pending_logits(const float* data, size_t count) {
    draft_pending_logits_.assign(data, data + count);
    draft_pending_valid_ = true;
  }
  bool draft_pending_valid() const { return draft_pending_valid_; }

 private:
  const SpeculativeDecodingModel& model_;
  std::unique_ptr<State> target_state_;
  std::unique_ptr<State> draft_state_;

  // Draft's raw logits for the next token position (unsoftmaxed, so the strategy can apply the
  // same min-length / repetition penalties the target rows receive before sampling).
  // Set by Run() and refreshed at the end of each strategy round.
  std::vector<float> draft_pending_logits_;
  bool draft_pending_valid_{false};
};

}  // namespace Generators

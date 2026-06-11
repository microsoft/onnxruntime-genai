// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <vector>

namespace Generators {

struct Model;
struct Generator;
struct GeneratorParams;
struct Tensor;

// In-engine Multi-Token-Prediction (MTP) self-speculative decoder for Qwen3.6-style models.
//
// It composes two genai generators on the shared compute stream:
//   * the main decoder (exported with include_hidden_states so it emits a hidden_states output)
//   * the MTP head (mtp.onnx, a single decoder layer that drafts the next-next token)
//
// The main model's last hidden state is handed to the MTP head device-to-device (no host
// round-trip), and the draft is verified against the main model in a single 2-token forward.
// Greedy, batch size 1. The output is identical to plain greedy decoding (lossless), modulo
// floating-point near-ties in the batched verify forward.
struct MtpGenerator {
  MtpGenerator(const Model& main_model, const Model& mtp_model, const GeneratorParams& params);

  // Seed the prompt (runs the main model's prefill).
  void AppendTokens(cpu_span<const int32_t> input_ids);

  // Produce the next token via the draft/verify loop. On an accepted draft this commits two
  // tokens (and harvests a free third prediction from the verify pass); on a rejected draft it
  // commits one and rolls back the speculative forward.
  void GenerateNextToken();

  bool IsDone() const;

  // The full committed token sequence (batch index 0).
  const std::vector<int32_t>& GetSequence() const { return sequence_; }

  // Speculative-decoding statistics.
  size_t Forwards() const { return forwards_; }
  size_t Accepts() const { return accepts_; }
  size_t Trials() const { return trials_; }

 private:
  // Run the MTP head on a single (hidden_state, token) pair and return its drafted next token.
  int32_t DraftNextToken(OrtValue* hidden_last_position, int32_t token);
  // Copy one [1,1,H] position out of a [1,S,H] hidden_states OrtValue into hidden_slice_ (D2D).
  void ExtractHiddenPosition(OrtValue* hidden, int position);
  // Greedy argmax over `num_rows` consecutive vocab rows of the main model's raw logits output
  // ([1,S,V]), starting at `first_row`, writing the token ids to `out`. Uses the device's
  // on-device Top-K kernel when available (no full-logits host copy); falls back to a host argmax.
  void ArgmaxMainRows(int first_row, int num_rows, int32_t* out);

  const Model& main_model_;
  const Model& mtp_model_;

  std::unique_ptr<Generator> main_;  // main decoder generator
  std::unique_ptr<Generator> mtp_;   // MTP head generator (drafts)

  std::shared_ptr<Tensor> hidden_slice_;  // reusable [1,1,hidden] device buffer for the handoff
  std::unique_ptr<OrtValue> logits_fp32_;  // reusable fp32 cast of the main model's raw logits

  std::vector<int32_t> sequence_;  // committed tokens (batch 0)
  int hidden_size_{};
  int vocab_size_{};
  int max_length_{};

  // Loop carry state (see the design doc draft/verify invariant):
  int32_t next_token_{};   // token predicted for the current cache length L (not yet committed)
  size_t length_{};        // committed cache length L
  bool primed_{false};     // whether AppendTokens has run the prompt
  bool done_{false};

  size_t forwards_{};
  size_t accepts_{};
  size_t trials_{};
};

}  // namespace Generators

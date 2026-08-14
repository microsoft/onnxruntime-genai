// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <deque>
#include <memory>
#include <random>
#include <vector>

#include "sampling_distribution.h"
#include "speculative_stats.h"

namespace Generators {

struct Model;
struct Generator;
struct GeneratorParams;
struct Tensor;
struct DeviceInterface;

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

  // Produce exactly one user-visible token. A speculative round may commit several tokens
  // internally; later calls drain those buffered tokens before another round runs.
  void GenerateNextToken();

  bool IsDone() const;

  // The full committed token sequence (batch index 0).
  const std::vector<int32_t>& GetSequence() const { return emitted_sequence_; }

  // Speculative-decoding statistics.
  SpeculativeStats GetSpeculativeStats() const;
  size_t Forwards() const { return stats_.target_forward_passes; }
  size_t Accepts() const { return stats_.draft_tokens_accepted; }
  size_t Trials() const { return stats_.draft_tokens_evaluated; }

 private:
  void RunRound();

  // Run the MTP head on a single (hidden_state, token) pair. When `need_draft` is true, returns the
  // head's greedy drafted next token; when false, only advances the head's KV cache (skipping the
  // 248K-vocab argmax + its stream sync) and returns 0. The KV-advance-only mode is used after an
  // accepted draft, where the next token comes from the verify pass rather than a fresh draft.
  int32_t DraftNextToken(OrtValue* hidden_last_position, int32_t token, bool need_draft = true);
  // Feed the MTP head two tokens in one forward: (hidden@row0, tok0) then (hidden@row1, tok1),
  // where hidden rows come from `hidden` (a [1,S,H] verify output). Returns the greedy argmax of
  // the last (tok1) position -- the draft for the token after tok1. This fuses the post-accept
  // KV-advance (tok0) and the next step's draft (tok1) into a single 2-token MTP forward.
  int32_t DraftTwo(OrtValue* hidden, int32_t tok0, int32_t tok1);
  // Copy one [1,1,H] position out of a [1,S,H] hidden_states OrtValue into hidden_slice_ (D2D).
  void ExtractHiddenPosition(OrtValue* hidden, int position);
  // Copy one [1,1,H] row out of a [1,S,H] hidden OrtValue (on `main_model_`'s device) into `dst`.
  void CopyHiddenRow(OrtValue* hidden, int position, Tensor& dst);
  // One MTP-head forward on a single token: the head's `hidden_states` input must already be set
  // (via SetHiddenStates) by the caller. Appends `token` to the head KV, captures the head's own
  // post-final-norm output (hidden_states_out, last row) into `head_out_hidden_` for the next
  // chained step, and returns the greedy draft (or 0 if need_draft is false).
  int32_t DraftHeadStep(int32_t token, bool need_draft = true);
  // Greedy multi-token fast path: keep the draft on device so it can feed the next head forward.
  void DraftHeadStepToDevice(int32_t token, DeviceSpan<int32_t> draft);
  void DraftHeadStepToDevice(DeviceSpan<int32_t> token, DeviceSpan<int32_t> draft);
  void CaptureDraftToDevice(DeviceSpan<int32_t> draft);
  // One MTP-head forward over `count` (hidden, token) pairs. The head's `hidden_states` input must
  // already be set by the caller to a [1,count,H] buffer whose every row is a MAIN-model hidden.
  // Appends all `count` tokens in a single forward, captures the head's own post-final-norm output
  // at the LAST row into head_out_hidden_ for the next chained step, and returns the greedy draft
  // for the token after tokens[count-1]. Used by the fused refeed+draft forward (see
  // pending_refeed_count_).
  int32_t DraftHeadStepMulti(const int32_t* tokens, int count);
  void DraftHeadStepMultiToDevice(const int32_t* tokens, int count, DeviceSpan<int32_t> draft);
  // Single-token (num_speculative_tokens == 1) draft/verify step (the original fast path).
  void GenerateStepSingle(int32_t t);
  // Multi-token (num_speculative_tokens > 1) chained draft/verify step: chains the single MTP
  // module N times (feeding its own hidden back), verifies [t, d0..d_{N-1}] in one main forward,
  // commits the longest accepted prefix + 1 bonus, and rolls the head/main state back losslessly.
  void GenerateStepMulti(int32_t t);
  // Speculative-sampling variant of the chained step (do_sample=true). Instead of greedy argmax
  // accept, each draft d_i is sampled from its truncated draft distribution q_i and accepted with
  // probability min(1, p_i(d_i)/q_i(d_i)) against the target's truncated distribution p_i; on the
  // first rejection the token is redrawn from the residual norm(max(0, p_i - q_i)); if every draft
  // is accepted a bonus is drawn from the target's next-position distribution. This makes MTP
  // draw from the same distribution as plain top-k/top-p sampling (distribution-lossless), so
  // near-tie kernel/CUDA-graph numeric flips become valid draws rather than mismatches.
  void GenerateStepMultiSample(int32_t t);
  // One head forward that samples the draft from its truncated distribution q (top_k/top_p/
  // temperature), storing the sparse q (kept ids + probs) into draft_idx_[k]/draft_prob_[k] for
  // the accept test; otherwise identical to DraftHeadStep.
  int32_t DraftHeadStepSample(int32_t token, int k);
  // Capture the head's own post-final-norm hidden (hidden_states_out, last row) into
  // head_out_hidden_ for the next chained draft step. Shared by DraftHeadStep{,Sample}.
  void CaptureHeadFeedbackHidden();
  // Cast the main model's raw logits to fp32 (device) and copy num_rows rows starting at first_row
  // into main_logits_cpu_ (host, fp32, row-major [num_rows, vocab]). Returns main_logits_cpu_.data().
  const float* MainLogitsRowsCpu(int first_row, int num_rows);
  // On-device top-k of `logits` (device ptr, [num_rows, vocab], element type `onnx_type` passed as
  // int) into topk_tok_scratch_/topk_score_scratch_ (only k*num_rows values leave the GPU). Returns
  // false if the device has no TopKScores impl (caller falls back to the host ComputeSampledCategorical).
  bool TopKScoresRows(const void* logits, int onnx_type, int num_rows, DeviceInterface& dev);
  // Apply the same min-length, repetition, and no-repeat-ngram transforms as Generator to one
  // target-logit row. `row` is its index in the current main-model output.
  std::span<const float> ProcessMainLogitsRow(std::span<const float> logits, int row);
  // Build one row's truncated (top_k/top_p/temperature) sparse distribution from the device top-k
  // scratch: softmax with temperature over the k sorted scores, then a top-p nucleus cutoff.
  void SparseFromTopKRow(int row, std::vector<int32_t>& idx, std::vector<float>& prob);
  // Greedy argmax over `num_rows` consecutive vocab rows of the main model's raw logits output
  // ([1,S,V]), starting at `first_row`, writing the token ids to `out`. Uses the device's
  // on-device Top-K kernel when available (no full-logits host copy); falls back to a host argmax.
  void ArgmaxMainRows(int first_row, int num_rows, int32_t* out);

  const Model& main_model_;
  const Model& mtp_model_;

  std::shared_ptr<GeneratorParams> main_params_;
  std::unique_ptr<Generator> main_;  // main decoder generator
  std::unique_ptr<Generator> mtp_;   // MTP head generator (drafts)
  // State retains GeneratorParams via shared_from_this, so the head needs a distinct persistent
  // parameter object. It must use the head's Config and keep graph capture off (head graph replay
  // corrupts chained drafts; the main model alone captures the verify shapes).
  std::shared_ptr<GeneratorParams> mtp_params_;

  std::shared_ptr<Tensor> hidden_slice_;     // reusable [1,1,hidden] device buffer for the handoff
  std::shared_ptr<Tensor> hidden_slice2_;    // reusable [1,2,hidden] buffer for the batched 2-token draft
  std::shared_ptr<Tensor> head_out_hidden_;  // [1,1,hidden] capture of the head's own hidden (chain feedback)
  std::shared_ptr<Tensor> refeed_hidden_;    // [1,1,hidden] scratch for re-feeding accepted drafts (main hidden)
  // Per-size [1,j,hidden] hidden buffers (j=1..N) used to re-materialize the accepted drafts in the
  // head KV in ONE batched head forward instead of one forward per accepted draft (sampling path).
  std::vector<std::shared_ptr<Tensor>> refeed_multi_;
  std::unique_ptr<OrtValue> logits_fp32_;  // reusable fp32 cast of the main model's raw logits

  std::vector<int32_t> sequence_;          // internally committed tokens (may include lookahead)
  std::vector<int32_t> emitted_sequence_;  // prompt + tokens exposed through GenerateNextToken
  std::deque<int32_t> pending_tokens_;     // internally committed tokens waiting to be exposed
  int hidden_size_{};
  int vocab_size_{};
  int max_length_{};

  // Number of speculative draft tokens per step (N). 1 = the original single-token fast path;
  // >1 chains the single MTP module N times (Qwen3.6 / vLLM-style). Taken from
  // speculative.max_draft_tokens, clamped to 1 for a head that cannot emit its own hidden.
  int num_speculative_tokens_{1};
  // Keep chained greedy drafts on device between CUDA head forwards; other devices use the
  // existing host-token path.
  bool device_draft_chain_{false};
  // search.chunk_size: max tokens per prompt forward. A one-shot forward over a long prompt
  // drives the ORT activation arena far past what the chunked path needs (measured 94 GB vs 54 GB
  // on a 2.8k-token prompt), so 0 = single forward; defaults to 256 on a windowed-state model,
  // off otherwise.
  int prefill_chunk_{0};
  bool prefill_chunk_explicit_{false};
  // Head KV length invariant (multi-token path): number of committed generated tokens currently
  // in the MTP head's KV cache (each fed once with its main hidden). The draft phase temporarily
  // extends this speculatively, then rolls it back to this value + accepted drafts.
  size_t head_len_{0};
  std::vector<int32_t> drafts_;         // scratch: the N chained draft tokens
  DeviceSpan<int32_t> drafts_device_;   // same drafts, kept on device between chained head forwards
  std::vector<int32_t> verify_tokens_;  // scratch: [t, d0..d_{N-1}] for the verify forward
  std::vector<int32_t> verify_argmax_;  // scratch: main argmax of the N+1 verify rows

  // --- Speculative-sampling (do_sample=true) state. ---
  bool sampling_{false};                // true when do_sample && temperature > 0: use rejection sampling
  int top_k_{};                         // top-k truncation shared by draft (q) and target (p) distributions
  float top_p_{};                       // top-p (nucleus) truncation
  float temperature_{1.0f};             // sampling temperature
  std::mt19937 rng_;                    // RNG for draft sampling + accept/reject + correction/bonus draws
  SampledCategorical sampled_scratch_;  // reused truncated-distribution scratch
  std::unique_ptr<LogitsPenaltyProcessor> main_logits_penalties_;
  std::vector<std::vector<int32_t>> draft_idx_;   // per draft position: truncated draft support
  std::vector<std::vector<float>> draft_prob_;    // per draft position: truncated draft probs q_k
  std::vector<std::vector<int32_t>> target_idx_;  // per verify row: truncated target support
  std::vector<std::vector<float>> target_prob_;   // per verify row: truncated target probs p_k
  std::vector<float> main_logits_cpu_;            // host fp32 copy of the requested verify rows (CPU fallback)
  std::vector<int32_t> topk_tok_scratch_;         // device top-k token ids (host, [num_rows*k])
  std::vector<float> topk_score_scratch_;         // device top-k raw scores (host, [num_rows*k])
  std::vector<float> topk_prob_scratch_;          // per-row softmax scratch over the k scores
  int topk_k_{};                                  // effective k used for the last device top-k

  // Loop carry state (see the design doc draft/verify invariant):
  int32_t next_token_{};  // token predicted for the current cache length L (not yet committed)
  size_t length_{};       // committed cache length L
  bool primed_{false};    // whether AppendTokens has run the prompt
  bool done_{false};
  // Pipelined draft: on an accepted step the next step's draft is computed ahead (fused into the
  // post-accept KV-advance as one 2-token MTP forward), so the next GenerateNextToken reuses it
  // instead of issuing a separate draft forward.
  int32_t pending_draft_{};
  bool has_pending_draft_{false};

  // Deferred + fused head refeed (multi-token greedy path). Re-materializing the accepted drafts in
  // the head KV and drafting the next step's first token are two head forwards that both consume
  // MAIN-model hidden states over consecutive positions, so they are fused into ONE (a+1)-token
  // head forward issued at the top of the next step: rows [d0..d_{a-1}, next_token_] with the
  // matching hidden rows, last-row argmax = the first draft. The head KV is therefore left in its
  // speculative state between steps and rewound lazily (nothing reads it in between).
  // -1 = no pending refeed (fresh prompt / N==1 / sampling path).
  int pending_refeed_count_{-1};
  size_t pending_refeed_head_len_{0};   // head KV length to rewind to before the fused forward
  std::vector<int32_t> merged_tokens_;  // [d0..d_{a-1}, next_token_] for the fused forward

  SpeculativeStats stats_{};
};

}  // namespace Generators

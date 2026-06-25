// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Moonshine Streaming ASR — official UsefulSensors stateful 5-session
// pipeline (frontend → encoder → adapter → cross_kv → decoder_kv).
//
// Per audio chunk (in the StreamingProcessor):
//   frontend.onnx  : causal audio frontend with explicit sample / conv-cache
//                    state buffers that persist across chunks.
//   encoder.onnx   : stateless sliding-window attention encoder.
//   adapter.onnx   : adds positional encoding using a global pos_offset
//                    (= encoder frame count BEFORE this chunk).
//   cross_kv.onnx  : precomputes cross-attention K/V from the chunk's memory.
//
// Per emitted token (in the State):
//   decoder_kv.onnx: one autoregressive step. Reads (token, self_K, self_V,
//                    cross_K, cross_V) → (logits, new_self_K, new_self_V,
//                    cross_K passthrough, cross_V passthrough). Self-KV
//                    persists across chunks within a stream until an EOS
//                    is emitted (which resets back to BOS + empty self-KV).
//
// Driven by the Generator via the shared TransducerState interface:
//   * StreamingProcessor::Process() drains chunk_samples of audio and runs
//     the four encoding stages, returning {k_cross, v_cross} as NamedTensors.
//   * generator.set_inputs(...) routes those into State::SetExtraInputs().
//   * each generate_next_token() call invokes StepToken() which runs one
//     decoder_kv step and emits at most one token.
#pragma once

#include "model.h"
#include "transducer_state.h"

namespace Generators {

struct MoonshineConfig {
  // Audio framing.
  int sample_rate{16000};
  int chunk_samples{8000};  // 500ms at 16 kHz

  // Tokens.
  int bos_token_id{1};
  int eos_token_id{2};

  // Encoder geometry (frontend output / encoder hidden dim).
  int encoder_dim{768};

  // Decoder geometry (from decoder_kv.onnx self-KV shape [layers,1,heads,T,head_size]).
  int num_decoder_layers{14};
  int num_decoder_heads{10};
  int decoder_head_size{64};
  int decoder_dim{640};

  // Frontend state shapes (from streaming_config.json).
  int sample_buffer_size{79};
  int conv1_channels{768};
  int conv1_buffer_size{4};
  int conv2_channels{1536};
  int conv2_buffer_size{4};

  // Encoder sliding-window geometry. The encoder has lookahead and requires
  // `left_context_frames` of past context per chunk. For the medium model,
  // depth=14 → left_context_frames = 16 * 14 = 224. Lookahead is held back
  // from the "stable" frame count until Flush().
  int total_lookahead{16};
  int left_context_frames{224};

  // Decoder token-emission cap. Per chunk, tokens are limited to
  //   min(ceil(memory_len * 0.020s * 6.5 tok/s), max_seq_len)
  // (matching the official moonshine streaming reference impl).
  int max_seq_len{448};
  float tokens_per_second{6.5f};
  float seconds_per_memory_frame{0.020f};

  // ONNX filenames (resolved relative to the model directory).
  std::string frontend_filename{"frontend.onnx"};
  std::string encoder_filename{"encoder.onnx"};
  std::string adapter_filename{"adapter.onnx"};
  std::string cross_kv_filename{"cross_kv.onnx"};
  std::string decoder_kv_filename{"decoder_kv.onnx"};

  void PopulateFromConfig(const Config& config);
};

struct MoonshineStreamingModel : Model {
  MoonshineStreamingModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_frontend_;
  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_adapter_;
  std::unique_ptr<OrtSession> session_cross_kv_;
  std::unique_ptr<OrtSession> session_decoder_kv_;

  std::unique_ptr<OrtSessionOptions> session_options_;

  MoonshineConfig moonshine_config_;
};

/// State for the Moonshine streaming encoder-decoder pipeline.
///
/// Moonshine streaming re-decodes the entire accumulated memory from BOS on
/// every chunk, so consecutive passes can disagree on the tail (until more
/// audio arrives and the model commits). To present token-by-token
/// incremental output to the user without retracting earlier tokens, we
/// emit only the longest-common-prefix between this pass and the previous
/// pass that hasn't been emitted yet ("committed delta"). On the final
/// flush chunk, the full pass is committed.
///
/// Per SetExtraInputs() call:
///   1. cache the chunk's cross-attention K/V tensors,
///   2. reset self-KV to length 0 and last_token = BOS,
///   3. run a full AR decode pass (BOS → EOS or cap),
///   4. compute the new committed prefix (LCP with previous pass, or the
///      full pass if is_final),
///   5. queue tokens in (previously_emitted .. new_committed] for delivery
///      via StepToken().
///
/// StepToken() drains one token at a time from the queue so the existing
/// Generator pump (generate_next_token + get_next_tokens) keeps working as
/// a standard streaming API for the caller.
struct MoonshineStreamingState : TransducerState {
  MoonshineStreamingState(const MoonshineStreamingModel& model, const GeneratorParams& params);
  ~MoonshineStreamingState() override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;
  void StepToken() override;

  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  const MoonshineStreamingModel& moonshine_model_;
  MoonshineConfig config_;

  // Current chunk's precomputed cross-attention K/V (kept alive for the
  // duration of the chunk's decoder loop).
  std::shared_ptr<Tensor> k_cross_tensor_;
  std::shared_ptr<Tensor> v_cross_tensor_;

  // Self-attention KV cache, scoped to a single SetExtraInputs() pass.
  // Reset to length 0 at the start of each pass.
  std::unique_ptr<OrtValue> k_self_;
  std::unique_ptr<OrtValue> v_self_;

  // Pre-allocated [1, 1] int64 token tensor (mutated each step).
  std::unique_ptr<OrtValue> token_tensor_;

  // Tokens queued for incremental delivery via StepToken(). Contains the
  // newly-committed delta vs the previous pass (or the entire un-emitted
  // suffix on the final chunk).
  std::vector<int32_t> pending_tokens_;
  size_t pending_idx_{0};

  // Full token sequence from the previous chunk's pass. Used to compute the
  // longest-common-prefix that has now "committed" (won't be retracted).
  std::vector<int32_t> previous_pass_tokens_;

  // How many tokens we've already emitted in this stream. Drives the
  // delta-emission so we never emit a token twice.
  size_t emitted_count_{0};

  // Memory length of the previous chunk's cross-KV. Used to detect a new
  // utterance (memory shrinks back to a small value after Flush) so we can
  // reset the commit tracking.
  int64_t previous_memory_len_{0};

  /// Run one decoder_kv step, return the argmax token. Updates k_self_/v_self_.
  int RunDecoderStep(int64_t input_token);
  void ResetSelfKv();
};

}  // namespace Generators

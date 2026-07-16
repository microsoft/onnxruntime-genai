// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Moonshine Streaming ASR — official UsefulSensors stateful 5-session
// pipeline (frontend → encoder → adapter → cross_kv → decoder_kv).
//
// Following the Nemotron speech design, ALL five ONNX sessions are owned by
// the orchestrator State (MoonshineStreamingState) as sub-states, and the
// StreamingProcessor is reduced to a DSP-only stage: it buffers audio, runs
// VAD, and hands the State a raw audio chunk plus per-chunk {is_silent,
// is_final} signals. The State owns every piece of per-stream state
// (frontend ring buffers, accumulated features / memory, incremental
// cross-KV cache, VAD-driven segment resets) and drives the whole pipeline.
//
// Sub-states (each a State wrapping one session):
//   frontend  : causal audio frontend with explicit sample / conv-cache
//               state buffers that persist across chunks.
//   encoder   : stateless sliding-window attention encoder.
//   adapter   : adds positional encoding using a global pos_offset
//               (= encoder frame count BEFORE this chunk).
//   cross_kv  : precomputes cross-attention K/V from the chunk's memory.
//   decoder_kv: one autoregressive step. Reads (token, self_K, self_V,
//               cross_K, cross_V) → (logits, new_self_K, new_self_V,
//               cross_K passthrough, cross_V passthrough).
//
// Driven by the Generator via the shared TransducerState interface:
//   * StreamingProcessor::Process() drains chunk_samples of audio and emits
//     {audio_chunk, is_silent, is_final} as NamedTensors.
//   * generator.set_inputs(...) routes those into State::SetExtraInputs(),
//     which merely caches the chunk + signals and flags a pending run.
//   * the first generate_next_token() after a new chunk invokes StepToken(),
//     which runs the full frontend→encoder→adapter→cross_kv→decode pipeline
//     once, queues the newly-committed tokens, and emits the first one;
//     subsequent StepToken() calls drain the queue one token at a time.
#pragma once

#include "model.h"
#include "transducer_state.h"

namespace Generators {

// All values come from genai_config.json — the generic `model`,
// `model.encoder`/`model.decoder` sections, and the bespoke `model.moonshine`
// section. There are NO code defaults: every field is populated by
// PopulateFromConfig, and a field absent from the JSON is left zero-initialized.
// genai_config.json is the single source of truth.
struct MoonshineConfig {
  // Audio framing.
  int chunk_samples{};  // e.g. 8000 = 500ms at 16 kHz

  // Tokens.
  int bos_token_id{};
  int eos_token_id{};

  // Encoder geometry (frontend output / encoder hidden dim).
  int encoder_dim{};

  // Decoder geometry (from decoder_kv self-KV shape [layers,1,heads,T,head_size]).
  int num_decoder_layers{};
  int num_decoder_heads{};
  int decoder_head_size{};
  int decoder_dim{};

  // Frontend state shapes (from streaming_config.json).
  int sample_buffer_size{};
  int conv1_channels{};
  int conv1_buffer_size{};
  int conv2_channels{};
  int conv2_buffer_size{};

  // Encoder sliding-window geometry. The encoder has lookahead and requires
  // `left_context_frames` of past context per chunk. Lookahead is held back
  // from the "stable" frame count until Flush().
  int total_lookahead{};
  int left_context_frames{};

  // Decoder token-emission cap. Per chunk, tokens are limited to
  //   min(ceil(memory_len * seconds_per_memory_frame * tokens_per_second), max_seq_len)
  // (matching the official moonshine streaming reference impl).
  int max_seq_len{};
  float tokens_per_second{};
  float seconds_per_memory_frame{};

  // Hard-cap on accumulated memory before forcing a state reset. Mirrors
  // upstream Moonshine's `vad_max_segment_duration`. Without this, the parallel
  // teacher-forcing per chunk costs O(emitted_tokens) which sums to O(N^2)
  // over the full audio. With the cap, each "segment" is independent and the
  // AR loop stays small (e.g. 500 frames = 10s @ 50fps). Set <=0 to disable.
  int max_segment_memory_frames{};

  // Minimum accumulated memory before VAD-detected silence is allowed to
  // trigger a segment break. Below this threshold, silent chunks are dropped
  // (treated as pre/inter-utterance pause) but do NOT cut the segment — we
  // want enough committed speech behind us that an early-segment short
  // silence (e.g. comma-pause, breath) doesn't fragment the transcript.
  // After this threshold, any silent chunk triggers an is_final flush + reset
  // (e.g. 250 frames = 5s @ 50fps). Only active when VAD is enabled; set <=0
  // to disable VAD-based segmentation entirely (hard cap still applies).
  int min_segment_memory_frames{};

  // Pipeline component filenames. All 5 are loaded from
  // genai_config.json's `model.moonshine` section — there are no defaults.
  // PopulateFromConfig throws if any is missing.
  std::string frontend_filename;
  std::string encoder_filename;
  std::string adapter_filename;
  std::string cross_kv_filename;
  std::string decoder_kv_filename;

  void PopulateFromConfig(const Config& config);
};

struct MoonshineStreamingModel : Model {
  MoonshineStreamingModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  // The streaming pipeline has five sessions (frontend, encoder, adapter,
  // cross_kv, decoder_kv). They are consumed exclusively by the per-stream
  // sub-states declared below (owned by MoonshineStreamingState), each of
  // which wraps a single session behind the shared State::Run() machinery.
  std::unique_ptr<OrtSession> session_frontend_;
  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_adapter_;
  std::unique_ptr<OrtSession> session_cross_kv_;
  std::unique_ptr<OrtSession> session_decoder_kv_;

  std::unique_ptr<OrtSessionOptions> session_options_;

  MoonshineConfig moonshine_config_;
};

// ---------------------------------------------------------------------------
// Per-stream sub-states. Each wraps one ONNX session using the shared State
// I/O machinery (input_names_/inputs_ + output_names_/outputs_ + State::Run).
// All persistent cross-chunk state lives in the orchestrator
// (MoonshineStreamingState), which is a friend of each sub-state so it can
// take ownership of freshly-allocated output tensors after every Run.
// ---------------------------------------------------------------------------

/// frontend: audio → new feature frames + updated causal state buffers.
struct MoonshineFrontendSubState : State {
  MoonshineFrontendSubState(const MoonshineStreamingModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  /// Point the audio_chunk input at `audio_chunk` (owned by the caller for
  /// the duration of the Run).
  void SetAudioInput(OrtValue* audio_chunk);

  /// Re-point the five state-buffer inputs at the sub-state's current
  /// buffers (call after the caller moves the *_out outputs back in).
  void UpdateStateInputs();

  /// Reallocate zeroed state buffers (new utterance / segment).
  void Reset();

 private:
  friend struct MoonshineStreamingState;

  const MoonshineStreamingModel& model_;
  MoonshineConfig config_;

  std::unique_ptr<OrtValue> sample_buffer_;
  std::unique_ptr<OrtValue> sample_len_;
  std::unique_ptr<OrtValue> conv1_buffer_;
  std::unique_ptr<OrtValue> conv2_buffer_;
  std::unique_ptr<OrtValue> frame_count_;

  size_t audio_input_idx_{};
  size_t sample_buffer_input_idx_{};
  size_t sample_len_input_idx_{};
  size_t conv1_buffer_input_idx_{};
  size_t conv2_buffer_input_idx_{};
  size_t frame_count_input_idx_{};

  void AllocateStateBuffers();
};

/// encoder: stateless sliding-window attention. features → encoded.
struct MoonshineEncoderSubState : State {
  MoonshineEncoderSubState(const MoonshineStreamingModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetFeaturesInput(OrtValue* features);

 private:
  friend struct MoonshineStreamingState;
  const MoonshineStreamingModel& model_;
};

/// adapter: (encoded, pos_offset) → memory. Adds positional encoding.
struct MoonshineAdapterSubState : State {
  MoonshineAdapterSubState(const MoonshineStreamingModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetInputs(OrtValue* encoded, int64_t pos_offset);

 private:
  friend struct MoonshineStreamingState;
  const MoonshineStreamingModel& model_;
  std::unique_ptr<OrtValue> pos_tensor_;  // int64[1]
  size_t encoded_input_idx_{};
  size_t pos_input_idx_{};
};

/// cross_kv: memory → (k_cross, v_cross). Pure per-frame projection.
struct MoonshineCrossKvSubState : State {
  MoonshineCrossKvSubState(const MoonshineStreamingModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetMemoryInput(OrtValue* memory);

 private:
  friend struct MoonshineStreamingState;
  const MoonshineStreamingModel& model_;
};

/// decoder_kv: one autoregressive step over a [1, N] token input.
/// (token, k_self, v_self, out_k_cross, out_v_cross)
///   → (logits, out_k_self, out_v_self, out_k_cross, out_v_cross).
struct MoonshineDecoderKvSubState : State {
  MoonshineDecoderKvSubState(const MoonshineStreamingModel& model, const GeneratorParams& params);

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

  void SetInputs(OrtValue* token, OrtValue* k_self, OrtValue* v_self,
                 OrtValue* k_cross, OrtValue* v_cross);

 private:
  friend struct MoonshineStreamingState;
  const MoonshineStreamingModel& model_;
  size_t token_input_idx_{};
  size_t k_self_input_idx_{};
  size_t v_self_input_idx_{};
  size_t k_cross_input_idx_{};
  size_t v_cross_input_idx_{};
};

/// Orchestrator state for the Moonshine streaming encoder-decoder pipeline.
///
/// Owns all five ONNX sub-states and every piece of per-stream state
/// (frontend ring buffers, accumulated features / memory, incremental
/// cross-KV cache, self-KV, VAD-driven segment resets). The processor only
/// buffers audio + runs VAD and hands us one raw chunk plus {is_silent,
/// is_final} per SetExtraInputs() call.
///
/// Moonshine streaming re-decodes the entire accumulated memory from BOS on
/// every chunk, so consecutive passes can disagree on the tail (until more
/// audio arrives and the model commits). To present token-by-token
/// incremental output to the user without retracting earlier tokens, we
/// emit only the longest-common-prefix between this pass and the previous
/// pass that hasn't been emitted yet ("committed delta"). On the final
/// flush chunk (or a VAD/hard-cap segment break), the full pass is committed.
///
/// SetExtraInputs() only caches the chunk + signals and flags a pending run
/// (it may be called more than once per chunk by the Generator pump). The
/// first StepToken() after a new chunk runs the full pipeline once:
///   1. run frontend → accumulate features,
///   2. run encoder (sliding window) + adapter → append to memory,
///   3. incrementally refresh the cross-KV cache,
///   4. reset self-KV, teacher-force the committed prefix + AR-decode the
///      new suffix,
///   5. commit (LCP vs previous pass, or the full pass on is_final) and
///      queue the newly-committed tokens.
/// Subsequent StepToken() calls drain that queue one token at a time so the
/// existing Generator pump (generate_next_token + get_next_tokens) keeps
/// working as a standard streaming API for the caller.
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

  // ---- Owned ONNX sub-states (one per session) --------------------------
  std::unique_ptr<MoonshineFrontendSubState> frontend_state_;
  std::unique_ptr<MoonshineEncoderSubState> encoder_state_;
  std::unique_ptr<MoonshineAdapterSubState> adapter_state_;
  std::unique_ptr<MoonshineCrossKvSubState> cross_kv_state_;
  std::unique_ptr<MoonshineDecoderKvSubState> decoder_kv_state_;

  // ---- Current chunk signals (set by SetExtraInputs, consumed once by the
  // first StepToken() via RunPipeline) -----------------------------------
  std::shared_ptr<Tensor> current_audio_;  // [1, num_samples] float32
  bool current_is_silent_{false};
  bool current_is_final_{false};
  bool need_pipeline_run_{false};

  // ---- Accumulated encoder / adapter / memory state ---------------------
  // Accumulated encoder-input features ([total_features_, encoder_dim]).
  std::vector<float> accumulated_features_;
  int total_features_{0};
  // Number of features already adapted into memory.
  int encoder_frames_emitted_{0};
  // Running pos_offset fed to the adapter (== total memory frames so far).
  int64_t adapter_pos_offset_{0};
  // Accumulated decoder memory ([memory_len_, decoder_dim]).
  std::vector<float> accumulated_memory_;
  int memory_len_{0};

  // Cached cross-KV tensors, grown incrementally (cross_kv is a pure
  // per-frame projection): when memory grows by `new_frames`, run cross_kv
  // on just those rows and concat into the cached [L,1,H,M,D] tensors.
  std::shared_ptr<Tensor> cached_k_cross_;
  std::shared_ptr<Tensor> cached_v_cross_;
  int memory_in_cross_kv_{0};  // memory frames already projected into cached.
  bool cross_kv_valid_{false};

  // Set when the current chunk closed a segment (hard memory cap reached OR
  // VAD detected silence past min_segment_memory_frames OR Flush). The next
  // RunPipeline() resets all accumulation before processing the new chunk so
  // the next segment starts from a clean BOS.
  bool needs_reset_{false};

  // Current chunk's precomputed cross-attention K/V (kept alive for the
  // duration of the chunk's decoder loop).
  std::shared_ptr<Tensor> k_cross_tensor_;
  std::shared_ptr<Tensor> v_cross_tensor_;

  // Self-attention KV cache, scoped to a single decode pass.
  // Reset to length 0 at the start of each pass.
  std::unique_ptr<OrtValue> k_self_;
  std::unique_ptr<OrtValue> v_self_;

  // Pre-allocated [1, 1] int64 token tensor (mutated each AR step). A
  // separate [1, N] tensor is created on-the-fly when teacher-forcing the
  // committed prefix in one parallel call.
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
  // utterance (memory shrinks back to a small value after a segment break)
  // so we can reset the commit tracking.
  int64_t previous_memory_len_{0};

  // Runs the full frontend→encoder→adapter→cross_kv→decode pipeline for the
  // cached chunk exactly once. Invoked by the first StepToken() after a new
  // chunk (gated by need_pipeline_run_). Handles VAD routing + segment
  // resets, then queues the newly-committed tokens.
  void RunPipeline();

  /// Run the frontend on `num` samples of audio, then accumulate features
  /// and (if any new stable frames have been produced) encoder + adapter +
  /// append-to-memory. is_final=true treats all accumulated features as
  /// stable (no lookahead held back).
  void RunFrontendAndAccumulate(const float* audio, size_t num, bool is_final);

  /// If !cross_kv_valid_, run cross_kv incrementally on the memory rows
  /// produced since the last refresh and concat into cached {k_cross,
  /// v_cross}. No-op when cross_kv is already valid or memory is empty.
  void RefreshCrossKv();

  /// Reset self-KV, run the full teacher-forced + AR decode pass over the
  /// current cross-KV, and queue the newly-committed tokens. `is_final`
  /// commits the entire pass (vs the longest-common-prefix with the
  /// previous pass).
  void DecodeAndQueue(bool is_final);

  /// Clear all accumulated per-segment state (frontend buffers, features,
  /// memory, cross-KV cache) so the next chunk starts a fresh segment.
  void ResetAccumulation();

  /// Run decoder_kv with `tokens` as input (length >= 1). decoder_kv accepts
  /// a dynamic seq dim, so this is one call regardless of length. Updates
  /// k_self_/v_self_ (grown by `tokens.size()`) and returns the argmax of
  /// the LAST position's logits — i.e. the predicted token that would come
  /// after `tokens`.
  int RunDecoderForward(const std::vector<int64_t>& tokens);
  void ResetSelfKv();
};

}  // namespace Generators

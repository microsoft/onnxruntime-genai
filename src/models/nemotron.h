// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"

namespace Generators {

// =============================================================================
// Nemotron ASR Model (FastConformer-CacheAware RNNT)
//
// Supports two modes:
//
// 1. BATCH mode (non-streaming encoder):
//    - Encoder: audio_signal[B,128,T] + length[B] -> outputs[B,1024,T'] + encoded_lengths[B]
//    - Processes entire utterance at once
//
// 2. STREAMING mode (cache-aware encoder):
//    - Encoder: audio_signal[B,128,chunk_T] + length[B] +
//               cache_last_channel[B,24,70,1024] + cache_last_time[B,24,1024,8] +
//               cache_last_channel_len[B]
//            -> outputs[B,1024,T'] + encoded_lengths[B] +
//               cache_last_channel_next[...] + cache_last_time_next[...] +
//               cache_last_channel_len_next[B]
//    - Processes audio chunk-by-chunk with carried-forward caches
//
// Decoder: targets[B,L] + h_in[2,B,640] + c_in[2,B,640]
//          -> decoder_output[B,640,L] + h_out[2,B,640] + c_out[2,B,640]
// Joint:   encoder_output[B,T',1024] + decoder_output[B,L,640] -> joint_output[B,T',L,1025]
// =============================================================================

struct NemotronModel : Model {
  NemotronModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_decoder_;
  std::unique_ptr<OrtSession> session_joint_;

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;

  // Model dimensions (from config / ONNX model)
  int encoder_hidden_size_{1024};
  int decoder_hidden_size_{640};
  int vocab_size_{1025};       // 1024 BPE tokens + 1 blank
  int blank_id_{1024};         // RNNT blank is last token
  int num_encoder_layers_{24};
  int num_decoder_layers_{2};  // LSTM layers

  // Streaming config (from genai_config.json)
  bool streaming_enabled_{false};
  int cache_last_channel_size_{70};   // MHA left-context cache length
  int conv_context_size_{8};          // causal conv buffer size
  int chunk_mel_frames_{112};         // mel frames per chunk (~1.12s)
  int valid_out_len_{14};             // encoder output frames per chunk
};

// =============================================================================
// NemotronState - Orchestrates encoder + RNNT decode loop
//
// In streaming mode:
//   - Each Run() call processes one audio chunk through encoder
//   - Encoder cache tensors are carried forward across calls
//   - RNNT greedy decode runs on NEW encoder frames only
//   - Decoder LSTM state is carried forward across chunks
// =============================================================================

struct NemotronState : State {
  NemotronState(const NemotronModel& model, const GeneratorParams& params, DeviceSpan<int32_t> sequence_lengths);
  NemotronState(const NemotronState&) = delete;
  NemotronState& operator=(const NemotronState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;
  OrtValue* GetInput(const char* name) override;
  OrtValue* GetOutput(const char* name) override;

 private:
  const NemotronModel& model_;

  // --- Encoder ---
  void RunEncoder();
  void RunStreamingEncoder();        // Cache-aware chunked encoder
  void TransposeEncoderOutput();     // [B,1024,T'] -> [B,T',1024]
  bool encoder_done_{false};
  int encoded_length_{0};
  int batch_size_{1};

  // Encoder I/O tensors
  std::unique_ptr<OrtValue> audio_signal_;               // [B, 128, T]
  std::unique_ptr<OrtValue> audio_length_;               // [B]
  std::unique_ptr<OrtValue> encoder_output_raw_;         // [B, 1024, T'] from ONNX
  std::unique_ptr<OrtValue> encoded_lengths_;            // [B]

  // Transposed encoder output for Joint network
  std::vector<float> encoder_output_transposed_;         // [B, T', 1024]

  // --- Streaming encoder caches ---
  std::unique_ptr<OrtValue> cache_last_channel_;         // [B, n_layers, cache_len, d_model]
  std::unique_ptr<OrtValue> cache_last_time_;            // [B, n_layers, d_model, conv_ctx]
  std::unique_ptr<OrtValue> cache_last_channel_len_;     // [B]
  bool caches_initialized_{false};
  bool has_new_audio_{false};        // True when new audio_signal is set via SetExtraInputs

  // --- Decoder ---
  void RunDecoder(int64_t token_id);
  void ResetDecoderState();  // Initialize LSTM states to zeros
  std::unique_ptr<OrtValue> decoder_targets_;            // [B, 1]
  std::unique_ptr<OrtValue> decoder_target_length_;      // [B]

  // Decoder hidden output for Joint network
  std::vector<float> decoder_hidden_out_;                // [1, 640] extracted hidden for Joint

  // LSTM hidden/cell states carried forward between decode steps
  std::unique_ptr<OrtValue> decoder_h_state_;            // [num_layers, B, 640]
  std::unique_ptr<OrtValue> decoder_c_state_;            // [num_layers, B, 640]
  bool decoder_initialized_{false};  // True after first ResetDecoderState + initial RunDecoder

  // --- Joint ---
  int RunJoint(const float* enc_frame, const float* dec_hidden);  // Returns argmax token
  std::unique_ptr<OrtValue> joint_enc_input_;            // [1, 1, 1024]
  std::unique_ptr<OrtValue> joint_dec_input_;            // [1, 1, 640]

  // --- RNNT Greedy Decode ---
  void GreedyDecode();
  void GreedyDecodeIncremental(int num_new_frames);  // Decode only new encoder frames
  std::vector<int32_t> decoded_tokens_;
  int emit_index_{0};   // Index into decoded_tokens_ for step-by-step emission
  bool rnnt_done_{false};

  // Logits output tensor exposed to the framework
  std::unique_ptr<OrtValue> logits_tensor_;              // [1, vocab_size]
  DeviceSpan<float> logits_span_;                        // Wrapped view of logits_tensor_

  // Track the last decoder token for RNNT continuity across chunks
  int64_t last_decoder_token_{0};  // Carries across chunks in streaming mode
};

}  // namespace Generators

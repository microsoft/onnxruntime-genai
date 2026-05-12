// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet TDT speech recognition model.
//
// Mirrors the standard batch model structure (Model + State subclasses) so it can
// be driven by the standard Generator pipeline:
//
//     model = og.Model(config)
//     processor = model.create_multimodal_processor()
//     audios = og.Audios.open("audio.wav")
//     inputs = processor("", audios=audios)
//     params = og.GeneratorParams(model)
//     generator = og.Generator(model, params)
//     generator.set_inputs(inputs)
//     while not generator.is_done():
//         generator.generate_next_token()
//     transcription = processor.decode(generator.get_sequence(0))
//
// Internally the encoder is fed in fixed-length chunks with left+right
// context (matching the original NeMo streaming reference); the TDT
// (Token-and-Duration Transducer) decoder runs against the encoder output
// frames, producing one token id per call to State::Run().

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model.h"

namespace Generators {

struct ParakeetTdtConfig {
  // Encoder dimensions
  int hidden_dim{};
  int num_encoder_layers{};

  // Decoder LSTM dimensions
  int decoder_lstm_dim{};
  int decoder_lstm_layers{};

  // Vocabulary
  int blank_id{};

  // Streaming chunk config
  int sample_rate{};
  int chunk_samples{};
  int subsampling_factor{};
  int max_symbols_per_step{};
  int left_context_samples{};
  int right_context_samples{};

  // TDT parameters
  std::vector<int> tdt_durations;

  // Encoder I/O names
  std::string enc_in_audio;
  std::string enc_in_length;
  std::string enc_out_encoded;
  std::string enc_out_length;

  // Decoder (prediction network) I/O names
  std::string dec_in_targets;
  std::string dec_in_targets_length;
  std::string dec_in_lstm_hidden_state;
  std::string dec_in_lstm_cell_state;
  std::string dec_out_outputs;
  std::string dec_out_outputs_length;
  std::string dec_out_lstm_hidden_state;
  std::string dec_out_lstm_cell_state;

  // Joiner I/O names
  std::string join_in_encoder;
  std::string join_in_decoder;
  std::string join_out_logits;

  void PopulateFromConfig(const Config& config);
};

struct ParakeetTdtModel : Model {
  ParakeetTdtModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_encoder_;
  std::unique_ptr<OrtSession> session_decoder_;
  std::unique_ptr<OrtSession> session_joiner_;

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;
  std::unique_ptr<OrtSessionOptions> decoder_session_options_;
  std::unique_ptr<OrtSessionOptions> joiner_session_options_;

  ParakeetTdtConfig parakeet_config_;
};

// State driving the streaming TDT decoder.
//
// SetExtraInputs() only caches the full mel tensor produced by the processor;
// no encoder/decoder/joiner work is done there. Each call to State::Run()
// advances the TDT loop by exactly one emitted token: the encoder is run
// lazily, one chunk at a time, only when the current encoder window has been
// fully consumed. Blank frames are skipped silently inside Run() so that the
// caller always observes a real token (or eos when the audio is exhausted).
struct ParakeetTdtState : State {
  ParakeetTdtState(const ParakeetTdtModel& model, const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length,
                        DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 private:
  // LSTM decoder state held between TDT decoding steps.
  struct DecState {
    std::unique_ptr<OrtValue> state_h;         // [lstm_layers, 1, lstm_dim]
    std::unique_ptr<OrtValue> state_c;         // [lstm_layers, 1, lstm_dim]
    std::unique_ptr<OrtValue> decoder_output;  // [1, lstm_dim, 1]
    int64_t last_token{0};
  };

  // Encode the next audio chunk and update current_encoder_ / current_t_ /
  // current_end_frame_. Sets finished_ when the trailing chunk is encoded.
  void EncodeNextChunk();
  // Run the TDT loop until a non-blank token is emitted, or return blank_id
  // (== eos) when the whole utterance has been consumed.
  int32_t EmitNextToken();
  void InitializeDecoderState();
  void StepDecoder(int32_t token_id);

  const ParakeetTdtModel& model_;
  ParakeetTdtConfig cfg_;

  DecState dec_;

  // Full-utterance mel features supplied by ParakeetTdtProcessor via
  // SetExtraInputs and reused by every chunk (no per-chunk featurizer state).
  // This matches PyTorch implementation from Nvidia.
  // Layout: [num_mels, total_mel_frames_], row-major, already normalized in
  // the processor via ort_extensions::PerFeatureNormalize.
  std::vector<float> full_mel_;
  int total_mel_frames_{0};
  bool mel_loaded_{false};

  // Streaming state: where the next chunk starts (in audio-sample space),
  // total audio length, and whether the trailing chunk has been encoded.
  size_t total_audio_{0};
  size_t next_chunk_start_{0};
  bool finished_{false};
  bool initialized_{false};

  // Current encoder window (output of the most recent encoder run), staged
  // on CPU for per-frame slicing into the joiner. The encoder runs on
  // chunk + left context + right context, so the underlying tensor
  // [1, hidden_dim, T'] covers all three regions. The three indices below
  // carve out only the chunk-proper region for decoding; the context frames
  // exist so the encoder has enough receptive field at chunk boundaries.
  //
  //   encoder frames:  0 ........ a ........ b ........ T'
  //                    └─ left ──┘└── chunk ─┘└─ right ─┘
  //                                ↑          ↑          ↑
  //                            current_t_  current_end_  current_enc_time_
  //                            (start)      _frame_       (= T')
  std::vector<float> current_encoder_cpu_;  // [hidden_dim, T'] flattened
  int64_t current_enc_time_{0};
  int64_t current_t_{0};
  int64_t current_end_frame_{0};
  int symbols_this_frame_{0};

  // Logits buffer reused across calls (size = vocab_size).
  int logits_size_{};
  std::vector<float> logits_buffer_;
  DeviceSpan<float> logits_device_buffer_;

  // Per-step joiner inputs, allocated lazily on the first decoding step and
  // reused across every subsequent step to avoid per-frame allocator churn.
  // Shapes are fixed: encoder_frame_* = [1, 1, encoder_hidden_dim],
  // decoder_frame_ = [1, 1, decoder_lstm_dim].
  std::unique_ptr<OrtValue> encoder_frame_cpu_;
  std::unique_ptr<OrtValue> encoder_frame_;
  std::unique_ptr<OrtValue> decoder_frame_;
};

}  // namespace Generators

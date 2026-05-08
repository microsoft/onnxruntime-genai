// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Parakeet TDT speech recognition model.
//
// Mirrors the Whisper model structure (Model + State subclasses) so it can
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
  int vocab_size{};
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
  int tdt_num_extra_outputs{};

  // Encoder I/O names
  std::string enc_in_audio;
  std::string enc_in_length;
  std::string enc_out_encoded;
  std::string enc_out_length;

  // Decoder (prediction network) I/O names
  std::string dec_in_targets;
  std::string dec_in_target_length;
  std::string dec_in_lstm_hidden_state;
  std::string dec_in_lstm_cell_state;
  std::string dec_out_outputs;   // decoder_output
  std::string dec_out_prednet_lengths;
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

// State holding the chunked TDT decoding pipeline.
//
// On the first call to SetExtraInputs() (which receives the raw PCM tensor
// produced by ParakeetTdtProcessor) the entire utterance is transcribed in
// one shot internally — chunk by chunk — and the resulting token ids are
// stored in `decoded_tokens_`. Each subsequent State::Run() returns a
// one-hot logits row that selects the next pre-computed token; once the
// list is exhausted the eos token id is emitted so that the search loop
// terminates.
struct ParakeetTdtState : State {
  ParakeetTdtState(const ParakeetTdtModel& model, const GeneratorParams& params);

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int total_length,
                        DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 private:
  // LSTM decoder state held between TDT decoding steps.
  struct DecState {
    std::unique_ptr<OrtValue> state_h;        // [lstm_layers, 1, lstm_dim]
    std::unique_ptr<OrtValue> state_c;        // [lstm_layers, 1, lstm_dim]
    std::unique_ptr<OrtValue> decoder_output; // [1, lstm_dim, 1]
    int64_t last_token{0};
  };

  void TranscribeAll(const float* audio, size_t num_samples);
  void ProcessChunk(const float* audio, size_t total_audio,
                    size_t chunk_start, size_t chunk_end, bool is_last);
  void RunTDTDecoder(OrtValue* encoder_output,
                     int64_t start_frame,
                     int64_t end_frame);
  void InitializeDecoderState();
  void StepDecoder(int32_t token_id);

  const ParakeetTdtModel& model_;
  ParakeetTdtConfig cfg_;

  DecState dec_;
  bool decoded_{false};
  std::vector<int32_t> decoded_tokens_;
  int32_t eos_token_id_{};

  // Per-mel-bin global mean/std computed once over the entire utterance and
  // applied to every chunk's mel — matches NeMo non-streaming
  // `normalize_batch` ("per_feature" with N-1, eps=1e-5).
  std::vector<float> global_mel_mean_;
  std::vector<float> global_mel_inv_std_;

  // Full-utterance mel features computed once in TranscribeAll and reused by
  // every chunk (NeMo-style preprocessing — no per-chunk featurizer state).
  // Layout: [num_mels, total_mel_frames_], row-major, already globally
  // mean/std normalized.
  std::vector<float> full_mel_;
  int total_mel_frames_{0};

  // Logits buffer reused across calls (size = vocab_size + 1).
  int logits_size_{};
  std::vector<float> logits_buffer_;
  DeviceSpan<float> logits_device_;
};

}  // namespace Generators

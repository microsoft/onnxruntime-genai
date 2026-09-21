// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "models/preprocessing/processor.h"

namespace Generators {

// Marker the prompt carries once per audio clip, in clip order. LFM2-Audio has no audio input token
// of its own: the reference implementation (liquid-audio's ChatState) splices the encoder output
// between the text tokens by position, so the marker is a convention of this runtime, and the
// processor replaces it with model.audio_token_id repeated once per encoder frame.
inline constexpr char kLfm2AudioMarker[] = "<|audio|>";

// Log-mel front end of the FastConformer encoder, NeMo's AudioToMelSpectrogramPreprocessor with
// the settings of the published checkpoints (config.json "preprocessor").
struct Lfm2AudioMelConfig {
  int num_mels{128};
  int fft_size{512};
  int win_length{400};  // window_size 0.025 s at 16 kHz
  int hop_length{160};  // window_stride 0.01 s at 16 kHz
  int sample_rate{16000};
  float preemph{0.97f};
  float log_eps{5.960464477539063e-08f};  // log_zero_guard_value 2^-24, added before the log
  float norm_eps{1e-5f};                  // added to the per-feature standard deviation
};

// Frame accounting of the reference preprocessor. torch.stft with center=True yields one frame per
// hop plus one, but NeMo's get_seq_len counts only num_samples / hop_length of them as valid: the
// per-feature normalization uses the valid frames and the rest are zeroed, yet every frame is still
// handed to the encoder and counted towards the decoder tokens.
int64_t Lfm2AudioNumMelFrames(int64_t num_samples, const Lfm2AudioMelConfig& config);
int64_t Lfm2AudioNumValidMelFrames(int64_t num_samples, const Lfm2AudioMelConfig& config);

// Decoder tokens one clip contributes: the encoder subsamples the mel frames by subsampling_factor
// with ceil rounding (three stride-2 convolutions with padding 1 for the shipped models).
int64_t Lfm2AudioNumTokens(int64_t num_mel_frames, int64_t subsampling_factor);

// Computes the normalized log-mel spectrogram of one mono clip, time-major [num_frames, num_mels],
// the layout the encoder's mel_spectrogram input takes. Throws if the clip is too short to
// normalize (fewer than two valid frames).
std::vector<float> ComputeLfm2AudioMel(const float* pcm, int64_t num_samples, const Lfm2AudioMelConfig& config,
                                       int64_t& num_frames);

// Splits the prompt at each marker. Throws unless the prompt holds exactly one marker per clip, the
// rule the vision processor applies to <image>, so the encoder features and the placeholders can
// never be misaligned or silently dropped. Returns num_clips + 1 text segments.
std::vector<std::string> SplitLfm2AudioPrompt(const std::string& prompt, size_t num_clips);

struct Lfm2AudioProcessor : Processor {
  Lfm2AudioProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  // One clip's log-mel frames, with what the rest of the pipeline has to know about them.
  struct ClipMel {
    std::vector<float> mel;  // num_frames * num_mels, frame-major
    int64_t num_frames{};    // what the encoder is given for this clip
    int64_t num_tokens{};    // placeholders it is worth, one per encoder frame
  };

  // Decodes clip `index` to mono PCM at the encoder's sample rate and turns it into log-mel frames.
  ClipMel ComputeClipMel(const Audios& audios, size_t index) const;

  // The prompt's text tokenized segment by segment, with each clip's placeholders spliced between.
  std::vector<int32_t> MakePromptTokens(const Tokenizer& tokenizer, const std::vector<std::string>& segments,
                                        const std::vector<ClipMel>& clips) const;

  // Stages the clips in one zero-padded [num_clips, longest, num_mels] tensor, cast to the speech
  // model's input type, and adds it with the per-clip frame and placeholder counts.
  void AddAudioTensors(const std::vector<ClipMel>& clips, Ort::Allocator& allocator, NamedTensors& named_tensors) const;

  ONNXTensorElementDataType mel_type_;
  Lfm2AudioMelConfig mel_config_;
  int64_t subsampling_factor_{};
  int32_t audio_token_id_{};
};

}  // namespace Generators

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "models/preprocessing/processor.h"

namespace Generators {

// Placeholder the prompt carries once per audio clip, in clip order. The checkpoint's chat template
// writes <|audio_start|><|audio_pad|><|audio_end|> for each clip, and the reference processor
// (transformers' Qwen3ASRProcessor) repeats <|audio_pad|> once per encoder output token. This
// processor does the same at the token level: model.audio_token_id once per token of that clip.
inline constexpr char kQwen3AsrAudioMarker[] = "<|audio_pad|>";

// The encoder runs its convolutions on chunks of 2 * n_window = 100 mel frames, so its mel input is
// always a whole number of chunks (the speech state slices clips on this grid too).
inline constexpr int64_t kQwen3AsrChunkFrames = 100;

// Log-mel front end of Qwen3-ASR, transformers' Qwen3ASRFeatureExtractor as Qwen3ASRProcessor
// calls it (padding=True, so a single clip is not padded to 30 s), with the checkpoint's settings
// (processor_config.json "feature_extractor").
struct Qwen3AsrMelConfig {
  int num_mels{128};
  int fft_size{400};  // also the window length: a periodic Hann window of fft_size samples
  int hop_length{160};
  int sample_rate{16000};
  int64_t min_samples{8000};  // shorter clips are zero-padded to this length, and the padding counts as audio
  int64_t chunk_frames{kQwen3AsrChunkFrames};
  float log_floor{1e-10f};    // clamp before log10
  float dynamic_range{8.0f};  // log10 values below the clip's maximum minus this are raised to it
};

// Valid log-mel frames of a clip of num_samples samples: torch.stft(center=True) yields one frame
// per hop plus one, and the feature extractor drops the last one.
int64_t Qwen3AsrNumValidMelFrames(int64_t num_samples, const Qwen3AsrMelConfig& config);

// Decoder tokens a clip of valid_frames log-mel frames contributes. The encoder runs three
// stride-2 convolutions (kernel 3, padding 1) on each chunk of chunk_frames frames, so a full chunk
// gives 13 tokens and the remainder r gives ceil(ceil(ceil(r / 2) / 2) / 2)
// (transformers' _get_feat_extract_output_lengths).
int64_t Qwen3AsrNumTokens(int64_t valid_frames, int64_t chunk_frames);

// Computes the log-mel spectrogram of one mono clip at config.sample_rate, time-major
// [num_frames, num_mels] (the layout of the encoder's mel_spectrogram input). The first
// valid_frames frames are the clip's; zero frames follow up to num_frames, the next multiple of
// chunk_frames, as the feature extractor pads them.
std::vector<float> ComputeQwen3AsrMel(const float* pcm, int64_t num_samples, const Qwen3AsrMelConfig& config,
                                      int64_t& valid_frames, int64_t& num_frames);

// Splits the prompt at each marker. Throws unless the prompt holds exactly one marker per clip.
// Returns num_clips + 1 text segments.
std::vector<std::string> SplitQwen3AsrPrompt(const std::string& prompt, size_t num_clips);

struct Qwen3AsrProcessor : Processor {
  Qwen3AsrProcessor(Config& config, const SessionInfo& session_info);

  std::unique_ptr<NamedTensors> Process(const Tokenizer& tokenizer, const Payload& payload) const override;

 private:
  struct ClipMel {
    std::vector<float> mel;  // num_frames * num_mels, frame-major
    int64_t valid_frames{};  // the clip's own frames: the encoder's mel_lengths
    int64_t num_frames{};    // valid_frames rounded up to a multiple of chunk_frames
    int64_t num_tokens{};    // placeholders the clip is worth
  };

  // Decodes clip `index` to mono PCM at the encoder's sample rate and turns it into log-mel frames.
  ClipMel ComputeClipMel(const Audios& audios, size_t index) const;

  // The prompt's text tokenized segment by segment, with each clip's placeholders spliced between.
  std::vector<int32_t> MakePromptTokens(const Tokenizer& tokenizer, const std::vector<std::string>& segments,
                                        const std::vector<ClipMel>& clips) const;

  // Stages the clips in one zero-padded [num_clips, longest, num_mels] tensor, cast to the speech
  // model's input type, and adds it with the per-clip valid frame and placeholder counts.
  void AddAudioTensors(const std::vector<ClipMel>& clips, Ort::Allocator& allocator, NamedTensors& named_tensors) const;

  ONNXTensorElementDataType mel_type_;
  Qwen3AsrMelConfig mel_config_;
  int32_t audio_token_id_{};
};

}  // namespace Generators

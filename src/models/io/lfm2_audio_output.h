// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <vector>

#include "models/model.h"

namespace Generators {

struct MultiModalLanguageModel;

// Speech output of LFM2-Audio, as liquid-audio's generate_sequential and generate_interleaved
// produce it.
//
// The decoder emits one hidden state per step. While the answer is text the search samples a token
// from the logits as usual; while it is speech the depthformer turns that hidden state into one
// audio frame of num_codebooks codes instead, and the sum of the codes' embeddings is the decoder's
// next input. The generator's sequence records each frame as one model.audio_token_id placeholder,
// so lengths and stopping work unchanged, and the codes themselves are read with
// generator.get_output("audio_codes").
//
// The turn passes to speech on <|audio_start|> and back on the end-of-audio code. With
// search.audio_interleaved it also alternates by count, interleaved_n_text tokens against
// interleaved_n_audio frames, until <|text_end|> leaves the rest of the turn to speech.
struct Lfm2AudioOutput {
  Lfm2AudioOutput(const MultiModalLanguageModel& model, const GeneratorParams& params);

  // Called at the top of every pipeline run with the tokens the search just chose.
  void BeginStep(DeviceSpan<int32_t>& next_tokens, bool is_prompt);

  // True when the decoder's next input is the last audio frame rather than a token.
  bool HasPendingFrame() const { return !pending_embedding_.empty(); }
  // Overwrites the decoder's [1, 1, hidden_size] input with the last frame's embedding.
  void WritePendingFrame(OrtValue& inputs_embeds);

  bool InAudio() const { return modality_ == Modality::Audio; }
  // Samples the next frame from the decoder's last hidden state and returns logits that make the
  // search pick the placeholder token.
  DeviceSpan<float> SampleFrame(OrtValue& hidden_states);

  // Every frame so far, [num_frames, num_codebooks] int64, end-of-audio frames left out.
  OrtValue* GetAudioCodes();

 private:
  enum struct Modality { Text,
                         Audio };

  std::vector<int64_t> RunDepthformer(std::span<const float> hidden);
  int64_t SampleCode(std::span<const float> logits);
  void EmbedFrame(std::span<const int64_t> frame);

  const MultiModalLanguageModel& model_;
  const Config::Model::AudioOutput& config_;
  const bool interleaved_;
  const float temperature_;
  const int top_k_;
  const int32_t placeholder_token_id_;
  std::mt19937 rng_;

  // Depthformer geometry, read off the graph.
  int64_t hidden_size_{};
  int64_t depth_size_{};
  int64_t num_layers_{};
  int64_t num_kv_heads_{};
  int64_t head_size_{};

  Modality modality_{Modality::Text};
  bool previous_was_text_{};
  bool text_done_{};
  int modality_left_{};

  std::vector<float> pending_embedding_;  // the last frame, as the decoder's next input
  std::vector<int64_t> audio_codes_;      // num_frames * num_codebooks
  std::unique_ptr<OrtValue> audio_codes_tensor_;
  DeviceSpan<float> placeholder_logits_;
};

}  // namespace Generators

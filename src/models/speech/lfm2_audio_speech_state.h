// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/speech/multi_modal_speech.h"

namespace Generators {

// lfm2_audio exchanges tensors with its embedding and speech sessions in buffers allocated on the
// decoder's devices. Throws if either has session_options of its own that leave it on CPU while a
// buffer it would be handed is device memory, which the session would treat as host memory and
// corrupt: the decoder's inputs for the embedding session, and with_audio, the audio features for
// both.
void CheckLfm2AudioSessionDevices(const Config& config, DeviceType decoder_device, DeviceType inputs_device,
                                  bool with_audio);

// Lfm2AudioSpeechState: per-clip encoder loop for LFM2-Audio.
//
// The processor stacks the clips of one prompt into a zero-padded [N, T_max, num_mels] mel tensor
// with their real frame counts alongside. The published encoder export is traced for a single clip
// (its subsampling mask cannot broadcast over a batch), so with several clips this subclass slices
// each clip's own frames out of that tensor, runs the encoder on [1, T_i, num_mels], and writes the
// results one after another into the contiguous [1, total_tokens, hidden] feature buffer the
// embedding model expects, in clip order.
struct Lfm2AudioSpeechState : SpeechState {
  using SpeechState::SpeechState;  // inherit constructor

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs, const int64_t num_audio_tokens) override;
  void ReuseFeaturesBuffer(MultiModalFeatures& embedding_features) override;
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  // Where the processor's staged mel tensor and the encoder's feature buffer are bound.
  struct SpeechBindings {
    size_t mel_index{};       // the [num_clips, longest, num_mels] staging tensor
    size_t lengths_index{};   // its per-clip frame counts
    size_t features_index{};  // the [1, total_tokens, hidden] buffer the embedding model reads
    int64_t num_clips{};
    int64_t longest_clip{};
    int64_t num_mels{};
    int64_t hidden_size{};
    ONNXTensorElementDataType mel_type{};
    ONNXTensorElementDataType features_type{};
  };

  // Reads the bound shapes and checks them against the per-clip token counts.
  SpeechBindings ResolveBindings() const;

  // Runs the encoder on clip `index`'s own frames and returns its features.
  std::unique_ptr<OrtValue> RunClip(const SpeechBindings& bindings, int64_t index, int64_t num_frames);

  size_t FindInput(const std::string& name) const;
  size_t FindOutput(const std::string& name) const;

  std::vector<int64_t> tokens_per_clip_;
};

}  // namespace Generators

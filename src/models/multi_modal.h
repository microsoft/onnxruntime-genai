// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "models/io/extra_inputs.h"
#include "model_type.h"
#include "models/vision/multi_modal_vision.h"
#include "models/vision/qwen_vision_state.h"
#include "models/vision/pixtral_vision_state.h"
#include "models/speech/multi_modal_speech.h"
#include "models/speech/lfm2_audio_speech_state.h"
#include "models/embedding/multi_modal_embedding.h"
#include "models/multi_modal_decoder.h"
#include "models/gemma4_decoder_state.h"

namespace Generators {

struct Lfm2AudioOutput;

struct MultiModalLanguageModel : Model {
  MultiModalLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech);
  MultiModalLanguageModel(const MultiModalLanguageModel&) = delete;
  MultiModalLanguageModel& operator=(const MultiModalLanguageModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> vision_session_;     // pixel_values, [image_attention_mask], image_sizes -> image_features
  std::unique_ptr<OrtSession> speech_session_;     // audio_embeds, audio_sizes, audio_projection_mode -> audio_features
  std::unique_ptr<OrtSession> embedding_session_;  // input_ids, image_features, audio_features -> inputs_embeds
  std::unique_ptr<OrtSession> decoder_session_;    // inputs_embeds, attention_mask, kv_cache -> logits

  std::unique_ptr<OrtSessionOptions> vision_session_options_;
  std::unique_ptr<OrtSessionOptions> speech_session_options_;
  std::unique_ptr<OrtSessionOptions> embedding_session_options_;

  // LFM2-Audio speech output, present when model.audio_output names the two graphs.
  std::unique_ptr<OrtSession> depthformer_session_;      // hidden_states -> one audio code per run, a frame in num_codebooks runs
  std::unique_ptr<OrtSession> audio_embedding_session_;  // audio_codes -> audio_embeds, summed into the decoder's next input
  std::unique_ptr<OrtSessionOptions> depthformer_session_options_;
  std::unique_ptr<OrtSessionOptions> audio_embedding_session_options_;
};

struct MultiModalPipelineState : State {
  MultiModalPipelineState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                          const GeneratorParams& params);
  MultiModalPipelineState(const MultiModalPipelineState&) = delete;
  MultiModalPipelineState& operator=(const MultiModalPipelineState&) = delete;

  void SetExtraInputs(const std::vector<ExtraInput>& extra_inputs) override;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  OrtValue* GetInput(const char* name) override;

  OrtValue* GetOutput(const char* name) override;

 private:
  void UpdateInputsOutputs(const DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int current_length);
  // The decoder's logits while the answer is text, or the next audio frame's placeholder while it is speech.
  DeviceSpan<float> SampleAudioOrText(DeviceSpan<float> logits);

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_{};
  int64_t num_audio_tokens_{};
  int64_t num_images_{};
  std::unique_ptr<VisionState> vision_state_;
  std::unique_ptr<SpeechState> speech_state_;
  std::unique_ptr<EmbeddingState> embedding_state_;
  std::unique_ptr<DecoderState> decoder_state_;
  std::unique_ptr<Lfm2AudioOutput> audio_output_;  // LFM2-Audio speech output, when the model has it
  std::shared_ptr<Adapters> adapters_;
  bool is_prompt_{true};

  const std::string vision_adapter_name_{"vision"};
  const std::string speech_adapter_name_{"speech"};
};

}  // namespace Generators

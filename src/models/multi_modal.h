// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "model.h"
#include "input_ids.h"
#include "multi_modal_features.h"
#include "embeddings.h"
#include "extra_inputs.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"

namespace Generators {

int64_t GetNumImageTokens(const std::vector<GeneratorParams::Input>& extra_inputs,
                          const std::string& pixel_values_name,
                          const std::string& image_sizes_name);

int64_t GetNumAudioTokens(const std::vector<GeneratorParams::Input>& extra_inputs,
                          const std::string& audio_embeds_name,
                          const std::string& audio_sizes_name);

struct MultiModalLanguageModel : Model {
  MultiModalLanguageModel(std::unique_ptr<Config> config, OrtEnv& ort_env, bool vision, bool speech);
  MultiModalLanguageModel(const MultiModalLanguageModel&) = delete;
  MultiModalLanguageModel& operator=(const MultiModalLanguageModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const;

  std::unique_ptr<OrtSession> vision_session_;     // pixel_values, [image_attention_mask], image_sizes -> image_features
  std::unique_ptr<OrtSession> speech_session_;     // audio_embeds, audio_sizes, audio_projection_mode -> audio_features
  std::unique_ptr<OrtSession> embedding_session_;  // input_ids, image_features, audio_features -> inputs_embeds
  std::unique_ptr<OrtSession> decoder_session_;    // inputs_embeds, attention_mask, kv_cache -> logits
};

struct VisionState : State {
  VisionState(const MultiModalLanguageModel& model, const GeneratorParams& params, const int64_t num_image_tokens);
  VisionState(const VisionState&) = delete;
  VisionState& operator=(const VisionState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  friend struct MultiModalPipelineState;

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_;
  ExtraInputs extra_inputs_{*this};                                                        // Model inputs
  MultiModalFeatures image_features_{*this, MultiModalFeatures::Mode::Output,              // Model output
                                     model_.config_->model.vision.outputs.image_features,
                                     num_image_tokens_};
};

struct SpeechState : State {
  SpeechState(const MultiModalLanguageModel& model, const GeneratorParams& params, const int64_t num_image_tokens);
  SpeechState(const SpeechState&) = delete;
  SpeechState& operator=(const SpeechState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {}) override;

 private:
  friend struct MultiModalPipelineState;

  const MultiModalLanguageModel& model_;
  int64_t num_audio_tokens_;
  ExtraInputs extra_inputs_{*this};                                                        // Model inputs
  MultiModalFeatures audio_features_{*this, MultiModalFeatures::Mode::Output,              // Model output
                                     model_.config_->model.speech.outputs.audio_features,
                                     num_audio_tokens_};
};

struct EmbeddingState : State {
  EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params, const int64_t num_image_tokens, const int64_t num_audio_tokens);
  EmbeddingState(const EmbeddingState&) = delete;
  EmbeddingState& operator=(const EmbeddingState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {});

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt);

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_;
  int64_t num_audio_tokens_;

  DefaultInputIDs input_ids_{*this};                                                         // Model input
  MultiModalFeatures image_features_{*this, MultiModalFeatures::Mode::Input,                 // Optional model input
                                     model_.config_->model.embedding.inputs.image_features,
                                     num_image_tokens_};
  MultiModalFeatures audio_features_{*this, MultiModalFeatures::Mode::Input,                 // Optional model input
                                     model_.config_->model.embedding.inputs.audio_features,
                                     num_audio_tokens_};
  Embeddings inputs_embeds_{*this, Embeddings::Mode::Output,                                 // Model output
                            model_.config_->model.embedding.outputs.embeddings};
};

struct DecoderState : State {
  DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
               const GeneratorParams& params, const CapturedGraphInfo* captured_graph_info);
  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;

  const CapturedGraphInfo* GetCapturedGraphInfo() const override { return captured_graph_info_; };

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices);

  const MultiModalLanguageModel& model_;
  const CapturedGraphInfo* captured_graph_info_;
  Embeddings inputs_embeds_{*this, Embeddings::Mode::Input,                    // Model input
                            model_.config_->model.decoder.inputs.embeddings};
  DefaultPositionInputs position_inputs_;                                      // Model input
  DefaultKeyValueCache kv_cache_{*this};                                       // Model input
  Logits logits_{*this};                                                       // Model output
};

struct MultiModalPipelineState : State {
  MultiModalPipelineState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                          const GeneratorParams& params);
  MultiModalPipelineState(const MultiModalPipelineState&) = delete;
  MultiModalPipelineState& operator=(const MultiModalPipelineState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices);
  OrtValue* GetInput(const char* name);
  OrtValue* GetOutput(const char* name);

 private:
  void UpdateInputsOutputs(const DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int current_length);

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_{};
  int64_t num_audio_tokens_{};
  const CapturedGraphInfoPtr captured_graph_info_;
  std::unique_ptr<VisionState> vision_state_;
  std::unique_ptr<SpeechState> speech_state_;
  std::unique_ptr<EmbeddingState> embedding_state_;
  std::unique_ptr<DecoderState> decoder_state_;
  bool is_prompt_{true};
};

}  // namespace Generators

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "model.h"
#include "input_ids.h"
#include "image_features.h"
#include "embeddings.h"
#include "extra_inputs.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"

namespace Generators {

struct MultiModalVisionModel : Model {
  MultiModalVisionModel(std::unique_ptr<Config> config, OrtEnv& ort_env);
  MultiModalVisionModel(const MultiModalVisionModel&) = delete;
  MultiModalVisionModel& operator=(const MultiModalVisionModel&) = delete;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> vision_session_;     // pixel_values, image_sizes -> image_features
  std::unique_ptr<OrtSession> embedding_session_;  // input_ids, image_features -> inputs_embeds
  std::unique_ptr<OrtSession> decoder_session_;    // inputs_embeds, attention_mask, kv_cache -> logits
};

struct EmbeddingState : State {
  EmbeddingState(const MultiModalVisionModel& model, const GeneratorParams& params, const int64_t num_image_tokens);
  EmbeddingState(const EmbeddingState&) = delete;
  EmbeddingState& operator=(const EmbeddingState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt);

  const MultiModalVisionModel& model_;
  int64_t num_image_tokens_;

  DefaultInputIDs input_ids_{*this};                                // Model input
  ImageFeatures image_features_{*this, ImageFeatures::Mode::Input,  // Optional model input
                                model_.config_->model.embedding.inputs.image_features,
                                num_image_tokens_};
  Embeddings inputs_embeds_{*this, Embeddings::Mode::Output,  // Model output
                            model_.config_->model.embedding.outputs.embeddings};
};

struct VisionState : State {
  VisionState(const MultiModalVisionModel& model, const GeneratorParams& params, const int64_t num_image_tokens);
  VisionState(const VisionState&) = delete;
  VisionState& operator=(const VisionState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices = {}) override;

 private:
  friend struct MultiModalPipelineState;

  const MultiModalVisionModel& model_;
  int64_t num_image_tokens_;
  ExtraInputs extra_inputs_{*this};                                  // Model inputs
  ImageFeatures image_features_{*this, ImageFeatures::Mode::Output,  // Model output
                                model_.config_->model.vision.outputs.image_features,
                                num_image_tokens_};
};

struct DecoderState : State {
  DecoderState(const MultiModalVisionModel& model, DeviceSpan<int32_t> sequence_lengths,
               const GeneratorParams& params, const CapturedGraphInfo* captured_graph_info);
  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

  const CapturedGraphInfo* GetCapturedGraphInfo() const override { return captured_graph_info_; };

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices);

  const MultiModalVisionModel& model_;
  const CapturedGraphInfo* captured_graph_info_;
  Embeddings inputs_embeds_{*this, Embeddings::Mode::Input,  // Model input
                            model_.config_->model.decoder.inputs.embeddings};
  DefaultPositionInputs position_inputs_;  // Model input
  DefaultKeyValueCache kv_cache_{*this};   // Model input
  Logits logits_{*this};                   // Model output
};

struct MultiModalPipelineState : State {
  MultiModalPipelineState(const MultiModalVisionModel& model, DeviceSpan<int32_t> sequence_lengths,
                          const GeneratorParams& params);
  MultiModalPipelineState(const MultiModalPipelineState&) = delete;
  MultiModalPipelineState& operator=(const MultiModalPipelineState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens,
                        DeviceSpan<int32_t> next_indices) override;

 private:
  void UpdateInputsOutputs(const DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices,
                           int current_length);

  const MultiModalVisionModel& model_;
  int64_t num_image_tokens_{};
  const CapturedGraphInfoPtr captured_graph_info_;
  std::unique_ptr<EmbeddingState> embedding_state_;
  std::unique_ptr<VisionState> vision_state_;
  std::unique_ptr<DecoderState> decoder_state_;
  bool is_prompt_{true};
};

}  // namespace Generators

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "model.h"
#include "input_ids.h"
#include "embeddings.h"
#include "extra_inputs.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"

namespace Generators {

struct MultiModalVisionModel : Model {
  MultiModalVisionModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> embedding_session_;  // input_ids -> inputs_embeds
  std::unique_ptr<OrtSession> vision_session_;     // pixel_values, img_sizes -> visual_features
  std::unique_ptr<OrtSession> decoder_session_;    // inputs_embeds, attention_mask, kv_cache -> logits
};

struct EmbeddingState : State {
  EmbeddingState(const MultiModalVisionModel& model, const GeneratorParams& params, const CapturedGraphInfo* captured_graph_info);

  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens,
                          RoamingArray<int32_t> next_indices = {}) override;

  const CapturedGraphInfo* GetCapturedGraphInfo() const override { return captured_graph_info_; };

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputsAndOutputs(RoamingArray<int32_t> next_tokens);

  const MultiModalVisionModel& model_;
  const CapturedGraphInfo* captured_graph_info_;
  InputIDs input_ids_{model_, *this};                                 // Model input
  Embeddings inputs_embeds_{model_, *this, Embeddings::Mode::Output,  // Model output
                            model_.config_->model.embedding.outputs.embeddings};
};

struct VisionState : State {
  VisionState(const MultiModalVisionModel& model, const GeneratorParams& params);

  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens,
                          RoamingArray<int32_t> next_indices = {}) override;

 private:
  friend struct MultiModalPipelineState;

  const MultiModalVisionModel& model_;
  ExtraInputs extra_inputs_{model_, *this};    // Model inputs
  std::unique_ptr<OrtValue> visual_features_;  // Model output
  int32_t num_image_tokens_{};
};

struct DecoderState : State {
  DecoderState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths,
               const GeneratorParams& params, const CapturedGraphInfo* captured_graph_info);

  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens,
                          RoamingArray<int32_t> next_indices) override;

  const CapturedGraphInfo* GetCapturedGraphInfo() const override { return captured_graph_info_; };

 private:
  friend struct MultiModalPipelineState;

  void UpdateInputs(int current_length, RoamingArray<int32_t> beam_indices);

  const MultiModalVisionModel& model_;
  const CapturedGraphInfo* captured_graph_info_;
  Embeddings inputs_embeds_{model_, *this, Embeddings::Mode::Input,  // Model input
                            model_.config_->model.decoder.inputs.embeddings};
  PositionInputs position_inputs_;    // Model input
  KV_Cache kv_cache_{model_, *this};  // Model input
  Logits logits_{model_, *this};      // Model output
};

struct MultiModalPipelineState : State {
  MultiModalPipelineState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths,
                          const GeneratorParams& params);

  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens,
                          RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices,
                    int current_length);

  const MultiModalVisionModel& model_;
  const CapturedGraphInfoPtr captured_graph_info_;
  std::unique_ptr<EmbeddingState> embedding_state_;
  std::unique_ptr<VisionState> vision_state_;
  std::unique_ptr<DecoderState> decoder_state_;
  bool is_prompt_{true};
};

}  // namespace Generators

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>

#include "models/model.h"
#include "models/io/input_ids.h"
#include "models/io/multi_modal_features.h"
#include "models/io/embeddings.h"

namespace Generators {

struct MultiModalLanguageModel;
struct MultiModalPipelineState;
struct DecoderState;

struct EmbeddingState : State {
  EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params);
  EmbeddingState(const EmbeddingState&) = delete;
  EmbeddingState& operator=(const EmbeddingState&) = delete;

  void SetExtraInputs(const int64_t num_images_, const int64_t num_image_tokens_, const int64_t num_audio_tokens_);
  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices = {});

  // Hands the embedding buffers this state produced off to the decoder for the next run, so the
  // decoder can consume them without an extra device copy. Subclasses that carry additional
  // per-model outputs (e.g. Gemma4's per_layer_inputs) override this to also hand those off.
  virtual void ReuseBuffersInDecoder(DecoderState& decoder);

 protected:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, bool is_prompt);

  const MultiModalLanguageModel& model_;
  int64_t num_image_tokens_;
  int64_t num_audio_tokens_;

  DefaultInputIDs input_ids_{*this};                          // Model input
  std::unique_ptr<MultiModalFeatures> image_features_;        // Optional model input
  std::unique_ptr<MultiModalFeatures> audio_features_;        // Optional model input
  Embeddings inputs_embeds_;  // Model output
};

// Factory: pick the right EmbeddingState subclass based on model configuration.
std::unique_ptr<EmbeddingState> CreateEmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params);

}  // namespace Generators

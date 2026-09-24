// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <memory>

#include "model.h"
#include "models/io/input_ids.h"
#include "models/io/embeddings.h"
#include "models/io/logits.h"
#include "io/kv_cache.h"
#include "models/io/position_inputs.h"
#include "models/io/recurrent_state.h"

namespace Generators {

struct MultiModalLanguageModel;
struct MultiModalPipelineState;

struct DecoderState : State {
  DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
               const GeneratorParams& params);
  DecoderState(const DecoderState&) = delete;
  DecoderState& operator=(const DecoderState&) = delete;

  DeviceSpan<float> Run(int current_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices);

  // Prefill chunking (see search.chunk_size). The embedding model still runs once over the whole
  // prompt (it is a lookup/projection), while the decoder prefill is split into several runs so the
  // peak attention workspace scales with the chunk size instead of the full prompt length.
  bool SupportsPrefillChunking(bool has_multimodal_content) const;
  void PrepareEmbeddingsForPrefill(size_t new_length);
  DeviceSpan<float> RunPrefillWithChunking(int current_length, DeviceSpan<int32_t>& next_tokens,
                                           DeviceSpan<int32_t> next_indices, size_t chunk_size);

  // Accessors used by EmbeddingState/MultiModalPipelineState across translation units, replacing
  // direct friend access to private members with a stable, subclass-overridable interface.
  Embeddings& GetInputsEmbeds() { return inputs_embeds_; }
  PositionInputs& GetPositionInputs() { return *position_inputs_; }

  // Optional per-layer conditioning input (Gemma4). Base returns nullptr; the Gemma4-specific
  // subclass overrides this to expose its per_layer_inputs_ so EmbeddingState can hand its own
  // per_layer_inputs output off to the decoder alongside inputs_embeds.
  virtual Embeddings* GetPerLayerInputs() { return nullptr; }

 protected:
  friend struct MultiModalPipelineState;

  void UpdateInputsOutputs(DeviceSpan<int32_t>& next_tokens, int current_length, DeviceSpan<int32_t> beam_indices, size_t new_length);

  // Extension points for optional per-model decoder inputs that must track inputs_embeds_'s
  // sequence-length/chunk-view changes (e.g. Gemma4's per_layer_inputs_). Base is a no-op.
  virtual void UpdateExtraSequenceLength(size_t new_length) {}
  virtual void UseExtraChunkView(size_t offset, size_t count) {}
  virtual void RestoreExtraFullView() {}

  const MultiModalLanguageModel& model_;
  Embeddings inputs_embeds_{*this, Embeddings::Mode::Input,  // Model input
                            model_.config_->model.decoder.inputs.embeddings};
  std::unique_ptr<DefaultInputIDs> decoder_input_ids_;  // Optional model input (e.g., Gemma4 decoder needs input_ids)
  std::unique_ptr<PositionInputs> position_inputs_;     // Model input
  std::unique_ptr<KeyValueCache> kv_cache_;             // Model input
  std::unique_ptr<RecurrentState> recurrent_state_;     // Model input (for hybrid models)
  Logits logits_{*this};                                // Model output
};

// Factory: pick the right DecoderState subclass based on model configuration.
std::unique_ptr<DecoderState> CreateDecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                                                 const GeneratorParams& params);

}  // namespace Generators

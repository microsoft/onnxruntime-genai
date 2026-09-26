// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/multi_modal_decoder.h"

namespace Generators {

// Gemma4DecoderState: alongside inputs_embeds, Gemma4's decoder also accepts per_layer_inputs from
// the embedding model, and its sequence-length/chunk-view bookkeeping must track inputs_embeds_'s.
struct Gemma4DecoderState : DecoderState {
  Gemma4DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                     const GeneratorParams& params);

  Embeddings* GetPerLayerInputs() override { return per_layer_inputs_.get(); }

 protected:
  void UpdateExtraSequenceLength(size_t new_length) override;
  void UseExtraChunkView(size_t offset, size_t count) override;
  void RestoreExtraFullView() override;

 private:
  std::unique_ptr<Embeddings> per_layer_inputs_;  // Optional model input (Gemma4: per-layer conditioning)
};

}  // namespace Generators

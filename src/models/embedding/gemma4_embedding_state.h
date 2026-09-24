// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/embedding/multi_modal_embedding.h"

namespace Generators {

struct Gemma4EmbeddingState : EmbeddingState {
  Gemma4EmbeddingState(const MultiModalLanguageModel& model, const GeneratorParams& params);

  void SetExtraInputs(int64_t num_images, int64_t num_image_tokens, int64_t num_audio_tokens) override;
  void ReuseBuffersInDecoder(DecoderState& decoder) override;

 private:
  std::unique_ptr<Embeddings> per_layer_inputs_;
};

}  // namespace Generators

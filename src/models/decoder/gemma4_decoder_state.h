// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "models/decoder/multi_modal_decoder.h"

namespace Generators {

struct Gemma4DecoderState : DecoderState {
  Gemma4DecoderState(const MultiModalLanguageModel& model, DeviceSpan<int32_t> sequence_lengths,
                     const GeneratorParams& params);

  Embeddings* GetPerLayerInputs() override { return per_layer_inputs_.get(); }

 protected:
  void UpdateExtraSequenceLength(size_t new_length) override;
  void UseExtraChunkView(size_t offset, size_t count) override;
  void RestoreExtraFullView() override;

 private:
  std::unique_ptr<Embeddings> per_layer_inputs_;
};

}  // namespace Generators

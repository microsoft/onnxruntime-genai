// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "onnxruntime_api.h"

namespace Generators {

struct ExtraInput;
struct GeneratorParams;
struct OpenVINOMultiModalModel;

// Host-side vision production + image-feature merge strategy for an OpenVINO-partitioned VLM.
struct OpenVINOVisionMerger {
  virtual ~OpenVINOVisionMerger() = default;

  // Runs the model's vision graph(s) for the current prompt. Called once, from SetExtraInputs. A no-op
  // when the prompt contains no image.
  virtual void RunVision(const std::vector<ExtraInput>& extra_inputs) = 0;

  // Scatters the produced image features into the (already computed, text-only) embeddings buffer at
  // this model's image placeholder token rows. Called once, from Run, after the embedding step.
  virtual void Merge(OrtValue& inputs_embeds, const OrtValue& input_ids) = 0;
};

// Factory keyed on model.config_->model.type. Throws if the model type has no registered OpenVINO merger.
std::unique_ptr<OpenVINOVisionMerger> CreateOpenVINOVisionMerger(const OpenVINOMultiModalModel& model,
                                                                 const GeneratorParams& params);

}  // namespace Generators

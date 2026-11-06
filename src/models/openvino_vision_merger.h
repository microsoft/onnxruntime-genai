// Copyright (c) Intel® Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "onnxruntime_api.h"

namespace Generators {

struct ExtraInput;
struct GeneratorParams;
struct OpenVINOMultiModalModel;
template <typename T>
struct DeviceSpan;

// Host-side vision production + image-feature merge strategy for an OpenVINO-partitioned VLM.
struct OpenVINOVisionMerger {
  virtual ~OpenVINOVisionMerger() = default;

  // Runs the model's vision graph(s) for the current prompt. Called once, from SetExtraInputs. A no-op
  // when the prompt contains no image.
  virtual void RunVision(const std::vector<ExtraInput>& extra_inputs) = 0;

  // Scatters the produced image features into the (already computed, text-only) embeddings buffer at
  // this model's image placeholder token rows. Called once, from Run, after the embedding step.
  // input_ids are the canonical prompt tokens, which may live on a different device than the embeddings.
  virtual void Merge(OrtValue& inputs_embeds, DeviceSpan<int32_t> input_ids) = 0;
};

// Scatters image feature rows into an embeddings buffer on the host, at the given token positions.
size_t MergeImageFeaturesIntoEmbeddings(OrtValue& inputs_embeds,
                                        const OrtValue& image_features,
                                        std::span<const int64_t> target_token_rows);

// Factory keyed on model.config_->model.type. Throws if the model type has no registered OpenVINO merger.
std::unique_ptr<OpenVINOVisionMerger> CreateOpenVINOVisionMerger(const OpenVINOMultiModalModel& model,
                                                                 const GeneratorParams& params);

}  // namespace Generators

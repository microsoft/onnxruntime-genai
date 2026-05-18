// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "v1_translator.h"
#include "pipeline_presets.h"
#include "config.h"
#include "models/model_type.h"

namespace Generators {

namespace {
// Propagate KV cache name patterns from v1 decoder inputs/outputs to pipeline state.
void PropagateKVCachePatterns(PipelineConfig& config, const Config& v1) {
  const auto& inputs = v1.model.decoder.inputs;
  const auto& outputs = v1.model.decoder.outputs;
  if (!inputs.past_key_names.empty()) {
    config.state.kv_cache.past_key_pattern = inputs.past_key_names;
  }
  if (!inputs.past_value_names.empty()) {
    config.state.kv_cache.past_value_pattern = inputs.past_value_names;
  }
  if (!outputs.present_key_names.empty()) {
    config.state.kv_cache.present_key_pattern = outputs.present_key_names;
  }
  if (!outputs.present_value_names.empty()) {
    config.state.kv_cache.present_value_pattern = outputs.present_value_names;
  }
}
}  // namespace

PipelineConfig TranslateV1Config(const Config& v1) {
  const auto& type = v1.model.type;

  PipelineConfig config;

  if (ModelType::IsLLM(type) || ModelType::IsPipe(type)) {
    // Autoregressive decoder: single session
    config = GetPreset("autoregressive-decoder");

    // Override session filename from v1 config
    if (!v1.model.decoder.filename.empty()) {
      config.sessions["decoder"].file = v1.model.decoder.filename;
    }

    PropagateKVCachePatterns(config, v1);

  } else if (ModelType::IsVLM(type) || ModelType::IsMMM(type)) {
    // Vision-language model: vision + embedding + decoder sessions
    config = GetPreset("vision-language");

    if (!v1.model.decoder.filename.empty()) {
      config.sessions["decoder"].file = v1.model.decoder.filename;
    }
    if (!v1.model.vision.filename.empty()) {
      config.sessions["vision"].file = v1.model.vision.filename;
    }
    if (!v1.model.embedding.filename.empty()) {
      config.sessions["embedding"].file = v1.model.embedding.filename;
    }

    // VLM-specific: propagate vision I/O name overrides via dataflow
    // Find the vision→embedding wire by session names (not hardcoded index)
    const auto& vision_outputs = v1.model.vision.outputs;
    const auto& embed_inputs = v1.model.embedding.inputs;
    for (auto& wire : config.dataflow) {
      if (wire.from_session == "vision" && wire.to_session == "embedding") {
        if (!vision_outputs.image_features.empty()) {
          wire.from_output = vision_outputs.image_features;
        }
        if (!embed_inputs.image_features.empty()) {
          wire.to_input = embed_inputs.image_features;
        }
        break;
      }
    }

    PropagateKVCachePatterns(config, v1);

    // Position strategy for Qwen2.5-VL family (3D mRoPE)
    if (ModelType::IsQwenVLFamily(type)) {
      config.state.position_ids.strategy = "mrope_3d";
    }

  } else if (ModelType::IsALM(type)) {
    // Audio-language model (Whisper): encoder-decoder
    config = GetPreset("encoder-decoder");

    if (!v1.model.encoder.filename.empty()) {
      config.sessions["encoder"].file = v1.model.encoder.filename;
    }
    if (!v1.model.decoder.filename.empty()) {
      config.sessions["decoder"].file = v1.model.decoder.filename;
    }

  } else if (type == "marian-ssru") {
    // Marian encoder-decoder
    config = GetPreset("encoder-decoder");

    if (!v1.model.encoder.filename.empty()) {
      config.sessions["encoder"].file = v1.model.encoder.filename;
    }
    if (!v1.model.decoder.filename.empty()) {
      config.sessions["decoder"].file = v1.model.decoder.filename;
    }

  } else {
    // Unknown model type — create minimal config with just the decoder
    // so that pipeline_config is always populated, even if incomplete.
    config = GetPreset("autoregressive-decoder");
    if (!v1.model.decoder.filename.empty()) {
      config.sessions["decoder"].file = v1.model.decoder.filename;
    }
  }

  return config;
}

}  // namespace Generators

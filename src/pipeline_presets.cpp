// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "pipeline_presets.h"
#include <stdexcept>

namespace Generators {

namespace {

PipelineConfig MakeAutoRegressiveDecoder() {
  PipelineConfig config;

  // Single decoder session
  config.sessions["decoder"] = PipelineConfig::Session{
      .file = "model.onnx",
      .role = "decoder",
  };

  // Run decoder every step
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "decoder",
      .when = "step",
  });

  // Default KV cache + position ID state
  config.state.kv_cache.format = "auto";
  config.state.position_ids.strategy = "auto";

  return config;
}

PipelineConfig MakeVisionLanguage() {
  PipelineConfig config;

  // Three sessions: vision encoder, embedding projector, text decoder
  config.sessions["vision"] = PipelineConfig::Session{
      .file = "vision_encoder.onnx",
      .role = "vision",
  };
  config.sessions["embedding"] = PipelineConfig::Session{
      .file = "embedding.onnx",
      .role = "embedding",
  };
  config.sessions["decoder"] = PipelineConfig::Session{
      .file = "decoder.onnx",
      .role = "decoder",
  };

  // Flow: vision runs once on init, embedding on init, decoder every step
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "vision",
      .when = "init",
      .loop = "per_image",
  });
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "embedding",
      .when = "init",
  });
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "decoder",
      .when = "step",
  });

  // Wire vision output → embedding input
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "vision",
      .from_output = "image_features",
      .to_session = "embedding",
      .to_input = "image_features",
  });

  // Wire embedding output → decoder input
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "embedding",
      .from_output = "inputs_embeds",
      .to_session = "decoder",
      .to_input = "inputs_embeds",
  });

  config.state.kv_cache.format = "auto";
  config.state.position_ids.strategy = "auto";

  return config;
}

PipelineConfig MakeEncoderDecoder() {
  PipelineConfig config;

  config.sessions["encoder"] = PipelineConfig::Session{
      .file = "encoder.onnx",
      .role = "encoder",
  };
  config.sessions["decoder"] = PipelineConfig::Session{
      .file = "decoder.onnx",
      .role = "decoder",
  };

  // Encoder runs once on init, decoder runs every step
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "encoder",
      .when = "init",
  });
  config.flow.push_back(PipelineConfig::FlowStep{
      .run = "decoder",
      .when = "step",
  });

  // Wire encoder output → decoder cross-attention input
  config.dataflow.push_back(PipelineConfig::DataflowWire{
      .from_session = "encoder",
      .from_output = "encoder_hidden_states",
      .to_session = "decoder",
      .to_input = "encoder_hidden_states",
  });

  config.state.kv_cache.format = "auto";
  config.state.position_ids.strategy = "auto";

  return config;
}

}  // namespace

PipelineConfig GetPreset(const std::string& name) {
  if (name == "autoregressive-decoder") return MakeAutoRegressiveDecoder();
  if (name == "vision-language") return MakeVisionLanguage();
  if (name == "encoder-decoder") return MakeEncoderDecoder();

  throw std::runtime_error(
      "Unknown pipeline preset: '" + name +
      "'. Known presets: 'autoregressive-decoder', 'vision-language', 'encoder-decoder'.");
}

void ApplyOverrides(PipelineConfig& base, const PipelineConfig& overrides) {
  // Sessions: merge — override entries replace base entries with same key,
  // new keys are added
  for (const auto& [key, session] : overrides.sessions) {
    base.sessions[key] = session;
  }

  // Flow: override replaces base entirely if non-empty
  if (!overrides.flow.empty()) {
    base.flow = overrides.flow;
  }

  // Dataflow: override replaces base entirely if non-empty
  if (!overrides.dataflow.empty()) {
    base.dataflow = overrides.dataflow;
  }

  // State: override individual fields only when explicitly set (has_value)
  if (overrides.state.kv_cache.format.has_value()) {
    base.state.kv_cache.format = overrides.state.kv_cache.format;
  }
  if (overrides.state.kv_cache.past_key_pattern.has_value()) {
    base.state.kv_cache.past_key_pattern = overrides.state.kv_cache.past_key_pattern;
  }
  if (overrides.state.kv_cache.present_key_pattern.has_value()) {
    base.state.kv_cache.present_key_pattern = overrides.state.kv_cache.present_key_pattern;
  }
  if (overrides.state.kv_cache.past_value_pattern.has_value()) {
    base.state.kv_cache.past_value_pattern = overrides.state.kv_cache.past_value_pattern;
  }
  if (overrides.state.kv_cache.present_value_pattern.has_value()) {
    base.state.kv_cache.present_value_pattern = overrides.state.kv_cache.present_value_pattern;
  }
  if (overrides.state.position_ids.strategy.has_value()) {
    base.state.position_ids.strategy = overrides.state.position_ids.strategy;
  }

  // Generation loop: override if explicitly set
  if (overrides.generation_loop.has_value()) {
    base.generation_loop = overrides.generation_loop;
  }
}

}  // namespace Generators

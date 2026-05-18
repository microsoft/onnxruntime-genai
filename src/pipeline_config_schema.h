// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Pipeline-as-Config v2 schema definitions.
//
// These structs represent the v2 config format where pipeline behavior
// is declared in config JSON rather than hardcoded per model_type.
// A v2 genai_config.json looks like:
//
//   {
//     "version": 2,
//     "pipeline": {
//       "extends": "autoregressive-decoder",
//       "sessions": { "decoder": { "file": "model.onnx" } },
//       "flow": [ { "run": "decoder", "when": "step" } ],
//       "state": { "kv_cache": { "format": "auto" } }
//     },
//     "model": { ... },
//     "search": { ... }
//   }

#pragma once
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace Generators {

struct PipelineConfig {
  // A session corresponds to one ONNX model file.
  struct Session {
    std::string file;   // Filename relative to config directory
    std::string role;   // Semantic role: "decoder", "encoder", "vision", "embedding", "speech"
    // Per-session ORT options (optional; inherits global defaults if absent)
  };

  // A flow step describes when to run a session during generation.
  struct FlowStep {
    std::string run;    // Session name (key into sessions map)
    std::string when{"step"};  // "init", "step", "final"; aliases: "prompt"→"init", "always"→"step", "once"→"init"
    std::string loop;   // "" = run once, "per_image" = loop over images, "batched" = batch all inputs
  };

  // A dataflow wire connects an output tensor from one session to an
  // input tensor of another session.  Uses structured fields (not dot
  // notation) to avoid ambiguity with tensor names that contain dots.
  struct DataflowWire {
    std::string from_session;  // Source session name
    std::string from_output;   // Output tensor name in source session
    std::string to_session;    // Destination session name
    std::string to_input;      // Input tensor name in destination session
  };

  // State configuration controls how KV cache and position IDs are managed.
  struct StateConfig {
    struct KVCache {
      std::optional<std::string> format;  // "auto", "default", "cross", "windowed"; nullopt = use preset default
      std::optional<std::string> past_key_pattern;
      std::optional<std::string> present_key_pattern;
      std::optional<std::string> past_value_pattern;
      std::optional<std::string> present_value_pattern;
    } kv_cache;

    struct PositionIds {
      std::optional<std::string> strategy;  // "auto", "default", "mrope_3d"; nullopt = use preset default
    } position_ids;
  } state;

  std::optional<std::string> extends;  // Preset name to inherit from
  std::map<std::string, Session> sessions;
  std::vector<FlowStep> flow;
  std::vector<DataflowWire> dataflow;
  std::optional<std::string> generation_loop;  // "autoregressive", "single_pass", "denoising"; nullopt = use preset default
};

// Normalize backward-compat aliases in a PipelineConfig:
// "prompt"→"init", "always"→"step", "once"→"init"
void NormalizePipelineConfig(PipelineConfig& config);

// Validate a PipelineConfig for internal consistency:
// - All flow steps reference existing sessions
// - All dataflow wires reference existing sessions
// - Required sessions present based on flow
// - Valid generation_loop value
// Throws std::runtime_error with descriptive message on failure.
void ValidatePipelineConfig(const PipelineConfig& config);

}  // namespace Generators

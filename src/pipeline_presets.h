// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Built-in pipeline presets.
//
// Presets provide default PipelineConfig values for common architectures.
// A v2 config can specify `"extends": "autoregressive-decoder"` to inherit
// the preset's sessions, flow, and state, then override specific fields.

#pragma once
#include "pipeline_config_schema.h"
#include <string>

namespace Generators {

// Resolve a named preset into a PipelineConfig with default values.
// Known presets:
//   "autoregressive-decoder"  — single decoder session, KV cache, always-run flow
//   "vision-language"         — vision + embedding + decoder sessions
//   "encoder-decoder"         — encoder + decoder sessions with cross-attention
//
// Throws std::runtime_error for unknown preset names.
PipelineConfig GetPreset(const std::string& name);

// Apply overrides from |overrides| onto |base|.
// - Scalar fields: override replaces base if set
// - Maps (sessions): merge — override entries replace base entries with same key
// - Vectors (flow, dataflow): override replaces base entirely if non-empty
void ApplyOverrides(PipelineConfig& base, const PipelineConfig& overrides);

}  // namespace Generators

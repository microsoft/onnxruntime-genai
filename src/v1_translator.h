// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// V1 → V2 config translator.
//
// Translates a legacy v1 Config (model_type-based) into a PipelineConfig
// so that downstream code can use a single representation.  This enables
// incremental migration: existing v1 configs still work, and new code
// can read from PipelineConfig regardless of config version.

#pragma once
#include "pipeline_config_schema.h"

namespace Generators {

struct Config;  // Forward declaration

// Translate a v1 Config into a PipelineConfig.
//
// Maps known model categories to presets:
//   LLM model types  → "autoregressive-decoder" preset
//   VLM model types  → "vision-language" preset
//   ALM model types  → "encoder-decoder" preset (Whisper)
//   Encoder-decoder  → "encoder-decoder" preset
//
// Session filenames, input/output name patterns, and state config
// are populated from the v1 Config's decoder/vision/embedding/speech
// sub-structures.
//
// Unknown model types fall back to the "autoregressive-decoder" preset,
// since the majority of models are decoder-only LLMs.
PipelineConfig TranslateV1Config(const Config& v1_config);

}  // namespace Generators

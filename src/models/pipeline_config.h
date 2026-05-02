// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "decoder_only.h"

namespace Generators {

// Pipeline-as-Config model: dispatches based on config version rather
// than model_type strings.
//
// For v2 configs, CreateModel() routes here instead of the string-based
// dispatch chain.  The current (minimal) implementation delegates to
// DecoderOnly_Model for decoder-only models, ensuring identical behavior
// and avoiding code duplication that could drift over time.
//
// Future versions will support multi-session pipelines (VLM, encoder-
// decoder) by reading session layout from the config's pipeline section.
struct PipelineConfigModel : DecoderOnly_Model {
  PipelineConfigModel(std::unique_ptr<Config> config, OrtEnv& ort_env);
};

}  // namespace Generators

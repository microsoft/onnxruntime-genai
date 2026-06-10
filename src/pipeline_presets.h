// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.
#pragma once

// Built-in pipeline presets for the Pipeline-as-Config schema (issue #2114).
//
// A preset is a fully-formed Config::Pipeline skeleton (flow, dataflow, and state defaults) for a
// common model topology. A v2 config selects one via `pipeline.extends` and may override individual
// fields.
//
// `extends` override semantics (intentionally simple and documented here as the canonical rule):
//   - The named preset is used as the BASE.
//   - Any top-level array the config specifies explicitly (`flow`, `dataflow`) REPLACES the preset's
//     array WHOLESALE (no per-element merge). An empty/omitted array in the config keeps the preset's.
//   - For the `state` object, sub-objects the config touches override the preset's corresponding
//     sub-object; untouched sub-objects keep the preset defaults.
//   - `sessions` always come from the config (presets do not invent session file names).
// This "replace, don't merge" rule keeps preset resolution predictable; the issue left the merge vs.
// replace question under-specified, so we pin it down here.

namespace Generators {

// Returns the built-in preset for `name`, or nullptr if no such preset exists.
// Supported names: "autoregressive-decoder", "vision-language", "encoder-decoder", "speech-language".
const Config::Pipeline* GetPipelinePreset(std::string_view name);

}  // namespace Generators

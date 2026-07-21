// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Modifications Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "session_options.h"

#include "../models/session_options.h"
#include "interface.h"

namespace Generators::AMDGPUExecutionProvider {

namespace {

// Emit static-padding hints so the EP pads the prefill token axis to max_length and
// compiles it once, instead of recompiling per prompt length.
void SetStaticPaddingConfig(OrtSessionOptions& session_options, const Config& config) {
  const auto& decoder = config.model.decoder;
  const std::string seq_len = std::to_string(config.search.max_length);
  const std::string pad_inputs =
      decoder.inputs.input_ids + ":1," + decoder.inputs.position_ids + ":1";
  const std::string pad_outputs = decoder.outputs.logits + ":1";

  session_options.AddConfigEntry("ep.migraphx.static_pad_seq", "1");
  session_options.AddConfigEntry("ep.migraphx.static_pad_seq_len", seq_len.c_str());
  session_options.AddConfigEntry("ep.migraphx.static_pad_inputs", pad_inputs.c_str());
  session_options.AddConfigEntry("ep.migraphx.static_pad_outputs", pad_outputs.c_str());

  session_options.AddConfigEntry("ep.migraphx.hip_graph_enable", "1");
}

}  // namespace

DeviceInterface* AppendExecutionProvider(OrtSessionOptions& session_options,
                                         const Config::ProviderOptions& provider_options,
                                         const Config& config,
                                         bool /*disable_graph_capture*/) {
  SetStaticPaddingConfig(session_options, config);

  AppendExecutionProviderV2(session_options, provider_options,
                            DeviceType::AMDGPU, "AMDGPUExecutionProvider");

  return GetAMDGPUInterface();
}

}  // namespace Generators::AMDGPUExecutionProvider

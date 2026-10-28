// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "models/model.h"

namespace Generators {

void ApplyVisionRunOption(OrtRunOptions& options, const char* key, const char* value, size_t stage_index);
std::unique_ptr<OrtRunOptions> CreateVisionStageRunOptions(const Config::Model::Vision& vision, size_t stage_index);

// Preallocate concrete outputs on the encoder device; let ORT allocate dynamic outputs.
std::unique_ptr<OrtValue> RunVisionEncoder(
    OrtSession& session, OrtRunOptions* run_options, const char* const* input_names,
    const OrtValue* const* inputs, size_t input_count, const char* output_name,
    DeviceInterface& encoder_device);

// The returned tensor owns a CPU copy only when the projector cannot consume the encoder output.
std::unique_ptr<OrtValue> StageVisionEncoderOutput(OrtValue& encoded, DeviceInterface& encoder_device,
                                                   DeviceInterface& projector_device);

}  // namespace Generators

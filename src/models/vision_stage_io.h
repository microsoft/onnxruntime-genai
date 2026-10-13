// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"

namespace Generators {

void ApplyVisionRunOption(OrtRunOptions& options, const char* key, const char* value);
std::unique_ptr<OrtRunOptions> CreateVisionStageRunOptions(const Config::Model::Vision& vision, size_t stage_index);

// The returned tensor owns a CPU copy only when the projector cannot consume the encoder output.
std::unique_ptr<OrtValue> StageVisionEncoderOutput(OrtValue& encoded, DeviceInterface& encoder_device,
                                                   DeviceInterface& projector_device);

}  // namespace Generators

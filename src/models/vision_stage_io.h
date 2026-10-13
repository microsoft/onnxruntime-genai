// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"

namespace Generators {

// The returned tensor owns a CPU copy only when the projector cannot consume the encoder output.
std::unique_ptr<OrtValue> StageVisionEncoderOutput(OrtValue& encoded, DeviceInterface& encoder_device,
                                                   DeviceInterface& projector_device);

}  // namespace Generators

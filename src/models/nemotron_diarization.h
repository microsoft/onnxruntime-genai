// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "ort_genai_c.h"
#include <array>

namespace Generators {
struct NemotronSpeechModel;

struct NemotronDiarizationState : State {
  NemotronDiarizationState(const NemotronSpeechModel& model, const GeneratorParams& params);
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t> = {}) override;
  void Process(OrtValue& features, int64_t lookahead);
  OgaTokenMetadataSpeakerEvidence Evidence(int64_t local_frame) const;
  void Reset();

 private:
  const NemotronSpeechModel& model_;
  std::array<std::unique_ptr<OrtValue>, 6> cache_;
  std::unique_ptr<OrtValue> lookahead_;
  struct Boundary {
    std::array<double, 8> sums{};
    int64_t count{};
  };
  Boundary total_;
  Boundary previous_frame_;
  Boundary chunk_previous_;
  std::vector<Boundary> boundaries_;
};
}  // namespace Generators

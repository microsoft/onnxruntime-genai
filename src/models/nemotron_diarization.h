// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"
#include "ort_genai_c.h"
#include "generator/token_outputs.h"
#include <array>

namespace Generators {
struct NemotronSpeechModel;

void ValidateNemotronDiarizationGraph(const Config::Model& config, const SessionInfo& info);

struct NemotronDiarizationState : State {
  NemotronDiarizationState(const NemotronSpeechModel& model, const GeneratorParams& params);
  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t> = {}) override;
  void SetFeaturesInput(OrtValue& features, int64_t lookahead);
  // Returns cumulative word-window bounds for a chunk-local ASR encoder-frame index,
  // including one ASR frame of left context. No inference or speaker selection occurs here.
  TokenSpeakerEvidence GetTokenSpeakerEvidence(int64_t local_asr_frame) const;
  void Reset();

 private:
  const NemotronSpeechModel& model_;
  std::array<std::unique_ptr<OrtValue>, 6> cache_;
  std::unique_ptr<OrtValue> lookahead_;
  int64_t central_feature_frames_{};
  int64_t feature_frames_per_asr_frame_{};
  std::array<std::vector<int64_t>, 3> cache_shapes_;
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

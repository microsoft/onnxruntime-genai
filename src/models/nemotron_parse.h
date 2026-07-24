// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "model.h"

namespace Generators {

struct NemotronParseModel : Model {
  NemotronParseModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths,
                                     const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> encoder_session_;
  std::unique_ptr<OrtSession> prefill_session_;
  std::unique_ptr<OrtSession> decoder_session_;

  std::unique_ptr<OrtSessionOptions> encoder_session_options_;
  std::unique_ptr<OrtSessionOptions> prefill_session_options_;

  SessionInfo encoder_session_info_;
  SessionInfo prefill_session_info_;
  SessionInfo decoder_session_info_;
};

}  // namespace Generators

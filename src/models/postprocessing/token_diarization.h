// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "generator/token_outputs.h"

namespace Generators {

class TokenDiarization {
 public:
  static void ValidateSpeakerEvidence(const TokenSpeakerEvidence& evidence,
                                      const TokenSpeakerEvidence* previous = nullptr);
  static int AggregateWordSpeaker(const TokenSpeakerEvidence& first, const TokenSpeakerEvidence& last);
  static bool SpeakerChanged(bool has_pending, int pending_speaker, int speaker);
};

}  // namespace Generators

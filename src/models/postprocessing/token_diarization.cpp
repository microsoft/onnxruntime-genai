// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "token_diarization.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Generators {

void TokenDiarization::ValidateSpeakerEvidence(const TokenSpeakerEvidence& evidence,
                                               const TokenSpeakerEvidence* previous) {
  if (evidence.start_frame_count < 0 || evidence.stop_frame_count <= evidence.start_frame_count)
    throw std::runtime_error("Speaker evidence requires a positive evaluated-frame count");

  if (previous && (evidence.start_frame_count < previous->start_frame_count ||
                   evidence.stop_frame_count < previous->stop_frame_count))
    throw std::runtime_error("Speaker evidence must follow token emission order");

  const double count = static_cast<double>(evidence.stop_frame_count - evidence.start_frame_count);
  for (size_t speaker = 0; speaker < 8; ++speaker) {
    const double start = evidence.start_probability_sums[speaker];
    const double stop = evidence.stop_probability_sums[speaker];
    if (!std::isfinite(start) || !std::isfinite(stop) || start < 0 || stop < 0 ||
        start > static_cast<double>(evidence.start_frame_count) + 1e-9 ||
        stop > static_cast<double>(evidence.stop_frame_count) + 1e-9 ||
        stop - start < -1e-9 || stop - start > count + 1e-9)
      throw std::runtime_error("Speaker evidence contains invalid probability sums");

    if (previous) {
      if (start < previous->start_probability_sums[speaker] || stop < previous->stop_probability_sums[speaker])
        throw std::runtime_error("Speaker probability sums must follow token emission order");
      if (start - previous->start_probability_sums[speaker] >
              static_cast<double>(evidence.start_frame_count - previous->start_frame_count) + 1e-9 ||
          stop - previous->stop_probability_sums[speaker] >
              static_cast<double>(evidence.stop_frame_count - previous->stop_frame_count) + 1e-9)
        throw std::runtime_error("Speaker probability sums exceed the newly evaluated frame count");
    }
  }
}

int TokenDiarization::AggregateWordSpeaker(const TokenSpeakerEvidence& first, const TokenSpeakerEvidence& last) {
  TokenSpeakerEvidence word = first;
  std::copy(std::begin(last.stop_probability_sums), std::end(last.stop_probability_sums),
            word.stop_probability_sums);
  word.stop_frame_count = last.stop_frame_count;
  ValidateSpeakerEvidence(word);

  double scores[8];
  double total = 0;
  const double count = static_cast<double>(word.stop_frame_count - word.start_frame_count);
  for (size_t speaker = 0; speaker < 8; ++speaker) {
    scores[speaker] = std::clamp(
        (word.stop_probability_sums[speaker] - word.start_probability_sums[speaker]) / count, 0.01, 1.0);
    total += scores[speaker];
  }
  for (double& score : scores) score /= total;
  return static_cast<int>(std::max_element(std::begin(scores), std::end(scores)) - std::begin(scores));
}

bool TokenDiarization::SpeakerChanged(bool has_pending, int pending_speaker, int speaker) {
  return has_pending && pending_speaker != speaker;
}

}  // namespace Generators

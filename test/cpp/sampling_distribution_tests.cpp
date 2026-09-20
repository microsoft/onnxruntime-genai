// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <limits>

#include <gtest/gtest.h>

#include "sampling_distribution.h"

namespace Generators {
namespace {

TEST(SamplingDistributionTests, AppliesMinLengthAndRepetitionPenalty) {
  const std::array<int32_t, 1> eos_tokens{3};
  LogitsPenaltyProcessor processor(/*vocab_size=*/4, /*repetition_penalty=*/2.0f,
                                   /*min_length=*/5, /*no_repeat_ngram_size=*/0, eos_tokens);
  const std::array<float, 4> logits{4.0f, -2.0f, 3.0f, 5.0f};
  const std::array<int32_t, 2> prefix{0, 1};

  const auto processed = processor.Apply(logits, /*current_length=*/2, prefix);

  EXPECT_FLOAT_EQ(processed[0], 2.0f);
  EXPECT_FLOAT_EQ(processed[1], -4.0f);
  EXPECT_FLOAT_EQ(processed[2], 3.0f);
  EXPECT_EQ(processed[3], std::numeric_limits<float>::lowest());
}

TEST(SamplingDistributionTests, BansTokensThatRepeatNgram) {
  const std::array<int32_t, 1> eos_tokens{4};
  LogitsPenaltyProcessor processor(/*vocab_size=*/5, /*repetition_penalty=*/1.0f,
                                   /*min_length=*/0, /*no_repeat_ngram_size=*/3, eos_tokens);
  const std::array<float, 5> logits{0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  const std::array<int32_t, 6> prefix{1, 2, 3, 4, 1, 2};

  const auto processed = processor.Apply(logits, static_cast<int>(prefix.size()), prefix);

  EXPECT_EQ(processed[3], std::numeric_limits<float>::lowest());
  EXPECT_FLOAT_EQ(processed[4], 4.0f);
}

}  // namespace
}  // namespace Generators
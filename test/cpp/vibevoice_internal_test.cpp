// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// White-box test of the VibeVoice DPM-Solver++ against the reference scheduler.
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "models/vibevoice.h"

namespace Generators {
namespace {

TEST(VibeVoiceSolverTest, MatchesTheReferenceScheduler) {
  // The 5-step VibeVoice schedule, and the values of the NumPy DPM-Solver++ of the converter's
  // vibevoice_full_onnx.py for this velocity function.
  const std::vector<double> sigmas{20291.302734375, 3.103729724884033, 1.3907514810562134, 0.7402812242507935,
                                   0.3374606668949127, 0.0};
  const auto result = SampleDpmSolver(sigmas, {0.5f, -1.0f, 2.0f}, [](size_t step, const std::vector<float>& x) {
    std::vector<float> velocity(x.size());
    for (size_t i = 0; i < x.size(); ++i)
      velocity[i] = static_cast<float>(0.3 * x[i] + 0.1 * static_cast<double>(step) + 0.05 * static_cast<double>(i));
    return velocity;
  });
  ASSERT_EQ(result.size(), 3u);
  EXPECT_NEAR(result[0], -0.02794009, 1e-5);
  EXPECT_NEAR(result[1], -0.89829147, 1e-5);
  EXPECT_NEAR(result[2], 0.66338938, 1e-5);

  EXPECT_THROW(SampleDpmSolver(std::vector<double>{1.0}, {0.0f}, {}), std::runtime_error);
  EXPECT_THROW(SampleDpmSolver(std::vector<double>{1.0, 0.0}, {0.0f, 1.0f},
                               [](size_t, const std::vector<float>&) { return std::vector<float>(1); }),
               std::runtime_error);
}

}  // namespace
}  // namespace Generators

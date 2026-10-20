// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"

namespace {

Generators::Config::SessionOptions MakeSessionOptions(
    std::string provider,
    std::vector<Generators::NamedString> options = {}) {
  Generators::Config::SessionOptions session_options;
  session_options.providers.push_back(provider);
  session_options.provider_options.push_back(
      Generators::Config::ProviderOptions{std::move(provider), std::move(options), std::nullopt});
  return session_options;
}

TEST(ConfigTests, DmlGraphCaptureIsEnabledByDefault) {
  EXPECT_TRUE(Generators::IsGraphCaptureEnabled(MakeSessionOptions("DML")));
}

TEST(ConfigTests, DmlGraphCaptureOptOutAcceptsFalseValuesCaseInsensitively) {
  for (const auto& value : {"0", "false", "False", "FALSE"}) {
    SCOPED_TRACE(value);
    EXPECT_FALSE(Generators::IsGraphCaptureEnabled(
        MakeSessionOptions("DML", {{"enable_graph_capture", value}})));
  }
}

TEST(ConfigTests, DmlGraphCaptureRemainsEnabledForNonOptOutValues) {
  for (const auto& value : {"1", "true", "TRUE", "on", "unexpected"}) {
    SCOPED_TRACE(value);
    EXPECT_TRUE(Generators::IsGraphCaptureEnabled(
        MakeSessionOptions("DML", {{"enable_graph_capture", value}})));
  }
}

TEST(ConfigTests, UnrelatedDmlOptionsDoNotDisableGraphCapture) {
  EXPECT_TRUE(Generators::IsGraphCaptureEnabled(
      MakeSessionOptions("DML", {{"device_index", "0"}})));
}

TEST(ConfigTests, ProviderSpecificGraphCaptureOptionsRemainIndependent) {
  EXPECT_TRUE(Generators::IsGraphCaptureEnabled(
      MakeSessionOptions("NvTensorRtRtx", {{"enable_graph_capture", "false"},
                                           {"enable_cuda_graph", "1"}})));
}

}  // namespace

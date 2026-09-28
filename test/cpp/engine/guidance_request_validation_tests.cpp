// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Model-free tests for the guidance request validation centralized in
// constrained_logits_processor.h/.cpp. Both the generic Generator (via
// CreateGuidanceLogitsProcessor(const State&)) and the Engine's Request (via
// CreateGuidanceLogitsProcessor(shared_ptr<const GeneratorParams>)) route every guidance request
// through this validation, so these tests exercise it directly instead of duplicating the checks
// through a full Generator or Request.
//
// None of these cases need an ONNX model: a malformed/unsupported request and a request in a
// build without USE_GUIDANCE are all rejected before any model is touched. That keeps this test
// buildable and runnable regardless of how USE_GUIDANCE is configured.

#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include "generator/generators.h"
#include "constrained_logits_processor.h"

namespace Generators {
namespace test {
namespace {

// A minimal GeneratorParams with no associated model. GeneratorParams(const Config&) is the same
// model-free constructor search_checkpoint_tests.cpp uses; it leaves model_ null, which is exactly
// the state Engine::Request is in before Assign() binds a model-backed engine.
std::shared_ptr<GeneratorParams> MakeModelFreeParams() {
  static const Config config;
  return std::make_shared<GeneratorParams>(config);
}

TEST(GuidanceRequestValidationTest, NeitherTypeNorDataMeansNoGuidanceRequested) {
  EXPECT_FALSE(ValidateGuidanceRequest("", ""));
}

TEST(GuidanceRequestValidationTest, TypeWithoutDataIsMalformed) {
  try {
    ValidateGuidanceRequest("regex", "");
    FAIL() << "Expected a malformed guidance request to be rejected.";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "Guidance type and data must be provided together.");
  }
}

TEST(GuidanceRequestValidationTest, DataWithoutTypeIsMalformed) {
  try {
    ValidateGuidanceRequest("", "[0-9]+");
    FAIL() << "Expected a malformed guidance request to be rejected.";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "Guidance type and data must be provided together.");
  }
}

TEST(GuidanceRequestValidationTest, UnsupportedTypeIsRejectedIndependentOfUseGuidance) {
  // This must throw whether or not this build links llguidance: an unsupported grammar kind is a
  // caller error, not a build-configuration question.
  try {
    ValidateGuidanceRequest("xml_schema", "<x/>");
    FAIL() << "Expected an unsupported guidance type to be rejected.";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(),
                 "Unsupported guidance type: xml_schema (only json_schema, regex, and "
                 "lark_grammar are supported).");
  }
}

TEST(GuidanceRequestValidationTest, EachSupportedTypeValidatesWithMatchingData) {
  EXPECT_TRUE(ValidateGuidanceRequest("json_schema", "{}"));
  EXPECT_TRUE(ValidateGuidanceRequest("regex", "[0-9]+"));
  EXPECT_TRUE(ValidateGuidanceRequest("lark_grammar", "start: \"a\""));
}

TEST(CreateGuidanceLogitsProcessorTest, NoGuidanceRequestedReturnsNullWithoutAModel) {
  auto params = MakeModelFreeParams();
  EXPECT_EQ(CreateGuidanceLogitsProcessor(params), nullptr);
}

TEST(CreateGuidanceLogitsProcessorTest, IncompleteGuidanceConfigurationThrows) {
  auto params = MakeModelFreeParams();
  params->guidance_type = "regex";
  params->guidance_data = "";

  try {
    CreateGuidanceLogitsProcessor(params);
    FAIL() << "Expected an incomplete guidance configuration to be rejected.";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "Guidance type and data must be provided together.");
  }
}

TEST(CreateGuidanceLogitsProcessorTest, UnsupportedGuidanceTypeThrowsWithoutAModel) {
  auto params = MakeModelFreeParams();
  params->guidance_type = "xml_schema";
  params->guidance_data = "<x/>";

  EXPECT_THROW(CreateGuidanceLogitsProcessor(params), std::runtime_error);
}

#if !USE_GUIDANCE
// In a build without USE_GUIDANCE, a fully valid guidance request must fail loudly with a
// build-configuration error, never fall back to a warning plus unconstrained generation. This must
// happen before the request needs a model, so it must fail even though `params` has no model_.
TEST(CreateGuidanceLogitsProcessorTest, ValidRequestWithoutGuidanceBuildThrowsUnavailableError) {
  auto params = MakeModelFreeParams();
  params->guidance_type = "regex";
  params->guidance_data = "[0-9]+";

  try {
    CreateGuidanceLogitsProcessor(params);
    FAIL() << "Expected guidance to be rejected in a build without USE_GUIDANCE.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("use_guidance=true"), std::string::npos) << message;
  }
}
#else
// With USE_GUIDANCE, a valid request without a model is rejected for the model requirement
// specifically (guidance itself would otherwise be buildable), not the build-availability error.
TEST(CreateGuidanceLogitsProcessorTest, ValidRequestWithoutModelThrows) {
  auto params = MakeModelFreeParams();
  params->guidance_type = "regex";
  params->guidance_data = "[0-9]+";

  try {
    CreateGuidanceLogitsProcessor(params);
    FAIL() << "Expected guidance without a model to be rejected.";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(error.what(), "Guidance requires generator parameters associated with a model.");
  }
}
#endif

}  // namespace
}  // namespace test
}  // namespace Generators

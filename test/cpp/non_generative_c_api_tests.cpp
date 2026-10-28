// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai_c.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace {

void Check(OgaResult* result) {
  if (!result) return;
  std::string error = OgaResultGetError(result);
  OgaDestroyResult(result);
  FAIL() << error;
}

void ExpectOutError(OgaResult* result) {
  ASSERT_NE(result, nullptr);
  EXPECT_NE(std::string(OgaResultGetError(result)).find("out must not be null"),
            std::string::npos);
  OgaDestroyResult(result);
}

}  // namespace

TEST(NonGenerativeCApiTest, StructuredValueBuildersOwnNestedValues) {
  OgaStructuredValueHandle* object{};
  OgaStructuredValueHandle* array{};
  OgaStructuredValueHandle* integer{};
  Check(OgaCreateStructuredValueObject(&object));
  Check(OgaCreateStructuredValueArray(&array));
  Check(OgaCreateStructuredValueInt64(42, &integer));
  Check(OgaStructuredValueArrayAppend(array, integer));
  Check(OgaStructuredValueObjectAppend(object, "items", 5, array));

  OgaDestroyStructuredValue(integer);
  OgaDestroyStructuredValue(array);
  OgaDestroyStructuredValue(object);
}

TEST(NonGenerativeCApiTest, StructuredBuildersAcceptLengthDelimitedStrings) {
  constexpr char text[] = {'l', 'e', 'f', 't', '\0', 'r', 'i', 'g', 'h', 't'};
  constexpr char key[] = {'k', '\0', 'y'};
  OgaStructuredValueHandle *object{}, *value{};
  Check(OgaCreateStructuredValueObject(&object));
  Check(OgaCreateStructuredValueString(text, sizeof(text), &value));
  Check(OgaStructuredValueObjectAppend(object, key, sizeof(key), value));

  OgaDestroyStructuredValue(value);
  OgaDestroyStructuredValue(object);
}

TEST(NonGenerativeCApiTest, RequestBuildersCloneInputs) {
  OgaStructuredValueHandle* null_value{};
  OgaStructuredValueHandle* text{};
  OgaQuestionHandle* question{};
  OgaStructuredRequestHandle* request{};
  Check(OgaCreateStructuredValueNull(&null_value));
  Check(OgaCreateStructuredValueString("Choose", 6, &text));
  Check(OgaCreateQuestion("choice", text, null_value, &question));
  Check(OgaCreateStructuredRequest(&request));
  Check(OgaStructuredRequestSetState(request, null_value));
  Check(OgaStructuredRequestAddQuestion(request, "q", question));
  Check(OgaStructuredRequestSetTemperature(request, 0.5f));

  OgaDestroyQuestion(question);
  OgaDestroyStructuredValue(text);
  OgaDestroyStructuredValue(null_value);
  OgaDestroyStructuredRequest(request);
}

TEST(NonGenerativeCApiTest, FreeFormRankingAndAccessorsRunWhenConfigured) {
  const char* root_value = std::getenv("ORT_GENAI_NON_GENERATIVE_TEST_ROOT");
  if (!root_value || !*root_value)
    GTEST_SKIP() << "set ORT_GENAI_NON_GENERATIVE_TEST_ROOT";
  const auto package =
      std::filesystem::path(root_value) / "clm-v0.1-8b-fp32";

  OgaRankingSessionHandle* session{};
  OgaFreeFormRankRequestHandle* request{};
  OgaStructuredValueHandle *state{}, *instructions{}, *candidate{};
  OgaRankingResultHandle* result{};
  Check(OgaCreateRankingSession(package.string().c_str(), nullptr, 0, &session));
  Check(OgaCreateFreeFormRankRequest(&request));
  Check(OgaCreateStructuredValueNull(&state));
  Check(OgaCreateStructuredValueString("Choose", 6, &instructions));
  Check(OgaCreateStructuredValueString("same", 4, &candidate));
  Check(OgaFreeFormRankRequestSetState(request, state));
  Check(OgaFreeFormRankRequestSetInstructions(request, instructions));
  Check(OgaFreeFormRankRequestAddCandidate(request, "first", candidate));
  Check(OgaFreeFormRankRequestAddCandidate(request, "second", candidate));
  Check(OgaRankingSessionRank(session, request, &result));

  size_t count{}, rank{};
  const char* key{};
  double probability{};
  Check(OgaRankingResultGetCount(result, &count));
  ASSERT_EQ(count, 2u);
  Check(OgaRankingResultGetRank(result, 0, &rank));
  EXPECT_EQ(rank, 1u);
  Check(OgaRankingResultGetKey(result, 0, &key));
  EXPECT_STREQ(key, "first");
  Check(OgaRankingResultGetProbability(result, 0, &probability));
  EXPECT_NEAR(probability, 0.5, 1e-6);
  Check(OgaRankingResultGetKey(result, 1, &key));
  EXPECT_STREQ(key, "second");
  Check(OgaRankingResultGetProbability(result, 1, &probability));
  EXPECT_NEAR(probability, 0.5, 1e-6);

  OgaDestroyRankingResult(result);
  OgaDestroyStructuredValue(candidate);
  OgaDestroyStructuredValue(instructions);
  OgaDestroyStructuredValue(state);
  OgaDestroyFreeFormRankRequest(request);
  OgaDestroyRankingSession(session);
}

TEST(NonGenerativeCApiTest, ErrorsUseOgaResultConvention) {
  OgaStructuredValueHandle* integer{};
  Check(OgaCreateStructuredValueInt64(1, &integer));
  OgaResult* result = OgaStructuredValueArrayAppend(integer, integer);
  ASSERT_NE(result, nullptr);
  EXPECT_NE(std::string(OgaResultGetError(result)).find("not an array"),
            std::string::npos);
  OgaDestroyResult(result);
  OgaDestroyStructuredValue(integer);
}

TEST(NonGenerativeCApiTest, NullCreationOutputsFailBeforeAllocating) {
  for (int iteration = 0; iteration < 16; ++iteration) {
    ExpectOutError(OgaCreateStructuredValueNull(nullptr));
    ExpectOutError(OgaCreateStructuredValueBool(true, nullptr));
    ExpectOutError(OgaCreateStructuredValueInt64(1, nullptr));
    ExpectOutError(OgaCreateStructuredValueDouble(1.0, nullptr));
    ExpectOutError(OgaCreateStructuredValueString("value", 5, nullptr));
    ExpectOutError(OgaCreateStructuredValueArray(nullptr));
    ExpectOutError(OgaCreateStructuredValueObject(nullptr));
    ExpectOutError(OgaCreateQuestion(nullptr, nullptr, nullptr, nullptr));
    ExpectOutError(OgaCreateStructuredRequest(nullptr));
    ExpectOutError(OgaCreateFreeFormRankRequest(nullptr));

    ExpectOutError(OgaCreateRankingSession("unused", nullptr, 0, nullptr));
    ExpectOutError(OgaRankingSessionRun(nullptr, nullptr, nullptr));
    ExpectOutError(OgaRankingSessionRank(nullptr, nullptr, nullptr));
    ExpectOutError(OgaCreateDecisionSession("unused", nullptr, 0, nullptr));
    ExpectOutError(OgaDecisionSessionDecide(nullptr, nullptr, nullptr));
  }
}

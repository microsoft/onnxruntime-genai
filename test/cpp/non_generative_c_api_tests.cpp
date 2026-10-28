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

TEST(NonGenerativeCApiTest, StructuredValueLifecycleAndAccessors) {
  OgaStructuredValueHandle* object{};
  OgaStructuredValueHandle* array{};
  OgaStructuredValueHandle* integer{};
  Check(OgaCreateStructuredValueObject(&object));
  Check(OgaCreateStructuredValueArray(&array));
  Check(OgaCreateStructuredValueInt64(42, &integer));
  Check(OgaStructuredValueArrayAppend(array, integer));
  Check(OgaStructuredValueObjectAppend(object, "items", 5, array));

  OgaStructuredValueType type{};
  size_t count{};
  const char* key{};
  size_t key_length{};
  const OgaStructuredValueHandle* child{};
  Check(OgaStructuredValueGetType(object, &type));
  EXPECT_EQ(type, OgaStructuredValueType_Object);
  Check(OgaStructuredValueGetCount(object, &count));
  EXPECT_EQ(count, 1u);
  Check(OgaStructuredValueGetObjectItem(object, 0, &key, &key_length, &child));
  EXPECT_EQ(std::string_view(key, key_length), "items");
  Check(OgaStructuredValueGetType(child, &type));
  EXPECT_EQ(type, OgaStructuredValueType_Array);
  Check(OgaStructuredValueGetArrayItem(child, 0, &child));
  int64_t value{};
  Check(OgaStructuredValueGetInt64(child, &value));
  EXPECT_EQ(value, 42);

  OgaDestroyStructuredValue(integer);
  OgaDestroyStructuredValue(array);
  OgaDestroyStructuredValue(object);
}

TEST(NonGenerativeCApiTest, StructuredStringsPreserveEmbeddedNulls) {
  constexpr char text[] = {'l', 'e', 'f', 't', '\0', 'r', 'i', 'g', 'h', 't'};
  constexpr char key[] = {'k', '\0', 'y'};
  OgaStructuredValueHandle *object{}, *value{};
  Check(OgaCreateStructuredValueObject(&object));
  Check(OgaCreateStructuredValueString(text, sizeof(text), &value));
  Check(OgaStructuredValueObjectAppend(object, key, sizeof(key), value));

  const char* returned_key{};
  size_t returned_key_length{};
  const OgaStructuredValueHandle* returned_value{};
  Check(OgaStructuredValueGetObjectItem(
      object, 0, &returned_key, &returned_key_length, &returned_value));
  EXPECT_EQ(std::string_view(returned_key, returned_key_length),
            std::string_view(key, sizeof(key)));

  const char* returned_text{};
  size_t returned_text_length{};
  Check(OgaStructuredValueGetString(
      returned_value, &returned_text, &returned_text_length));
  EXPECT_EQ(std::string_view(returned_text, returned_text_length),
            std::string_view(text, sizeof(text)));

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
  const OgaStructuredValueHandle* value{};
  Check(OgaRankingResultGetCount(result, &count));
  ASSERT_EQ(count, 2u);
  Check(OgaRankingResultGetRank(result, 0, &rank));
  EXPECT_EQ(rank, 1u);
  Check(OgaRankingResultGetKey(result, 0, &key));
  EXPECT_STREQ(key, "first");
  Check(OgaRankingResultGetValue(result, 0, &value));
  const char* text{};
  size_t text_length{};
  Check(OgaStructuredValueGetString(value, &text, &text_length));
  EXPECT_EQ(std::string_view(text, text_length), "same");
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai_c.h"

#include <gtest/gtest.h>

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
  Check(OgaStructuredValueObjectAppend(object, "items", array));

  OgaStructuredValueType type{};
  size_t count{};
  const char* key{};
  const OgaStructuredValueHandle* child{};
  Check(OgaStructuredValueGetType(object, &type));
  EXPECT_EQ(type, OgaStructuredValueType_Object);
  Check(OgaStructuredValueGetCount(object, &count));
  EXPECT_EQ(count, 1u);
  Check(OgaStructuredValueGetObjectItem(object, 0, &key, &child));
  EXPECT_STREQ(key, "items");
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

TEST(NonGenerativeCApiTest, RequestBuildersCloneInputs) {
  OgaStructuredValueHandle* null_value{};
  OgaStructuredValueHandle* text{};
  OgaQuestionHandle* question{};
  OgaStructuredRequestHandle* request{};
  Check(OgaCreateStructuredValueNull(&null_value));
  Check(OgaCreateStructuredValueString("Choose", &text));
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

TEST(NonGenerativeCApiTest, ComponentInputsOwnSuppliedStorage) {
  OgaComponentInputs* inputs{};
  Check(OgaCreateComponentInputs(&inputs));
  {
    std::string name = "input";
    std::string bytes = "original bytes";
    std::vector<int64_t> shape{static_cast<int64_t>(bytes.size())};
    Check(OgaComponentInputsAdd(inputs, name.c_str(), bytes.data(), bytes.size(),
                                shape.data(), shape.size(), OgaElementType_uint8));
    name.assign("overwritten");
    bytes.assign(bytes.size(), 'x');
    shape[0] = 0;
  }
  // Destruction after all caller storage is gone is also exercised under ASAN.
  OgaDestroyComponentInputs(inputs);
}

TEST(NonGenerativeCApiTest, PrefixStatusCopyValidatesOutputsFirst) {
  ExpectOutError(OgaDecisionSessionCopyPrefixReuseStatus(
      nullptr, nullptr, 0, nullptr));
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

  OgaComponentSession* session{};
  result = OgaCreateComponentSession(
      "path-that-does-not-exist", "encoder", nullptr, 0, &session);
  ASSERT_NE(result, nullptr);
  EXPECT_FALSE(std::string(OgaResultGetError(result)).empty());
  OgaDestroyResult(result);
  EXPECT_EQ(session, nullptr);
  OgaDestroyComponentSession(session);
}

TEST(NonGenerativeCApiTest, NullCreationOutputsFailBeforeAllocating) {
  for (int iteration = 0; iteration < 16; ++iteration) {
    ExpectOutError(OgaCreateComponentSession("unused", "encoder", nullptr, 0, nullptr));
    ExpectOutError(OgaCreateComponentInputs(nullptr));
    ExpectOutError(OgaComponentSessionRun(nullptr, nullptr, nullptr, 0, nullptr));
    ExpectOutError(OgaCreateDirectoryTokenizer("unused", nullptr));
    ExpectOutError(OgaDirectoryTokenizerEncode(nullptr, "text", nullptr));

    ExpectOutError(OgaCreateStructuredValueNull(nullptr));
    ExpectOutError(OgaCreateStructuredValueBool(true, nullptr));
    ExpectOutError(OgaCreateStructuredValueInt64(1, nullptr));
    ExpectOutError(OgaCreateStructuredValueDouble(1.0, nullptr));
    ExpectOutError(OgaCreateStructuredValueString("value", nullptr));
    ExpectOutError(OgaCreateStructuredValueArray(nullptr));
    ExpectOutError(OgaCreateStructuredValueObject(nullptr));
    ExpectOutError(OgaCreateQuestion(nullptr, nullptr, nullptr, nullptr));
    ExpectOutError(OgaCreateStructuredRequest(nullptr));
    ExpectOutError(OgaCreateFreeFormRankRequest(nullptr));

    ExpectOutError(OgaCreateRankingSession("unused", nullptr, 0, nullptr));
    ExpectOutError(OgaRankingSessionCreateComponent(nullptr, "encoder", nullptr));
    ExpectOutError(OgaRankingSessionRun(nullptr, nullptr, nullptr));
    ExpectOutError(OgaRankingSessionRank(nullptr, nullptr, nullptr));
    ExpectOutError(OgaRankingSessionGetCacheStats(nullptr, nullptr));
    ExpectOutError(OgaCreateDecisionSession("unused", nullptr, 0, nullptr));
    ExpectOutError(OgaDecisionSessionCreateComponent(nullptr, "backbone", nullptr));
    ExpectOutError(OgaDecisionSessionRun(nullptr, nullptr, nullptr));
    ExpectOutError(OgaDecisionSessionDecide(nullptr, nullptr, nullptr));
    ExpectOutError(OgaDecisionSessionGetCacheStats(nullptr, nullptr));
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/component_session.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/models/"
#endif

TEST(ComponentPackageTokenizerTest, BatchRowsMatchIndividualEncoding) {
  Generators::ComponentPackageTokenizer tokenizer{
      std::string(MODEL_PATH) + "multimodal-decoder-no-input-ids"};
  const std::vector<std::string> texts{
      "hello", "a longer input with punctuation!", ""};

  const auto rows = tokenizer.EncodeBatch(texts);
  ASSERT_EQ(rows.size(), texts.size());
  for (size_t index = 0; index < texts.size(); ++index)
    EXPECT_EQ(rows[index], tokenizer.Encode(texts[index]));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

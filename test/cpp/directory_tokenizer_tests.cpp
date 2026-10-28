// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai.h"

#include <gtest/gtest.h>

#include <string>

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/models/"
#endif

TEST(DirectoryTokenizerTest, ReusesPublicTokenizerFromPath) {
  const auto path =
      std::string(MODEL_PATH) + "multimodal-decoder-no-input-ids";
  auto tokenizer = OgaTokenizer::Create(path.c_str());
  auto sequences = OgaSequences::Create();
  tokenizer->Encode("hello", *sequences);
  EXPECT_GT(sequences->SequenceCount(0), 0u);
  EXPECT_EQ(tokenizer->GetPadTokenId(), 0);
}

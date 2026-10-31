// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ort_genai.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

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

TEST(DirectoryTokenizerTest, BatchMatchesSingleEncodingInOrdinaryAndLiteralModes) {
  const auto path =
      std::string(MODEL_PATH) + "multimodal-decoder-no-input-ids";
  auto tokenizer = OgaTokenizer::Create(path.c_str());
  const auto compare = [&](OgaTokenizer& selected,
                           const std::vector<std::string>& strings,
                           const char* literal) {
    const char* keys[] = {"literal"};
    const char* options[] = {literal};
    selected.UpdateOptions(keys, options, 1);

    std::vector<const char*> values;
    for (const auto& value : strings) values.push_back(value.c_str());
    auto singles = OgaSequences::Create();
    size_t width{};
    for (const auto& value : strings) {
      selected.Encode(value.c_str(), *singles);
      width = std::max(width, singles->SequenceCount(singles->Count() - 1));
    }

    auto batch = selected.EncodeBatch(values.data(), values.size());
    EXPECT_EQ(batch->Shape(),
              (std::vector<int64_t>{
                  static_cast<int64_t>(strings.size()),
                  static_cast<int64_t>(width)}));
    const auto* ids = static_cast<const int32_t*>(batch->Data());
    ASSERT_NE(ids, nullptr);
    for (size_t row = 0; row < strings.size(); ++row) {
      const auto count = singles->SequenceCount(row);
      EXPECT_EQ(
          std::vector<int32_t>(ids + row * width,
                               ids + row * width + count),
          std::vector<int32_t>(singles->SequenceData(row),
                               singles->SequenceData(row) + count));
      for (size_t column = count; column < width; ++column)
        EXPECT_EQ(ids[row * width + column], selected.GetPadTokenId());
    }
  };

  compare(*tokenizer,
          {"", "hello", "Unicode: 编码器 🚀",
           "special tokens: <s> </s> <unk>",
           "a substantially longer row with punctuation, numbers 123, and spaces"},
          "false");

  auto literal_tokenizer =
      OgaTokenizer::Create((std::string(MODEL_PATH) + "phi3-v").c_str());
  compare(*literal_tokenizer,
          {"", "<|endoftext|>", "<|assistant|>",
           "<|endoftext|><|assistant|>",
           "Unicode: 编码器 🚀 and ordinary text"},
          "true");
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <span>
#include <cstring>  // for memcmp
#include <numeric>
#include <random>
#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>
#include <gtest/gtest.h>
#include "models/onnxruntime_api.h"
//#include "onnxruntime_c_api.h"

// Our working directory is generators/build so one up puts us in the root directory:
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif

TEST(StableDiffusionTests, ClipTokenizer) {
  auto config = OgaConfig::Create(MODEL_PATH "sd");

  auto model = OgaModel::Create(*config);
  auto tokenizer = OgaTokenizer::Create(*model);

  auto sequences = OgaSequences::Create();
  tokenizer->Encode("Capybara with his mom and dad in a beautiful stream", *sequences);

  std::vector<int32_t> expected_tokens{49406, 1289, 88, 19345, 593, 787, 2543, 537, 2639, 530, 320, 1215, 3322, 49407};

  EXPECT_TRUE(1 == sequences->Count());
  EXPECT_TRUE(sequences->SequenceCount(0) == expected_tokens.size());
  EXPECT_TRUE(0 == std::memcmp(sequences->SequenceData(0), expected_tokens.data(), expected_tokens.size() * sizeof(int32_t)));
}

TEST(StableDiffusionTests, TextEmbeddings) {
  auto config = OgaConfig::Create(MODEL_PATH "sd");

  auto model = OgaModel::Create(*config);
  auto tokenizer = OgaTokenizer::Create(*model);

  auto sequences = OgaSequences::Create();
  tokenizer->Encode("Capybara with his mom and dad in a beautiful stream", *sequences);

  OgaTensor* output{};
  OgaCheckResult(OgaSelenaTest(&output, sequences->SequenceData(0), sequences->SequenceCount(0), MODEL_PATH "sd/text_encoder/model.onnx", MODEL_PATH "sd/unet/model.onnx", MODEL_PATH "sd/vae_decoder/model.onnx"));
}

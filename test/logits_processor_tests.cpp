// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <logits_processor.h>
#include <cstdio>
#include <iostream>
#include <random>
#include <fstream>
#include <regex>

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif
#ifndef PHI2_PATH
#if USE_CUDA
#define PHI2_PATH MODEL_PATH "phi-2/int4/cuda"
#else
#define PHI2_PATH MODEL_PATH "phi-2/int4/cpu"
#endif
#endif
#ifndef SCHEMA_PATH
#define SCHEMA_PATH MODEL_PATH "grammars/blog.schema.json"
#endif

std::string read_file(const char* filePath) {
  std::ifstream file(filePath);
  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}
#if USE_GUIDANCE
TEST(LogitsProcessorTests, TestRegex) {
  std::string regex = "answer: .*";
  std::string text = "answer: I am a robot";
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto tokenizer = model->CreateTokenizer();
  auto params = Generators::CreateGeneratorParams(*model);
  params->SetGuidance("regex", regex);
  auto generator = Generators::CreateGenerator(*model, *params);
  auto processor = std::make_unique<Generators::GuidanceLogitsProcessor>(*generator->state_);
  auto target_ids = Generators::GuidanceLogitsProcessor::tokenize_partial(tokenizer.get(), tokenizer->Encode(Generators::GuidanceLogitsProcessor::kTokenizePrefixStr).size(),
                                                                          reinterpret_cast<const uint8_t*>(text.c_str()), text.size());
  for (auto id : target_ids) {
    auto mask = processor->GetMask();
    auto tokens = std::vector<int32_t>{static_cast<int32_t>(id)};
    processor->CommitTokens(std::span<int32_t>(tokens));
  }
}

TEST(LogitsProcessorTests, TestJsonSchema) {
  std::string json_schema = read_file(MODEL_PATH "grammars/blog.schema.json");
  std::string text = read_file(MODEL_PATH "grammars/blog.sample.json");
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto tokenizer = model->CreateTokenizer();
  auto params = Generators::CreateGeneratorParams(*model);
  params->SetGuidance("json_schema", json_schema);
  auto generator = Generators::CreateGenerator(*model, *params);
  auto processor = std::make_unique<Generators::GuidanceLogitsProcessor>(*generator->state_);
  auto target_ids = Generators::GuidanceLogitsProcessor::tokenize_partial(tokenizer.get(), tokenizer->Encode(Generators::GuidanceLogitsProcessor::kTokenizePrefixStr).size(),
                                                                          reinterpret_cast<const uint8_t*>(text.c_str()), text.size());
  for (auto id : target_ids) {
    auto mask = processor->GetMask();
    auto tokens = std::vector<int32_t>{static_cast<int32_t>(id)};
    processor->CommitTokens(std::span<int32_t>(tokens));
  }
}

#endif
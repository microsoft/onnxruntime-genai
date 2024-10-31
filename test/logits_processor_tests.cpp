// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <models/logits_processor.h>
#include <cstdio>
#include <iostream>
#include <random>
#include <fstream>

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

TEST(LogitsProcessorTests, TestRegex) {
  std::string regex = "answer: .*";
  std::string text = "\x02" + std::string("answer: I am a robot");
  // auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), "/home/yingxiong/projects/onnxruntime-genai/phi3_cpu_test");
  auto tokenizer = model->CreateTokenizer();
  auto processor = std::make_unique<Generators::ConstrainedLogitsProcessor>(model->config_->model.vocab_size,
                                                                            model->config_->model.eos_token_id, "regex",
                                                                            regex, tokenizer);
  auto target_ids = tokenizer->Encode(text.c_str());
  std::vector<int32_t> tids(target_ids.begin() + 2, target_ids.end());
  for (auto id : tids) {
    auto mask = processor->ComputeMask();
    processor->CommitTokens(id);
  }
}

TEST(LogitsProcessorTests, TestJsonSchema) {
  std::string json_schema = read_file(MODEL_PATH "grammars/blog.schema.json");
  std::string text = "\x02" + read_file(MODEL_PATH "grammars/blog.sample.json");
  // auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), "/home/yingxiong/projects/onnxruntime-genai/phi3_cpu_test");
  auto tokenizer = model->CreateTokenizer();
  auto processor = std::make_unique<Generators::ConstrainedLogitsProcessor>(model->config_->model.vocab_size,
                                                                            model->config_->model.eos_token_id, "json_schema",
                                                                            json_schema, tokenizer);
  auto target_ids = tokenizer->Encode(text.c_str());
  std::vector<int32_t> tids(target_ids.begin() + 2, target_ids.end());
  for (auto id : tids) {
    std::cout << id << std::endl;
    auto mask = processor->ComputeMask();
    processor->CommitTokens(id);
  }
}

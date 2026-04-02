// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdlib>
#include <cstring>  // for memcmp
#include <filesystem>
#include <fstream>
#include <numeric>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include "span.h"
#include <list>

#define OGA_USE_SPAN 1
#include "models/onnxruntime_api.h"
#include "ort_genai.h"

#include <gtest/gtest.h>

#include "test_utils.h"

#ifndef QWEN2_5_PATH
#define QWEN2_5_PATH test_utils::GetModelPath("qwen-2.5-0.5b").c_str()
#endif

#if USE_GUIDANCE
const std::string json_schema = R"json({"x-guidance": {"whitespace_flexible": false, "key_separator": ": ", "item_separator": ", "}, "type": "array", "items": {"anyOf": [{"description": "How to get the statistics for a city", "type": "object", "properties": {"name": {"const": "get_statistics"}, "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}, "required": ["name", "parameters"], "additionalProperties": false}, {"description": "How to get the weather for a city", "type": "object", "properties": {"name": {"const": "get_weather"}, "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}, "required": ["name", "parameters"], "additionalProperties": false}, {"description": "How to get the population for a city", "type": "object", "properties": {"name": {"const": "get_population"}, "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}, "required": ["name", "parameters"], "additionalProperties": false}]}, "minItems": 1})json";

TEST(GuidanceTests, UseRegex) {
#if TEST_QWEN_2_5
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto stream = OgaTokenizerStream::Create(*tokenizer);

  const char* input_string = "who are you?";
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 32);
  params->SetGuidance("regex", "answer: .*", false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
  auto output = std::string(out_string);
  EXPECT_TRUE(std::regex_match(output, std::regex("answer: .*")));
#endif
}

TEST(GuidanceTests, UseLarkGrammarSingleTurn) {
#if TEST_QWEN_2_5
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto stream = OgaTokenizerStream::Create(*tokenizer);

  const char* input_string = "What is the weather in San Francisco?";
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 256);
  std::string lark_grammar = std::string("start: functioncall\nfunctioncall: %json ") + json_schema;
  params->SetGuidance("lark_grammar", lark_grammar.c_str(), false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
  auto output = std::string(out_string);
  std::cout << output << std::endl;
  EXPECT_TRUE(std::regex_match(output, std::regex(R"(\[{"name": "get_weather", "parameters": {"city": "San Francisco"}}\])")));
#endif
}

TEST(GuidanceTests, UseJsonSchemaSingleTurn) {
#if TEST_QWEN_2_5
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto stream = OgaTokenizerStream::Create(*tokenizer);

  const char* input_string = "What is the weather in San Francisco?";
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 256);
  params->SetGuidance("json_schema", json_schema.c_str(), false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
  auto output = std::string(out_string);
  std::cout << output << std::endl;
  EXPECT_TRUE(std::regex_match(output, std::regex(R"(\[{"name": "get_weather", "parameters": {"city": "San Francisco"}}\])")));
#endif
}

TEST(GuidanceTests, UseLarkGrammarMultiTurn) {
#if TEST_QWEN_2_5
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto stream = OgaTokenizerStream::Create(*tokenizer);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 1024);
  std::string lark_grammar = std::string("start: functioncall\nfunctioncall: %json ") + json_schema;
  params->SetGuidance("lark_grammar", lark_grammar.c_str(), false);
  auto generator = OgaGenerator::Create(*model, *params);

  const std::vector<std::string> cities{"San Francisco", "Seattle", "Boston"};
  for (const auto& city : cities) {
    auto input_sequences = OgaSequences::Create();
    auto input_string = "What is the weather in " + city + "?";
    tokenizer->Encode(input_string.c_str(), *input_sequences);

    generator->AppendTokenSequences(*input_sequences);
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }

    auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
    auto output = std::string(out_string);
    std::cout << output << std::endl;
    auto expected_pattern = std::string(R"(\[{"name": "get_weather", "parameters": {"city": ")") + city + R"("}}\])";
    EXPECT_TRUE(std::regex_match(output, std::regex(expected_pattern)));
  }
#endif
}

TEST(GuidanceTests, UseJsonSchemaMultiTurn) {
#if TEST_QWEN_2_5
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto stream = OgaTokenizerStream::Create(*tokenizer);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 1024);
  params->SetGuidance("json_schema", json_schema.c_str(), false);
  auto generator = OgaGenerator::Create(*model, *params);

  const std::vector<std::string> cities{"San Francisco", "Seattle", "Boston"};
  for (const auto& city : cities) {
    auto input_sequences = OgaSequences::Create();
    auto input_string = "What is the weather in " + city + "?";
    tokenizer->Encode(input_string.c_str(), *input_sequences);

    generator->AppendTokenSequences(*input_sequences);
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }

    auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
    auto output = std::string(out_string);
    std::cout << output << std::endl;
    auto expected_pattern = std::string(R"(\[{"name": "get_weather", "parameters": {"city": ")") + city + R"("}}\])";
    EXPECT_TRUE(std::regex_match(output, std::regex(expected_pattern)));
  }
#endif
}

#endif

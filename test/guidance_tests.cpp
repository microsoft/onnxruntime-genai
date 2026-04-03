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

#if USE_GUIDANCE
const std::string json_schema = R"json({"x-guidance": {"whitespace_flexible": false, "key_separator": ": ", "item_separator": ", "}, "type": "array", "items": {"anyOf": [{"description": "How to get the statistics for a city", "type": "object", "properties": {"name": {"const": "get_statistics"}, "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}, "required": ["name", "parameters"], "additionalProperties": false}, {"description": "How to get the weather for a city", "type": "object", "properties": {"name": {"const": "get_weather"}, "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}, "required": ["name", "parameters"], "additionalProperties": false}, {"description": "How to get the population for a city", "type": "object", "properties": {"name": {"const": "get_population"}, "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}, "required": ["name", "parameters"], "additionalProperties": false}]}, "minItems": 1})json";

#if TEST_QWEN_2_5
auto model = OgaModel::Create(QWEN_2_5_PATH);
auto tokenizer = OgaTokenizer::Create(*model);
auto stream = OgaTokenizerStream::Create(*tokenizer);

const std::string qwen_2_5_system_prompt = "<|im_start|>system\nYou are a helpful AI assistant.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"name\": \"get_statistics\", \"description\": \"How to get the statistics for a city\", \"parameters\": {\"city\": {\"type\": \"str\"}}}\n{\"name\": \"get_weather\", \"description\": \"How to get the weather for a city\", \"parameters\": {\"city\": {\"type\": \"str\"}}}\n{\"name\": \"get_population\", \"description\": \"How to get the population for a city\", \"parameters\": {\"city\": {\"type\": \"str\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
const std::string qwen_2_5_generation_prompt = "<|im_start|>assistant\n";
const auto get_qwen_2_5_user_prompt = [](const std::string& user_content) {
  return "<|im_start|>user\n" + user_content + "<|im_end|>\n" + qwen_2_5_generation_prompt;
};
const auto get_qwen_2_5_prompt = [](const std::string& user_content) {
  auto user_prompt = get_qwen_2_5_user_prompt(user_content);
  return qwen_2_5_system_prompt + user_prompt;
};
#endif

TEST(GuidanceTests, UseRegex) {
#if TEST_QWEN_2_5
  const char* input_string = "who are you?";
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 32);
  params->SetGuidance("regex", "answer: .*", false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  std::string output = "";
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    const auto new_token = generator->GetNextTokens()[0];
    output += stream->Decode(new_token);
  }
  EXPECT_TRUE(std::regex_match(output, std::regex("answer: .*")));
#endif
}

TEST(GuidanceTests, UseLarkGrammarSingleTurn) {
#if TEST_QWEN_2_5
  auto input_string = get_qwen_2_5_prompt("What is the weather in Seattle?");
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string.c_str(), *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 512);
  std::string lark_grammar = std::string("start: functioncall\nfunctioncall: %json ") + json_schema;
  params->SetGuidance("lark_grammar", lark_grammar.c_str(), false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  std::string output = "";
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    const auto new_token = generator->GetNextTokens()[0];
    output += stream->Decode(new_token);
  }
  const std::string expected_output = R"([{"name": "get_weather", "parameters": {"city": "Seattle"}}])";
  EXPECT_EQ(output, expected_output);
#endif
}

TEST(GuidanceTests, UseJsonSchemaSingleTurn) {
#if TEST_QWEN_2_5
  auto input_string = get_qwen_2_5_prompt("What is the weather in Seattle?");
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string.c_str(), *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 512);
  params->SetGuidance("json_schema", json_schema.c_str(), false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  std::string output = "";
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    const auto new_token = generator->GetNextTokens()[0];
    output += stream->Decode(new_token);
  }
  const std::string expected_output = R"([{"name": "get_weather", "parameters": {"city": "Seattle"}}])";
  EXPECT_EQ(output, expected_output);
#endif
}

TEST(GuidanceTests, UseLarkGrammarMultiTurn) {
#if TEST_QWEN_2_5
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 1024);
  std::string lark_grammar = std::string("start: functioncall\nfunctioncall: %json ") + json_schema;
  params->SetGuidance("lark_grammar", lark_grammar.c_str(), false);
  auto generator = OgaGenerator::Create(*model, *params);

  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(qwen_2_5_system_prompt.c_str(), *input_sequences);
  generator->AppendTokenSequences(*input_sequences);

  const std::vector<std::string> cities{"Redmond", "Seattle", "Boston"};
  for (const auto& city : cities) {
    input_sequences = OgaSequences::Create();
    std::string user_content = "What is the weather in " + city + "?";
    auto input_string = get_qwen_2_5_user_prompt(user_content);
    tokenizer->Encode(input_string.c_str(), *input_sequences);

    generator->AppendTokenSequences(*input_sequences);
    std::string output = "";
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
      const auto new_token = generator->GetNextTokens()[0];
      output += stream->Decode(new_token);
    }
    auto expected_output = std::string(R"([{"name": "get_weather", "parameters": {"city": ")") + city + R"("}}])";
    EXPECT_EQ(output, expected_output);
  }
#endif
}

TEST(GuidanceTests, UseJsonSchemaMultiTurn) {
#if TEST_QWEN_2_5
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 1024);
  params->SetGuidance("json_schema", json_schema.c_str(), false);
  auto generator = OgaGenerator::Create(*model, *params);

  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(qwen_2_5_system_prompt.c_str(), *input_sequences);
  generator->AppendTokenSequences(*input_sequences);

  const std::vector<std::string> cities{"Redmond", "Seattle", "Boston"};
  for (const auto& city : cities) {
    input_sequences = OgaSequences::Create();
    std::string user_content = "What is the weather in " + city + "?";
    auto input_string = get_qwen_2_5_user_prompt(user_content);
    tokenizer->Encode(input_string.c_str(), *input_sequences);

    generator->AppendTokenSequences(*input_sequences);
    std::string output = "";
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
      const auto new_token = generator->GetNextTokens()[0];
      output += stream->Decode(new_token);
    }
    auto expected_output = std::string(R"([{"name": "get_weather", "parameters": {"city": ")") + city + R"("}}])";
    EXPECT_EQ(output, expected_output);
  }
#endif
}

#endif

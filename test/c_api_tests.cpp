// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>  // for memcmp
#include <fstream>
#include <numeric>
#include <iostream>
#include <thread>
#include <vector>
#include <regex>
#include "span.h"
#include <list>

#define OGA_USE_SPAN 1
#include "models/onnxruntime_api.h"
#include "ort_genai.h"

#include <gtest/gtest.h>

#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif
#ifndef PHI2_PATH
#if USE_CUDA
#define PHI2_PATH MODEL_PATH "phi-2/int4/cuda"
#elif USE_DML
#define PHI2_PATH MODEL_PATH "phi-2/int4/dml"
#else
#define PHI2_PATH MODEL_PATH "phi-2/int4/cpu"
#endif
#endif

#ifndef ENABLE_ENGINE_TESTS
#define ENABLE_ENGINE_TESTS TEST_PHI2 && !USE_DML
#endif

TEST(CAPITests, Config) {
#if TEST_PHI2
  // Test modifying config settings
  auto config = OgaConfig::Create(PHI2_PATH);
  config->AppendProvider("brainium");
  config->SetProviderOption("super_ai", "custom_field", "hello");
  config->AppendProvider("human");
  config->SetProviderOption("brainium", "custom_field1", "hello1");
  config->SetProviderOption("brainium", "custom_field2", "hello2");
  config->ClearProviders();
  config->AppendProvider("cuda");
  config->AppendProvider("dml");
#endif
}

TEST(CAPITests, TokenizerCAPI) {
#if TEST_PHI2
  auto config = OgaConfig::Create(PHI2_PATH);
  auto model = OgaModel::Create(*config);
  auto tokenizer = OgaTokenizer::Create(*model);

  // Encode single decode single
  {
    const char* input_string = "She sells sea shells by the sea shore.";
    auto input_sequences = OgaSequences::Create();
    tokenizer->Encode(input_string, *input_sequences);

    auto out_string = tokenizer->Decode(input_sequences->SequenceData(0), input_sequences->SequenceCount(0));
    ASSERT_STREQ(input_string, out_string);
  }

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  auto sequences = OgaSequences::Create();

  // Encode all strings
  {
    for (auto& string : input_strings)
      tokenizer->Encode(string, *sequences);
  }

  // Decode one at a time
  for (size_t i = 0; i < sequences->Count(); i++) {
    auto out_string = tokenizer->Decode(sequences->SequenceData(i), sequences->SequenceCount(i));
    std::cout << "Decoded string:" << out_string << std::endl;
    if (strcmp(input_strings[i], out_string) != 0)
      throw std::runtime_error("Token decoding mismatch");
  }

  // Stream Decode one at a time
  for (size_t i = 0; i < sequences->Count(); i++) {
    auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

    auto* sequence = sequences->SequenceData(i);
    std::string stream_result;
    for (size_t j = 0; j < sequences->SequenceCount(i); j++) {
      stream_result += tokenizer_stream->Decode(sequence[j]);
    }
    std::cout << "Stream decoded string:" << stream_result << std::endl;
    if (strcmp(input_strings[i], stream_result.c_str()) != 0)
      throw std::runtime_error("Stream token decoding mismatch");
  }
#endif
}

TEST(CAPITests, ChatTemplate) {
#if TEST_PHI2
  // We load the phi-2 model just to get a tokenizer (phi-2 does not have a chat template)
  auto tokenizer = OgaTokenizer::Create(*OgaModel::Create(PHI2_PATH));

  // Testing phi-4 chat template
  const char* messages_json = R"(
    [
      {
        "role": "system",
        "content": "You are a helpful assistant.",
        "tools": "Calculator"
      },
      {
        "role": "user",
        "content": "How do I add two numbers?"
      },
      {
        "role": "assistant",
        "content": "You can add numbers by using the '+' operator."
      }
    ])";
  const char* chat_template = R"({% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %})";

  // From HuggingFace Python output for 'microsoft/Phi-4-multimodal-instruct'
  const char* expected_output =
      "<|system|>You are a helpful assistant.<|tool|>Calculator<|/tool|><|end|><|user|>"
      "How do I add two numbers?<|end|><|assistant|>You can add numbers by using the '+' operator.<|end|><|assistant|>";

  auto out_string = tokenizer->ApplyChatTemplate(chat_template, messages_json, nullptr, true);
  ASSERT_STREQ(expected_output, out_string);

#endif
}

TEST(CAPITests, AppendTokensToSequence) {
#if TEST_PHI2
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  auto sequences = OgaSequences::Create();
  auto appended_sequences = OgaSequences::Create();

  // Encode all strings
  {
    for (auto& string : input_strings)
      tokenizer->Encode(string, *sequences);
  }

  // Append token sequence to another sequence
  // Basically create a copy
  for (size_t i = 0; i < sequences->Count(); i++) {
    auto* sequence = sequences->SequenceData(i);
    appended_sequences->Append(sequence, sequences->SequenceCount(i));
  }
  // All sequences should be copied
  EXPECT_EQ(appended_sequences->Count(), sequences->Count());

  // Compare each token in each sequence
  for (int i = 0; i < sequences->Count(); i++) {
    auto* sequence = sequences->SequenceData(i);
    auto* appended_sequence = appended_sequences->SequenceData(i);
    EXPECT_EQ(sequences->SequenceCount(i), appended_sequences->SequenceCount(i));

    for (size_t j = 0; j < sequences->SequenceCount(i); j++) {
      EXPECT_EQ(sequence[j], appended_sequence[j]);
    }
  }
#endif
}

TEST(CAPITests, MaxLength) {
  // Batch size 1 case
  std::vector<int32_t> input_ids_0{1, 2, 3, 5, 8};
  std::vector<int32_t> input_ids_1{13, 21, 34, 55, 89};

  int max_length = 7;

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids_0.data(), input_ids_0.size());
  EXPECT_THROW(generator->AppendTokens(input_ids_1.data(), input_ids_1.size()), std::runtime_error);

#if !USE_DML
  // Batch size 3 case
  std::vector<int32_t> input_ids_2{1, 2, 3, 5, 8, 13, 21, 34, 55, 89,
                                   0, 0, 0, 52, 104, 52, 53, 54, 55, 56,
                                   0, 0, 195, 731, 731, 195, 64, 45, 23, 12};
  params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", 3);

  generator = OgaGenerator::Create(*model, *params);
  EXPECT_THROW(generator->AppendTokens(input_ids_2.data(), input_ids_2.size()), std::runtime_error);
#endif
}

#if ENABLE_ENGINE_TESTS
TEST(CAPIEngineTests, MaxLength) {
  std::vector<int32_t> input_ids{1, 2, 3, 5, 8, 2, 1, 4, 5, 7};

  auto model = OgaModel::Create(PHI2_PATH);
  auto engine = OgaEngine::Create(*model);

  auto sequence = OgaSequences::Create();
  sequence->Append(input_ids.data(), input_ids.size());

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", static_cast<int>(input_ids.size()) - 1);  // Set max_length to one less than input size
  auto request = OgaRequest::Create(*params);
  EXPECT_THROW(request->AddTokens(*sequence), std::runtime_error);

  params->SetSearchOption("max_length", static_cast<int>(input_ids.size()) + 1);  // Set max_length to one more than input size
  request->AddTokens(*sequence);
  ASSERT_TRUE(request != nullptr);
  ASSERT_FALSE(request->IsDone());
}
#endif

// DML doesn't support batch_size > 1
TEST(CAPITests, EndToEndPhiBatch) {
#if TEST_PHI2 && !USE_DML
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  auto input_sequences = OgaSequences::Create();
  for (auto& string : input_strings)
    tokenizer->Encode(string, *input_sequences);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 40);
  params->SetSearchOption("batch_size", 3);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Decode The Batch
  for (size_t i = 0; i < 3; i++) {
    auto out_string = tokenizer->Decode(generator->GetSequenceData(i), generator->GetSequenceCount(i));
    std::cout << "Decoded string:" << out_string << std::endl;
  }

  // Verify outputs match expected outputs
  std::vector<int32_t> expected_output{
      1212, 318, 257, 1332, 13, 50256, 50256, 50256, 50256, 50256, 198, 50280, 2, 16926, 1330, 1635, 10412, 6617, 278, 6335, 32994, 21857, 13849, 38665, 82, 21815, 1108, 9557, 40755, 27446, 2417, 6381, 6, 7131, 6, 14870, 31314, 21411, 46009, 3974,
      49, 1381, 389, 7427, 17252, 0, 50256, 50256, 50256, 50256, 198, 50284, 37811, 628, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
      464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13, 198, 50284, 37811, 628, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256};

  for (size_t i = 0; i < 3; i++) {
    const auto sequence_length = generator->GetSequenceCount(i);
    const auto* sequence_data = generator->GetSequenceData(i);

    ASSERT_LE(sequence_length, 40);

    const auto* expected_output_start = &expected_output[i * 40];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
  }
#endif
}

#if ENABLE_ENGINE_TESTS
TEST(CAPIEngineTests, EndToEndPhiBatch) {
  auto model = OgaModel::Create(PHI2_PATH);
  auto engine = OgaEngine::Create(*model);
  auto tokenizer = OgaTokenizer::Create(*model);

  constexpr size_t batch_size = 3;
  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  std::vector<std::unique_ptr<OgaRequest>> requests;
  std::vector<std::unique_ptr<OgaGeneratorParams>> params;
  std::vector<std::unique_ptr<OgaTokenizerStream>> tokenizer_streams;
  std::array<std::vector<int32_t>, batch_size> generated_tokens;
  for (auto& string : input_strings) {
    auto input_sequences = OgaSequences::Create();
    tokenizer->Encode(string, *input_sequences);
    generated_tokens[requests.size()] = std::vector<int32_t>(input_sequences->SequenceData(0),
                                                             input_sequences->SequenceData(0) +
                                                                 input_sequences->SequenceCount(0));
    params.emplace_back(OgaGeneratorParams::Create(*model));
    params.back()->SetSearchOption("max_length", 40);
    requests.push_back(OgaRequest::Create(*params.back()));
    requests.back()->AddTokens(*input_sequences);
    requests.back()->SetOpaqueData(&generated_tokens[requests.size() - 1]);
    tokenizer_streams.emplace_back(OgaTokenizerStream::Create(*tokenizer));

    engine->Add(*requests.back());
  }

  while (auto request = engine->Step()) {
    while (request->HasUnseenTokens()) {
      auto* tokens = reinterpret_cast<std::vector<int32_t>*>(request->GetOpaqueData());
      tokens->push_back(request->GetUnseenToken());
    }
  }

  for (size_t i = 0; i < batch_size; i++) {
    auto out_string = tokenizer->Decode(generated_tokens[i].data(), generated_tokens[i].size());
    std::cout << "Decoded string:" << out_string << std::endl;
  }

  // Verify outputs match expected outputs
  std::vector<std::vector<int32_t>> expected_output{
      {1212, 318, 257, 1332, 13, 198, 50280, 2, 16926, 1330,
       1635, 10412, 6617, 278, 6335, 32994, 21857, 13849, 38665, 82,
       21815, 1108, 9557, 40755, 27446, 2417, 6381, 6, 7131, 6,
       14870, 31314, 21411, 46009, 3974, 82, 1039, 889, 263, 3684},
      {49, 1381, 389, 7427, 17252, 0, 198, 50284, 37811, 628, 50256},
      {464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13,
       198, 50284, 37811, 628, 50256}};

  for (size_t i = 0; i < batch_size; i++) {
    ASSERT_LE(generated_tokens[i].size(), 40);
    EXPECT_EQ(expected_output[i].size(), generated_tokens[i].size());
    EXPECT_EQ(expected_output[i], generated_tokens[i]);
  }
}
#endif

#if ENABLE_ENGINE_TESTS
TEST(CAPIEngineTests, EndToEndPhiStaggeredBatch) {
  auto model = OgaModel::Create(PHI2_PATH);
  auto engine = OgaEngine::Create(*model);
  auto tokenizer = OgaTokenizer::Create(*model);

  constexpr size_t batch_size = 3;
  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  std::vector<std::unique_ptr<OgaRequest>> requests;
  std::vector<std::unique_ptr<OgaGeneratorParams>> params;
  std::vector<std::unique_ptr<OgaTokenizerStream>> tokenizer_streams;
  std::array<std::vector<int32_t>, batch_size> generated_tokens;
  for (auto& string : input_strings) {
    auto input_sequences = OgaSequences::Create();
    tokenizer->Encode(string, *input_sequences);
    generated_tokens[requests.size()] = std::vector<int32_t>(input_sequences->SequenceData(0),
                                                             input_sequences->SequenceData(0) +
                                                                 input_sequences->SequenceCount(0));
    params.emplace_back(OgaGeneratorParams::Create(*model));
    params.back()->SetSearchOption("max_length", 40);
    requests.push_back(OgaRequest::Create(*params.back()));
    requests.back()->AddTokens(*input_sequences);
    requests.back()->SetOpaqueData(&generated_tokens[requests.size() - 1]);
    tokenizer_streams.emplace_back(OgaTokenizerStream::Create(*tokenizer));
  }

  // Add the first request to the engine
  engine->Add(*requests[0]);

  size_t num_steps = 0;
  while (auto request = engine->Step()) {
    num_steps++;
    while (request->HasUnseenTokens()) {
      auto* tokens = reinterpret_cast<std::vector<int32_t>*>(request->GetOpaqueData());
      tokens->push_back(request->GetUnseenToken());
    }

    if (num_steps == 5)
      engine->Add(*requests[1]);  // Stagger the second request

    if (num_steps == 10)
      engine->Add(*requests[2]);  // Stagger the third request
  }

  for (size_t i = 0; i < batch_size; i++) {
    auto out_string = tokenizer->Decode(generated_tokens[i].data(), generated_tokens[i].size());
    std::cout << "Decoded string:" << out_string << std::endl;
  }

  // Verify outputs match expected outputs
  std::vector<std::vector<int32_t>> expected_output{
      {1212, 318, 257, 1332, 13, 198, 50280, 2, 16926, 1330,
       1635, 10412, 6617, 278, 6335, 32994, 21857, 13849, 38665, 82,
       21815, 1108, 9557, 40755, 27446, 2417, 6381, 6, 7131, 6,
       14870, 31314, 21411, 46009, 3974, 82, 1039, 889, 263, 3684},
      {49, 1381, 389, 7427, 17252, 0, 198, 50284, 37811, 628, 50256},
      {464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13,
       198, 50284, 37811, 628, 50256}};

  for (size_t i = 0; i < batch_size; i++) {
    ASSERT_LE(generated_tokens[i].size(), 40);
    EXPECT_EQ(expected_output[i].size(), generated_tokens[i].size());
    EXPECT_EQ(expected_output[i], generated_tokens[i]);
  }
}
#endif

TEST(CAPITests, EndToEndPhi) {
#if TEST_PHI2
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_string = "This is a test.";
  auto input_sequence = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequence);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 40);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequence);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Decode The Batch
  auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
  std::cout << "Decoded string:" << out_string << std::endl;

  // Verify outputs match expected outputs
  std::vector<int32_t> expected_output{
      1212, 318, 257, 1332, 13, 198, 50280, 2, 16926, 1330, 1635, 10412, 6617, 278,
      6335, 32994, 21857, 13849, 38665, 82, 21815, 1108, 9557, 40755, 27446, 2417,
      6381, 6, 7131, 6, 14870, 31314, 21411, 46009, 3974, 82, 1039, 889, 263, 3684};

  const auto sequence_length = generator->GetSequenceCount(0);
  const auto* sequence_data = generator->GetSequenceData(0);

  ASSERT_LE(sequence_length, 40);

  const auto* expected_output_start = &expected_output[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
#endif
}

TEST(CAPITests, EndToEndPhiEOSPAD) {
#if TEST_PHI2
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_string = "This is a test.<|endoftext|>";
  auto input_sequence = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequence);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 40);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequence);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Decode The Batch
  auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
  std::cout << "Decoded string:" << out_string << std::endl;

  // Verify outputs match expected outputs
  std::vector<int32_t> expected_output{
      1212, 318, 257, 1332, 13, 50256, 198, 198, 198, 198, 4010, 4420, 43168, 15666,
      10503, 82, 26268, 11451, 12735, 82, 19445, 427, 278, 49292, 3087, 26762, 5101,
      14453, 5421, 278, 829, 319, 8378, 8378, 10257, 82, 1028, 1028, 16219, 263};

  const auto sequence_length = generator->GetSequenceCount(0);
  const auto* sequence_data = generator->GetSequenceData(0);

  ASSERT_LE(sequence_length, 40);

  const auto* expected_output_start = &expected_output[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
#endif
}

#if ENABLE_ENGINE_TESTS
TEST(CAPIEngineTests, EndToEndPhi) {
  auto model = OgaModel::Create(PHI2_PATH);
  auto engine = OgaEngine::Create(*model);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto streaming_tokenizer = OgaTokenizerStream::Create(*tokenizer);

  const char* input_string = "This is a test.";
  auto input_sequence = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequence);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 40);
  auto request = OgaRequest::Create(*params);
  request->AddTokens(*input_sequence);

  engine->Add(*request);
  std::string out_string;
  std::vector<int32_t> generated_tokens(input_sequence->SequenceData(0), input_sequence->SequenceData(0) + input_sequence->SequenceCount(0));

  while (auto ready_request = engine->Step()) {
    while (ready_request->HasUnseenTokens()) {
      generated_tokens.push_back(ready_request->GetUnseenToken());
      out_string += streaming_tokenizer->Decode(generated_tokens.back());
    }
  }

  engine->Remove(*request);

  std::cout << "Decoded string:" << out_string << std::endl;

  // Verify outputs match expected outputs
  std::vector<int32_t> expected_output{
      1212, 318, 257, 1332, 13, 198, 50280, 2, 16926, 1330, 1635, 10412, 6617, 278,
      6335, 32994, 21857, 13849, 38665, 82, 21815, 1108, 9557, 40755, 27446, 2417,
      6381, 6, 7131, 6, 14870, 31314, 21411, 46009, 3974, 82, 1039, 889, 263, 3684};

  ASSERT_LE(generated_tokens.size(), 40);

  EXPECT_EQ(expected_output, generated_tokens);
}
#endif

TEST(CAPITests, LoadModelFromMemory) {
#if TEST_PHI2

  const char* model_path = PHI2_PATH "/model.onnx";
  std::ifstream model_file(model_path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(model_file.is_open()) << "Failed to open model file: " << model_path;
  std::streamsize size = model_file.tellg();
  model_file.seekg(0, std::ios::beg);
  std::vector<std::byte> model_data(size);
  model_file.read(reinterpret_cast<char*>(model_data.data()), size);

  auto config = OgaConfig::Create(PHI2_PATH);
  config->AddModelData("model.onnx", model_data);
  auto model = OgaModel::Create(*config);
  config->RemoveModelData("model.onnx");
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_string = "This is a test.";
  auto input_sequence = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequence);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 40);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequence);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Decode The Batch
  auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
  std::cout << "Decoded string:" << out_string << std::endl;

  // Verify outputs match expected outputs
  std::vector<int32_t> expected_output{
      1212, 318, 257, 1332, 13, 198, 50280, 2, 16926, 1330, 1635, 10412, 6617, 278,
      6335, 32994, 21857, 13849, 38665, 82, 21815, 1108, 9557, 40755, 27446, 2417,
      6381, 6, 7131, 6, 14870, 31314, 21411, 46009, 3974, 82, 1039, 889, 263, 3684};

  const auto sequence_length = generator->GetSequenceCount(0);
  const auto* sequence_data = generator->GetSequenceData(0);

  ASSERT_LE(sequence_length, 40);

  const auto* expected_output_start = &expected_output[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
#endif
}

TEST(CAPITests, Tensor_And_AddExtraInput) {
  // Create a [3 4] shaped tensor
  std::array<float, 12> data{0, 1, 2, 3,
                             10, 11, 12, 13,
                             20, 21, 22, 23};
  std::vector<int64_t> shape{3, 4};  // Use vector so we can easily compare for equality later

  auto tensor = OgaTensor::Create(data.data(), shape.data(), shape.size(), OgaElementType_float32);

  EXPECT_EQ(tensor->Data(), data.data());
  EXPECT_EQ(tensor->Shape(), shape);
  EXPECT_EQ(tensor->Type(), OgaElementType_float32);

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = OgaGeneratorParams::Create(*model);

  auto generator = OgaGenerator::Create(*model, *params);

  generator->SetModelInput("test_input", *tensor);
}

TEST(CAPITests, Logging) {
  // Trivial test to ensure the API builds properly
  Oga::SetLogBool("enabled", true);
  Oga::SetLogString("filename", nullptr);  // If we had a filename set, this would stop logging to the file and go back to the console
  Oga::SetLogBool("enabled", false);
}

// DML doesn't support GPT attention
#if !USE_DML
TEST(CAPITests, GreedySearchGptFp32CAPI) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    const auto sequence_length = generator->GetSequenceCount(i);
    const auto* sequence_data = generator->GetSequenceData(i);

    ASSERT_LE(sequence_length, max_length);

    const auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
  }
}
#endif

TEST(CAPITests, GetOutputCAPI) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  int batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());

  // check prompt
  // full logits has shape [2, 4, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 4, 5]
  std::vector<float> expected_sampled_logits_prompt{0.29694548f, 0.00955007f, 0.0430819f, 0.10063869f, 0.0437237f,
                                                    0.27329233f, 0.00841076f, -0.1060291f, 0.11328877f, 0.13369876f,
                                                    0.30323744f, 0.0545997f, 0.03894716f, 0.11702324f, 0.0410665f,
                                                    -0.12675379f, -0.04443946f, 0.14492269f, 0.03021223f, -0.03212897f,
                                                    0.29694548f, 0.00955007f, 0.0430819f, 0.10063869f, 0.0437237f,
                                                    0.27329233f, 0.00841076f, -0.1060291f, 0.11328877f, 0.13369876f,
                                                    -0.04699047f, 0.17915794f, 0.20838135f, 0.10888482f, -0.00277808f,
                                                    0.2938929f, -0.10538938f, -0.00226692f, 0.12050669f, -0.10622668f};

  auto prompt_logits_ptr = generator->GetOutput("logits");
  auto prompt_logits = static_cast<float*>(prompt_logits_ptr->Data());
  int num_prompt_outputs_to_check = 40;
  int sample_size = 200;
  float tolerance = 0.001f;
  // Verify outputs match expected outputs
  for (int i = 0; i < num_prompt_outputs_to_check; i++) {
    EXPECT_NEAR(expected_sampled_logits_prompt[i], prompt_logits[i * sample_size], tolerance);
  }

  generator->GenerateNextToken();
  generator->GenerateNextToken();
  // check for the 1st token generation
  // full logits has shape [2, 1, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
  std::vector<float> expected_sampled_logits_token_gen{0.03742531f, -0.05752287f, 0.14159015f, 0.04210977f, -0.1484456f,
                                                       0.3041716f, -0.08701379f, -0.03778192f, 0.07471392f, -0.02049096f};

  auto token_gen_logits_ptr = generator->GetOutput("logits");
  auto token_gen_logits = static_cast<float*>(token_gen_logits_ptr->Data());
  int num_token_gen_outputs_to_check = 10;

  for (int i = 0; i < num_token_gen_outputs_to_check; i++) {
    EXPECT_NEAR(expected_sampled_logits_token_gen[i], token_gen_logits[i * sample_size], tolerance);
  }
}

TEST(CAPITests, GetLogitsCAPI) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  int batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());

  // check prompt generation, GetLogits() returns last token logits
  // full logits has shape [2, 1, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
  std::vector<float> expected_sampled_logits_prompt{-0.12675379f, -0.04443946f, 0.14492269f, 0.03021223f, -0.03212897f,
                                                    0.2938929f, -0.10538938f, -0.00226692f, 0.12050669f, -0.10622668f};

  auto prompt_logits_ptr = generator->GetLogits();
  auto prompt_logits = reinterpret_cast<float*>(prompt_logits_ptr->Data());
  int num_prompt_outputs_to_check = 10;
  int sample_size = 200;
  float tolerance = 0.001f;
  // Verify outputs match expected outputs
  for (int i = 0; i < num_prompt_outputs_to_check; i++) {
    EXPECT_NEAR(expected_sampled_logits_prompt[i], prompt_logits[i * sample_size], tolerance);
  }

  generator->GenerateNextToken();
  // check for the 1st token generation
  // full logits has shape [2, 1, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
  std::vector<float> expected_sampled_logits_token_gen{0.03742531f, -0.05752287f, 0.14159015f, 0.04210977f, -0.1484456f,
                                                       0.3041716f, -0.08701379f, -0.03778192f, 0.07471392f, -0.02049096f};

  auto token_gen_logits_ptr = generator->GetLogits();
  auto token_gen_logits = reinterpret_cast<float*>(token_gen_logits_ptr->Data());
  int num_token_gen_outputs_to_check = 10;

  for (int i = 0; i < num_token_gen_outputs_to_check; i++) {
    EXPECT_NEAR(expected_sampled_logits_token_gen[i], token_gen_logits[i * sample_size], tolerance);
  }
}

TEST(CAPITests, SetLogitsCAPI) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  int batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());

  std::vector<float> expected_sampled_logits_prompt{0.29694548f, 0.00955007f, 0.0430819f, 0.10063869f, 0.0437237f};
  std::vector<float> dummy_logits(2 * 1000, 0.0f);
  for (int i = 0; i < dummy_logits.size(); i++) {
    dummy_logits[i] = expected_sampled_logits_prompt[i % expected_sampled_logits_prompt.size()];
  }
  std::vector<int64_t> dummy_logits_shape{2, 1, 1000};
  auto logits = OgaTensor::Create(dummy_logits.data(), dummy_logits_shape.data(), dummy_logits_shape.size(), OgaElementType_float32);
  auto raw_logits = generator->GetLogits();
  generator->SetLogits(*logits);
  auto retrieved_logits = generator->GetLogits();
  auto retrieved_data = reinterpret_cast<float*>(retrieved_logits->Data());
  for (int i = 0; i < dummy_logits.size(); i++) {
    EXPECT_EQ(dummy_logits[i], retrieved_data[i]);
  }
}

TEST(CAPITests, SetTerminate) {
#if TEST_PHI2

  auto GeneratorSetTerminateCall = [](OgaGenerator* generator) {
    // Set Terminate
    generator->SetRuntimeOption("terminate_session", "1");
  };

  auto GenerateOutput = [](OgaGenerator* generator, std::unique_ptr<OgaTokenizerStream> tokenizer_stream) {
    EXPECT_THROW({
      while (!generator->IsDone()) {
        generator->GenerateNextToken();
      } }, std::runtime_error);
  };

  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

  const char* input_string = "She sells sea shells by the sea shore.";
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 40);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  EXPECT_EQ(generator->IsSessionTerminated(), false);
  std::vector<std::thread> threads;
  threads.push_back(std::thread(GenerateOutput, generator.get(), std::move(tokenizer_stream)));
  threads.push_back(std::thread(GeneratorSetTerminateCall, generator.get()));

  for (auto& th : threads) {
    th.join();  // Wait for each thread to finish
  }
  EXPECT_EQ(generator->IsSessionTerminated(), true);
  // Unset terminate
  generator->SetRuntimeOption("terminate_session", "0");
  EXPECT_EQ(generator->IsSessionTerminated(), false);
#endif
}

// DML Doesn't support batch_size > 1
#if TEST_PHI2 && !USE_DML

struct Phi2Test {
  Phi2Test() {
    model_ = OgaModel::Create(PHI2_PATH);
    tokenizer_ = OgaTokenizer::Create(*model_);

    input_sequences_ = OgaSequences::Create();

    const char* input_strings[] = {
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    };

    for (auto& string : input_strings)
      tokenizer_->Encode(string, *input_sequences_);

    params_ = OgaGeneratorParams::Create(*model_);
    params_->SetSearchOption("max_length", 40);
    params_->SetSearchOption("batch_size", static_cast<int>(batch_size_));
  }

  void Run() {
    // Low level loop
    {
      auto generator = OgaGenerator::Create(*model_, *params_);
      generator->AppendTokenSequences(*input_sequences_);

      while (!generator->IsDone()) {
        generator->GenerateNextToken();
      }

      // Decode One at a time
      for (size_t i = 0; i < batch_size_; i++) {
        auto out_string = tokenizer_->Decode(generator->GetSequenceData(i), generator->GetSequenceCount(i));
        std::cout << "Decoded string:" << out_string << std::endl;
      }
    }
  }

  void RunEngine() {
    auto engine = OgaEngine::Create(*model_);
    constexpr size_t per_request_batch_size = 1;
    params_->SetSearchOption("batch_size", static_cast<int>(per_request_batch_size));

    std::vector<std::unique_ptr<OgaRequest>> requests_;
    std::array<std::vector<int32_t>, 3> generated_tokens;

    const char* input_strings[] = {
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    };

    for (size_t i = 0; i < batch_size_; i++) {
      auto input_sequence = OgaSequences::Create();
      tokenizer_->Encode(input_strings[i], *input_sequence);
      generated_tokens[i] = std::vector<int32_t>(input_sequence->SequenceData(0),
                                                 input_sequence->SequenceData(0) + input_sequence->SequenceCount(0));
      requests_.emplace_back(OgaRequest::Create(*params_));
      requests_.back()->AddTokens(*input_sequence);
      requests_.back()->SetOpaqueData(&generated_tokens[i]);

      engine->Add(*requests_.back());
    }

    while (auto request = engine->Step()) {
      while (request->HasUnseenTokens()) {
        auto* tokens = reinterpret_cast<std::vector<int32_t>*>(request->GetOpaqueData());
        tokens->push_back(request->GetUnseenToken());
      }
    }

    for (size_t i = 0; i < batch_size_; i++) {
      auto out_string = tokenizer_->Decode(generated_tokens[i].data(), generated_tokens[i].size());
      std::cout << "Decoded string:" << out_string << std::endl;
    }
  }

  std::unique_ptr<OgaModel> model_;
  std::unique_ptr<OgaTokenizer> tokenizer_;
  std::unique_ptr<OgaSequences> input_sequences_;
  std::unique_ptr<OgaGeneratorParams> params_;
  const size_t batch_size_ = 3;
};

class ParametrizedTopKCAPITestsTests : public ::testing::TestWithParam<bool> {
};

TEST_P(ParametrizedTopKCAPITestsTests, TopKCAPI) {
  Phi2Test test;

  test.params_->SetSearchOptionBool("do_sample", true);
  test.params_->SetSearchOption("top_k", 50);
  test.params_->SetSearchOption("temperature", 0.6f);

  if (GetParam()) {
    test.RunEngine();
  } else {
    test.Run();
  }
}

INSTANTIATE_TEST_SUITE_P(TopKCAPI,
                         ParametrizedTopKCAPITestsTests,
                         ::testing::Values(false, true));

class ParametrizedTopPCAPITestsTests : public ::testing::TestWithParam<bool> {
};

TEST_P(ParametrizedTopPCAPITestsTests, TopPCAPI) {
  Phi2Test test;

  test.params_->SetSearchOptionBool("do_sample", true);
  test.params_->SetSearchOption("top_p", 0.6f);
  test.params_->SetSearchOption("temperature", 0.6f);

  if (GetParam()) {
    test.RunEngine();
  } else {
    test.Run();
  }
}

INSTANTIATE_TEST_SUITE_P(TopPCAPI,
                         ParametrizedTopPCAPITestsTests,
                         ::testing::Values(false, true));

class ParametrizedTopKTopPCAPITestsTests : public ::testing::TestWithParam<bool> {
};

TEST_P(ParametrizedTopKTopPCAPITestsTests, TopKCAPITest) {
  Phi2Test test;

  test.params_->SetSearchOptionBool("do_sample", true);
  test.params_->SetSearchOption("top_k", 50);
  test.params_->SetSearchOption("top_p", 0.6f);
  test.params_->SetSearchOption("temperature", 0.6f);

  if (GetParam()) {
    test.RunEngine();
  } else {
    test.Run();
  }
}

INSTANTIATE_TEST_SUITE_P(TopKCAPITest,
                         ParametrizedTopKTopPCAPITestsTests,
                         ::testing::Values(false, true));

TEST(CAPITests, AdaptersTest) {
#ifdef USE_CUDA
  using OutputType = Ort::Float16_t;
#else
  using OutputType = float;
#endif

  // The python unit tests create the adapter model.
  // In order to run this test, the python unit test must have been run first.
  auto model = OgaModel::Create(MODEL_PATH "adapters");
  auto adapters = OgaAdapters::Create(*model);
  adapters->LoadAdapter(MODEL_PATH "adapters/adapters.onnx_adapter", "adapters_a_and_b");

  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  auto input_sequences = OgaSequences::Create();
  for (auto& string : input_strings)
    tokenizer->Encode(string, *input_sequences);

  // Run base scenario
  size_t output_size = 0;
  std::vector<int64_t> output_shape;
  std::vector<OutputType> base_output;
  {
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 20);
    params->SetSearchOption("batch_size", 3);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokenSequences(*input_sequences);

    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }

    auto logits = generator->GetOutput("logits");
    output_shape = logits->Shape();
    output_size = static_cast<size_t>(std::accumulate(output_shape.begin(), output_shape.end(), 1LL,
                                                      std::multiplies<int64_t>()));
    base_output.reserve(output_size);
    std::span<const OutputType> src(reinterpret_cast<const OutputType*>(logits->Data()), output_size);
    std::copy(src.begin(), src.end(), std::back_inserter(base_output));
  }
  // Run scenario with an adapter
  // We are expecting a difference in output
  {
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 20);
    params->SetSearchOption("batch_size", 3);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetActiveAdapter(*adapters, "adapters_a_and_b");
    generator->AppendTokenSequences(*input_sequences);

    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }

    auto logits = generator->GetOutput("logits");
    const auto shape = logits->Shape();
    // Expecting the same shape
    ASSERT_TRUE(std::equal(output_shape.begin(), output_shape.end(), shape.begin(), shape.end()));

    const auto size = static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), 1LL,
                                                          std::multiplies<int64_t>()));
    ASSERT_EQ(output_size, size);
    std::span<const OutputType> src(reinterpret_cast<const OutputType*>(logits->Data()), size);
    ASSERT_FALSE(std::equal(base_output.begin(), base_output.end(), src.begin(), src.end()));
  }

  // Unload the adapter. Will error out if the adapter is still active.
  // So, the generator must go out of scope before the adapter can be unloaded.
  adapters->UnloadAdapter("adapters_a_and_b");
}

TEST(CAPITests, AdaptersTestMultipleAdapters) {
  // The python unit tests create the adapter model.
  // In order to run this test, the python unit test must have been run first.
  auto model = OgaModel::Create(MODEL_PATH "multiple_adapters");
  auto adapters = OgaAdapters::Create(*model);
  adapters->LoadAdapter(MODEL_PATH "multiple_adapters/adapter_0.onnx_adapter", "adapter_a");
  adapters->LoadAdapter(MODEL_PATH "multiple_adapters/adapter_1.onnx_adapter", "adapter_b");

  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  auto input_sequences = OgaSequences::Create();
  for (auto& string : input_strings)
    tokenizer->Encode(string, *input_sequences);

  {
    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 20);
    params->SetSearchOption("batch_size", 3);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->SetActiveAdapter(*adapters, "adapter_a");
    generator->SetActiveAdapter(*adapters, "adapter_b");
    generator->AppendTokenSequences(*input_sequences);

    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }
  }

  // Unload the adapter. Will error out if the adapter is still active.
  // So, the generator must go out of scope before the adapter can be unloaded.
  adapters->UnloadAdapter("adapter_a");
  adapters->UnloadAdapter("adapter_b");
}
#endif  // TEST_PHI2 && !USE_DML

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

#if !USE_DML
TEST(CAPITests, BatchedRewindGptFp32CAPI) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    const auto sequence_length = generator->GetSequenceCount(i);
    const auto* sequence_data = generator->GetSequenceData(i);

    ASSERT_LE(sequence_length, max_length);

    const auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
  }

  // Rewind to length 0 and verify same output
  generator->RewindTo(0);

  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    const auto sequence_length = generator->GetSequenceCount(i);
    const auto* sequence_data = generator->GetSequenceData(i);

    ASSERT_LE(sequence_length, max_length);

    const auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
  }
}

TEST(CAPITests, RewindGptFp32CAPI) {
  std::vector<int64_t> input_ids_shape{1, 4};
  std::vector<int32_t> input_ids{0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  int max_length = 10;

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  auto sequence_length = generator->GetSequenceCount(0);
  auto* sequence_data = generator->GetSequenceData(0);

  ASSERT_LE(sequence_length, max_length);
  auto* expected_output_start = &expected_output[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));

  // Rewind to length 5 and verify same output
  generator->RewindTo(5);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  sequence_length = generator->GetSequenceCount(0);
  sequence_data = generator->GetSequenceData(0);
  ASSERT_LE(sequence_length, max_length);
  expected_output_start = &expected_output[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));

  // Rewind to length 3 and add tokens and verify same output
  generator->RewindTo(3);

  std::vector<int32_t> next_ids{731, 731};
  generator->AppendTokens(next_ids.data(), next_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  sequence_length = generator->GetSequenceCount(0);
  sequence_data = generator->GetSequenceData(0);
  ASSERT_LE(sequence_length, max_length);
  expected_output_start = &expected_output[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
}
#endif

#if USE_GUIDANCE
TEST(CAPITests, SetGuidance) {
#if TEST_PHI2

  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

  const char* input_string = "who are you?";
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 32);
  params->SetGuidance("regex", "answer: .*");

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
  auto output = std::string(out_string).substr(std::string(input_string).size());
  EXPECT_TRUE(std::regex_match(output, std::regex("answer: .*")));

#endif
}
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>  // for memcmp
#include <iostream>
#include <random>
#include <gtest/gtest.h>

#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>
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

// This model was generated using the model builder, but the config.json of the model had to be adjusted to have num_attention_heads and num_key_value_heads be 1
// to make the GroupQueryAttention op not complain about the head_size not being a multiple of 16.
// We use this tiny trivial file to do basic unit tests.
#define LLAMA_FP16_PATH MODEL_PATH "hf-internal-testing/tiny-random-LlamaForCausalLM-fp16"
#define LLAMA_FP32_PATH MODEL_PATH "hf-internal-testing/tiny-random-LlamaForCausalLM-fp32"

static const std::pair<const char*, const char*> c_tiny_llama_model_paths[] = {
    {LLAMA_FP32_PATH, "fp32"},
    {LLAMA_FP16_PATH, "fp16"},
};

#if USE_DML
TEST(ModelTests, DMLAdapterSelection) {
#if 0 // TEST_PHI2 TODO: Remove this? Can't access the device directly anymore.
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), PHI2_PATH);
  auto d3d12Device = model->GetD3D12Device();

  auto adapterLuid = d3d12Device->GetAdapterLuid();  
  for (const auto& provider_option: model->config_->model.decoder.session_options.provider_options) {
    if (provider_option.name == "dml") {
      for (const auto& [name, value] : provider_option.options) {
        if (name == "luid") {
          if (auto separator_position = value.find(":"); separator_position != std::string::npos) {
            EXPECT_EQ(adapterLuid.HighPart, std::stol(value.substr(0, separator_position)));
            EXPECT_EQ(adapterLuid.LowPart, std::stoul(value.substr(separator_position + 1)));
          }
        }
      }
    }
  }
#endif
}
#endif

// DML doesn't support GPT attention
#if !USE_DML
TEST(ModelTests, GreedySearchLlamaFp32) {
  std::array<int64_t, 2> input_ids_shape{2, 4};
  std::array<int32_t, 8> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::array<int32_t, 20> expected_output{
      0, 0, 0, 52, 12102, 30463, 4666, 17192, 3266, 18061,
      0, 0, 195, 731, 29592, 4877, 18112, 22607, 12936, 997};

  auto model = OgaModel::Create(LLAMA_FP32_PATH);

  auto params = OgaGeneratorParams::Create(*model);
  int max_length = 10;
  int batch_size = static_cast<int>(input_ids_shape[0]);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (size_t i = 0; i < static_cast<size_t>(batch_size); i++) {
    auto sequence = generator->GetSequence(i);
    auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));
  }
}

TEST(ModelTests, BeamSearchLlamaFp32) {
  std::array<int64_t, 2> input_ids_shape{3, 12};
  std::array<int32_t, 36> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::array<int32_t, 60> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 23145, 13392, 2754, 10010, 29136, 6868, 12230, 23861,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 1810, 4702, 29668, 11709, 27728, 12420, 616, 10337,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 21955, 5331, 12815, 1404, 5661, 18625, 10014, 8136};

  auto model = OgaModel::Create(LLAMA_FP32_PATH);

  int max_length = 20;
  int batch_size = static_cast<int>(input_ids_shape[0]);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);
  params->SetSearchOption("num_beams", 4);
  params->SetSearchOption("length_penalty", 1.0f);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    auto sequence = generator->GetSequence(i);
    auto* expected_output_start = &expected_output[static_cast<size_t>(i) * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));
  }
}
#endif

#if USE_CUDA

void Test_GreedySearch_Llama_Cuda(const char* model_path, const char* model_label) {
  std::array<int64_t, 2> input_ids_shape{2, 4};
  std::array<int32_t, 8> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::array<int32_t, 20> expected_output{
      0, 0, 0, 52, 12102, 30463, 4666, 17192, 3266, 18061,
      0, 0, 195, 731, 29592, 4877, 18112, 22607, 12936, 997};

  auto config = OgaConfig::Create(model_path);
  config->ClearProviders();
  config->AppendProvider("cuda");
  auto model = OgaModel::Create(*config);

  int max_length = 10;
  int batch_size = static_cast<int>(input_ids_shape[0]);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    auto sequence = generator->GetSequence(i);
    auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));
  }

  // Test batch size 1 continuous case
  input_ids_shape = {1, 4};

  batch_size = 1;
  params->SetSearchOption("batch_size", batch_size);

  generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(std::span(input_ids).subspan(4, 4));

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  
  // Verify outputs match expected outputs
  auto sequence = generator->GetSequence(0);
  auto* expected_output_start = &expected_output[max_length];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));

  generator->RewindTo(3);
  generator->AppendTokens(std::array<int32_t, 1>{731});
  generator->AppendTokens(std::array<int32_t, 1>{29592});
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  sequence = generator->GetSequence(0);
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));
}

TEST(ModelTests, GreedySearchLlamaCuda) {
  for (auto model_path : c_tiny_llama_model_paths)
    Test_GreedySearch_Llama_Cuda(model_path.first, model_path.second);
}

void Test_BeamSearch_Llama_Cuda(const char* model_path, const char* model_label) {
  std::array<int64_t, 2> input_ids_shape{3, 12};
  std::array<int32_t, 36> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::array<int32_t, 60> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 23145, 13392, 2754, 10010, 29136, 6868, 12230, 23861,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 1810, 4702, 29668, 11709, 27728, 12420, 616, 10337,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 21955, 5331, 12815, 1404, 5661, 18625, 10014, 8136};

  auto config = OgaConfig::Create(model_path);
  config->ClearProviders();
  config->AppendProvider("cuda");
  auto model = OgaModel::Create(*config);

  int batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 20;

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);
  params->SetSearchOption("num_beams", 4);
  params->SetSearchOption("length_penalty", 1.0f);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    auto sequence = generator->GetSequence(i);
    auto* expected_output_start = &expected_output[static_cast<size_t>(i) * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));
  }
}

TEST(ModelTests, BeamSearchLlamaCuda) {
  for (auto model_path : c_tiny_llama_model_paths)
    Test_BeamSearch_Llama_Cuda(model_path.first, model_path.second);
}
#endif

#if TEST_PHI2 && (USE_CUDA || USE_DML)
TEST(ModelTests, TestApiDevice) {
  auto prompt = R"(
def print_prime(n):
'''
Print all primes between 1 and n
'''
)";

  std::cout << "With prompt:" << prompt << "\r\n";

  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokens = OgaSequences::Create();
  tokenizer->Encode(prompt, *tokens);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 128);
  params->SetSearchOption("batch_size", 1);  // Redundant, but for testing

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(tokens->Get(0));
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto result = generator->GetSequence(0);

  std::cout << tokenizer->Decode(result) << "\r\n";
}

TEST(ModelTests, TestTopKDevice) {
  auto prompt = R"(
def print_prime(n):
'''
Print all primes between 1 and n
'''
)";

  std::cout << "With prompt:" << prompt << "\r\n";

  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto tokens = OgaSequences::Create();
  tokenizer->Encode(prompt, *tokens);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 128);
  params->SetSearchOption("batch_size", 1);  // Redundant, but for testing
  params->SetSearchOption("top_k", 3);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(tokens->Get(0));
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto result = generator->GetSequence(0);

  std::cout << tokenizer->Decode(result) << "\r\n";
}
#endif
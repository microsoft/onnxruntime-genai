// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>  // for memcmp
#include <iostream>
#include <random>
#include <filesystem>
#include <string>
#include <gtest/gtest.h>

#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>
#include <gtest/gtest.h>

#include "test_utils.h"

// External global variable from main.cpp for custom model path
extern std::string g_custom_model_path;

// To generate this file:
// python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
// And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)
static const std::pair<const char*, const char*> c_tiny_gpt2_model_paths[] = {
    {MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32-cuda", "fp32"},
    {MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp16-cuda", "fp16"},
};

#if USE_DML
TEST(ModelTests, DMLAdapterSelection) {
#if 0  // TEST_PHI2 TODO: Remove this? Can't access the device directly anymore.
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
TEST(ModelTests, GreedySearchGptFp32) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)
  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

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

TEST(ModelTests, BeamSearchGptFp32) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

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

void Test_GreedySearch_Gpt_Cuda(const char* model_path, const char* model_label) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto model = OgaModel::Create(model_path);

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
  input_ids = {0, 0, 195, 731};
  std::vector<int32_t> expected_output_continuous{0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  batch_size = static_cast<int>(input_ids_shape[0]);
  params->SetSearchOption("batch_size", batch_size);

  generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  auto sequence = generator->GetSequence(0);
  auto* expected_output_start = &expected_output_continuous[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));

  generator->RewindTo(3);
  std::vector<int32_t> next_ids{731, 731};
  generator->AppendTokens(next_ids);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  sequence = generator->GetSequence(0);
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), max_length * sizeof(int32_t)));
}

TEST(ModelTests, GreedySearchGptCuda) {
  for (auto model_path : c_tiny_gpt2_model_paths)
    Test_GreedySearch_Gpt_Cuda(model_path.first, model_path.second);
}

void Test_BeamSearch_Gpt_Cuda(const char* model_path, const char* model_label) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
  auto model = OgaModel::Create(model_path);

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

TEST(ModelTests, BeamSearchGptCuda) {
  for (auto model_path : c_tiny_gpt2_model_paths)
    Test_BeamSearch_Gpt_Cuda(model_path.first, model_path.second);
}

// Tiny GQA model for CUDA graph capture tests (GQA + past_present_share_buffer).
// Generated by test/create_gqa_model.py
static const char* c_cuda_graph_model_path = MODEL_PATH "hf-internal-testing/tiny-qwen35-cuda";

// Test that CUDA graph capture produces correct outputs for greedy search.
// Validates generation completes without CUDA errors under graph capture.
TEST(ModelTests, GreedySearchCudaGraphCapture) {
  if (!std::filesystem::exists(c_cuda_graph_model_path)) {
    GTEST_SKIP() << "CUDA graph capture model not available at: " << c_cuda_graph_model_path;
  }

  std::vector<int32_t> input_ids{10, 20, 30, 40};
  int max_length = 10;

  auto model = OgaModel::Create(c_cuda_graph_model_path);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", 1);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto seq = generator->GetSequence(0);
  EXPECT_EQ(seq.size(), static_cast<size_t>(max_length));
}

// Test continuous decoding (RewindTo) with CUDA graph capture enabled.
TEST(ModelTests, ContinuousDecodingCudaGraphCapture) {
  if (!std::filesystem::exists(c_cuda_graph_model_path)) {
    GTEST_SKIP() << "CUDA graph capture model not available at: " << c_cuda_graph_model_path;
  }

  std::vector<int32_t> input_ids{10, 20, 30, 40};
  int max_length = 10;

  auto model = OgaModel::Create(c_cuda_graph_model_path);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", 1);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto seq1 = generator->GetSequence(0);
  std::vector<int32_t> first_output(seq1.begin(), seq1.end());

  // Rewind and regenerate — should produce same output
  size_t rewind_pos = input_ids.size() - 1;
  generator->RewindTo(rewind_pos);
  std::vector<int32_t> resume_ids(input_ids.begin() + rewind_pos, input_ids.end());
  generator->AppendTokens(resume_ids);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto seq2 = generator->GetSequence(0);
  ASSERT_EQ(seq2.size(), first_output.size());
  EXPECT_TRUE(0 == std::memcmp(first_output.data(), seq2.data(), seq2.size() * sizeof(int32_t)))
      << "Continuous decoding output differs after RewindTo with graph capture";
}
#endif

// NvTensorRT test cases using Phi3 models
static const std::pair<const char*, const char*> c_phi3_nvtrt_model_paths[] = {
    {MODEL_PATH "hf-internal-testing/phi3-fp16-nvtrt", "fp16"},
};

void Test_GreedySearch_Phi3_NvTensorRtRtx(const char* model_path, const char* model_label) {
  // Use custom path if provided, otherwise use default
  std::string resolved_path = g_custom_model_path.empty() ? model_path : g_custom_model_path;

  // Skip test if NvTensorRT model is not available
  if (!std::filesystem::exists(resolved_path)) {
    GTEST_SKIP() << "NvTensorRT model not available at: " << resolved_path;
  }
  const std::vector<int64_t> input_ids_shape{1, 19};
  const std::vector<int32_t> input_ids{32006, 887, 526, 263, 8444, 29871, 23869, 20255, 29889, 32007, 32010, 6324, 29892, 1128, 526, 366, 29973, 32007, 32001};

  // Complete expected sequence (input + generated) from model_qa.cpp using the working phi3-fp16-nvtrt model
  const std::vector<int32_t> expected_output{
      32006, 887, 526, 263, 8444, 29871, 23869, 20255, 29889, 32007, 32010, 6324, 29892, 1128, 526, 366, 29973, 32007, 32001,  // Input tokens (19)
      15043, 29991, 306, 29915, 29885, 2599};
  auto config = OgaConfig::Create(resolved_path.c_str());
  config->ClearProviders();
  config->AppendProvider("NvTensorRtRtx");
  auto model = OgaModel::Create(*config);

  constexpr int max_length = 25;
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
}

TEST(ModelTests, GreedySearchPhi3NvTensorRtRtx) {
  for (auto model_path : c_phi3_nvtrt_model_paths)
    Test_GreedySearch_Phi3_NvTensorRtRtx(model_path.first, model_path.second);
}

void Test_OutOfPlaceKvCache_Phi3_NvTensorRtRtx(const char* model_path, const char* model_label) {
  // Use custom path if provided, otherwise use default
  std::string resolved_path = g_custom_model_path.empty() ? model_path : g_custom_model_path;

  // Skip test if NvTensorRT model is not available
  if (!std::filesystem::exists(resolved_path)) {
    GTEST_SKIP() << "NvTensorRT model not available at: " << resolved_path;
  }

  const std::vector<int64_t> input_ids_shape{1, 19};
  const std::vector<int32_t> input_ids{
      32006, 887, 526, 263, 8444, 29871, 23869, 20255, 29889,
      32007, 32010, 6324, 29892, 1128, 526, 366, 29973, 32007, 32001};

  // Expected output sequence (input + generated tokens) for validation with greedy search
  const std::vector<int32_t> expected_output{
      32006, 887, 526, 263, 8444, 29871, 23869, 20255, 29889, 32007, 32010, 6324, 29892, 1128, 526, 366, 29973, 32007, 32001,  // Input tokens (19)
      15043, 1554, 13, 16271, 29892, 8733};

  auto config = OgaConfig::Create(resolved_path.c_str());
  config->ClearProviders();
  config->AppendProvider("NvTensorRtRtx");
  auto model = OgaModel::Create(*config);

  constexpr int max_length = 25;
  int batch_size = static_cast<int>(input_ids_shape[0]);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);
  params->SetSearchOptionBool("past_present_share_buffer", false);
  params->SetSearchOptionBool("do_sample", false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto sequence = generator->GetSequence(0);

  // Verify output matches expected output
  EXPECT_EQ(sequence.size(), expected_output.size());
  EXPECT_TRUE(0 == std::memcmp(expected_output.data(), sequence.data(),
                               expected_output.size() * sizeof(int32_t)));
}

TEST(ModelTests, OutOfPlaceKvCachePhi3NvTensorRtRtx) {
  for (auto model_path : c_phi3_nvtrt_model_paths)
    Test_OutOfPlaceKvCache_Phi3_NvTensorRtRtx(model_path.first, model_path.second);
}

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
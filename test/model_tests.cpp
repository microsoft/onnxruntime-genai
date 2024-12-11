// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <logits_processor.h>
#include <search.h>
#include <models/model.h>
#include <iostream>
#include <random>
#include <smartptrs.h>
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
#if USE_DML
#include <DirectML.h>
#include <wrl.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#endif

// To generate this file:
// python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
// And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)
static const std::pair<const char*, const char*> c_tiny_gpt2_model_paths[] = {
    {MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32-cuda", "fp32"},
    {MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp16-cuda", "fp16"},
};

#if USE_DML
TEST(ModelTests, DMLAdapterSelection) {
#if TEST_PHI2
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
  auto model = Generators::CreateModel(Generators::GetOrtEnv(),
                                       MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = Generators::CreateGeneratorParams(*model);
  params->search.max_length = 10;
  params->search.batch_size = static_cast<int>(input_ids_shape[0]);

  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<int>(input_ids.data(), input_ids.size()));

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (size_t i = 0; i < static_cast<size_t>(params->search.batch_size); i++) {
    auto sequence = generator->GetSequence(i).CopyDeviceToCpu();
    auto* expected_output_start = &expected_output[i * params->search.max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), params->search.max_length * sizeof(int32_t)));
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

  auto model = Generators::CreateModel(Generators::GetOrtEnv(), MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = Generators::CreateGeneratorParams(*model);
  params->search.batch_size = static_cast<int>(input_ids_shape[0]);
  params->search.max_length = 20;
  params->search.length_penalty = 1.0f;
  params->search.num_beams = 4;

  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<int>(input_ids.data(), input_ids.size()));
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < params->search.batch_size; i++) {
    auto sequence = generator->GetSequence(i).CpuSpan();
    auto* expected_output_start = &expected_output[static_cast<size_t>(i) * params->search.max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), params->search.max_length * sizeof(int32_t)));
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

  auto model = Generators::CreateModel(Generators::GetOrtEnv(), model_path);

  auto params = Generators::CreateGeneratorParams(*model);
  params->search.batch_size = static_cast<int>(input_ids_shape[0]);
  params->search.max_length = 10;

  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<int>(input_ids.data(), input_ids.size()));

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < params->search.batch_size; i++) {
    auto sequence_gpu = generator->GetSequence(i);
    auto sequence = sequence_gpu.CopyDeviceToCpu();
    auto* expected_output_start = &expected_output[i * params->search.max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), params->search.max_length * sizeof(int32_t)));
  }

  // Test batch size 1 continuous case
  input_ids_shape = {1, 4};
  input_ids = {0, 0, 195, 731};
  std::vector<int32_t> expected_output_continuous{0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  params->search.batch_size = static_cast<int>(input_ids_shape[0]);
  generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<int>(input_ids.data(), input_ids.size()));

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }
  
  // Verify outputs match expected outputs
  auto sequence_gpu = generator->GetSequence(0);
  auto sequence = sequence_gpu.CopyDeviceToCpu();
  auto* expected_output_start = &expected_output_continuous[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), params->search.max_length * sizeof(int32_t)));

  generator->RewindToLength(3);
  std::vector<int32_t> next_ids{731, 731};
  generator->AppendTokens(Generators::cpu_span<int>(next_ids.data(), next_ids.size()));
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  sequence_gpu = generator->GetSequence(0);
  sequence = sequence_gpu.CopyDeviceToCpu();
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), params->search.max_length * sizeof(int32_t)));
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
  auto model = Generators::CreateModel(Generators::GetOrtEnv(), model_path);

  auto params = Generators::CreateGeneratorParams(*model);
  params->search.batch_size = static_cast<int>(input_ids_shape[0]);
  params->search.max_length = 20;
  params->search.num_beams = 4;
  params->search.length_penalty = 1.0f;

  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<int>(input_ids.data(), input_ids.size()));
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < params->search.batch_size; i++) {
    auto sequence = generator->GetSequence(i).CopyDeviceToCpu();
    auto* expected_output_start = &expected_output[static_cast<size_t>(i) * params->search.max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), params->search.max_length * sizeof(int32_t)));
  }
}

TEST(ModelTests, BeamSearchGptCuda) {
  for (auto model_path : c_tiny_gpt2_model_paths)
    Test_BeamSearch_Gpt_Cuda(model_path.first, model_path.second);
}

TEST(ModelTests, TestApiCuda) {
#if TEST_PHI2

  auto prompt = R"(
def print_prime(n):
'''
Print all primes between 1 and n
'''
)";

  std::cout << "With prompt:" << prompt << "\r\n";

  auto model = Generators::CreateModel(Generators::GetOrtEnv(), PHI2_PATH);
  auto tokenizer = model->CreateTokenizer();
  auto tokens = tokenizer->Encode(prompt);

  auto params = Generators::CreateGeneratorParams(*model);
  params->search.batch_size = 1;
  params->search.max_length = 128;

  // Generator version
  auto generator = Generators::CreateGenerator(*model, *params);
  generator->AppendTokens(Generators::cpu_span<int>(tokens.data(), tokens.size()));
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto result = generator->GetSequence(0);

  std::cout << tokenizer->Decode(result.CopyDeviceToCpu()) << "\r\n";
#endif
}

#endif
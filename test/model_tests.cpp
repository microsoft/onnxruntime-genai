// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <iostream>
#include <random>
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

// To generate this file:
// python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
// And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)
static const std::pair<const char*, const char*> c_tiny_gpt2_model_paths[] = {
    {MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32-cuda", "fp32"},
    {MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp16-cuda", "fp16"},
};

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
  params->batch_size = static_cast<int>(input_ids_shape[0]);
  params->sequence_length = static_cast<int>(input_ids_shape[1]);
  params->input_ids = input_ids;

  auto generator = Generators::CreateGenerator(*model, *params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (size_t i = 0; i < static_cast<size_t>(params->batch_size); i++) {
    auto sequence = generator->GetSequence(i).CpuSpan();
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
  params->batch_size = static_cast<int>(input_ids_shape[0]);
  params->sequence_length = static_cast<int>(input_ids_shape[1]);
  params->input_ids = input_ids;
  params->search.max_length = 20;
  params->search.length_penalty = 1.0f;
  params->search.num_beams = 4;

  auto generator = Generators::CreateGenerator(*model, *params);
  auto result = Generators::Generate(*model, *params);

  // Verify outputs match expected outputs
  for (int i = 0; i < params->batch_size; i++) {
    auto sequence = std::span<int32_t>(result[i].data(), params->search.max_length);
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
  params->batch_size = static_cast<int>(input_ids_shape[0]);
  params->sequence_length = static_cast<int>(input_ids_shape[1]);
  params->search.max_length = 10;
  params->input_ids = input_ids;

  auto generator = Generators::CreateGenerator(*model, *params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();
    generator->GenerateNextToken();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < params->batch_size; i++) {
    auto sequence_gpu = generator->GetSequence(i);
    auto sequence = sequence_gpu.CpuSpan();
    auto* expected_output_start = &expected_output[i * params->search.max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence.data(), params->search.max_length * sizeof(int32_t)));
  }
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
  params->batch_size = static_cast<int>(input_ids_shape[0]);
  params->sequence_length = static_cast<int>(input_ids_shape[1]);
  params->input_ids = input_ids;
  params->search.max_length = 20;
  params->search.num_beams = 4;
  params->search.length_penalty = 1.0f;

  auto generator = Generators::CreateGenerator(*model, *params);
  auto result = Generators::Generate(*model, *params);

  // Verify outputs match expected outputs
  for (int i = 0; i < params->batch_size; i++) {
    auto sequence = std::span<int32_t>(result[i].data(), params->search.max_length);
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
  params->batch_size = 1;
  params->sequence_length = static_cast<int>(tokens.size());
  params->input_ids = tokens;
  params->search.max_length = 128;

  // Generator version
  auto generator = Generators::CreateGenerator(*model, *params);
  while (!generator->IsDone()) {
    generator->ComputeLogits();
    generator->GenerateNextToken();
  }

  auto result = generator->GetSequence(0);

  std::cout << tokenizer->Decode(result.GetCPU()) << "\r\n";
#endif
}

TEST(ModelTests, TestHighLevelApiCuda) {
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
  params->batch_size = 1;
  params->sequence_length = static_cast<int>(tokens.size());
  params->input_ids = tokens;
  params->search.max_length = 128;

  // High level version
  auto result = Generators::Generate(*model, *params);

  std::cout << tokenizer->Decode(result[0]) << "\r\n";
#endif
}

#endif
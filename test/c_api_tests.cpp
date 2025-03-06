// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>  // for memcmp
#include <numeric>
#include <iostream>
#include <thread>
#include <vector>
#include "span.h"

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

#define LLAMA_FP32_PATH MODEL_PATH "hf-internal-testing/tiny-random-LlamaForCausalLM-fp32"

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
  std::array<int32_t, 5> input_ids_0{1, 2, 3, 5, 8};
  std::array<int32_t, 5> input_ids_1{13, 21, 34, 55, 89};

  int max_length = 7;

  auto model = OgaModel::Create(LLAMA_FP32_PATH);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids_0);
  EXPECT_THROW(generator->AppendTokens(input_ids_1), std::runtime_error);

#if !USE_DML
  // Batch size 3 case
  std::array<int32_t, 30> input_ids_2{1, 2, 3, 5, 8, 13, 21, 34, 55, 89,
                                   0, 0, 0, 52, 104, 52, 53, 54, 55, 56,
                                   0, 0, 195, 731, 731, 195, 64, 45, 23, 12};
  params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", 3);

  generator = OgaGenerator::Create(*model, *params);
  EXPECT_THROW(generator->AppendTokens(input_ids_2), std::runtime_error);
#endif
}

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
  std::array<int32_t, 120> expected_output{
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

TEST(CAPITests, EndToEndPhi) {
#if TEST_PHI2
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_strings[] = {
      "This is a test."};

  auto input_sequences = OgaSequences::Create();
  for (auto& string : input_strings)
    tokenizer->Encode(string, *input_sequences);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 40);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);

  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Decode The Batch
  auto out_string = tokenizer->Decode(generator->GetSequenceData(0), generator->GetSequenceCount(0));
  std::cout << "Decoded string:" << out_string << std::endl;

  // Verify outputs match expected outputs
  std::array<int32_t, 40> expected_output{
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

  auto model = OgaModel::Create(LLAMA_FP32_PATH);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetModelInput("test_input", *tensor);
}

TEST(CAPITests, Logging) {
  // Trivial test to ensure the API builds properly
  Oga::SetLogBool("enabled", true);
  Oga::SetLogString("filename", nullptr);  // If we had a filename set, this would stop logging to the file and go back to the console
  Oga::SetLogBool("enabled", false);
}

// DML doesn't support GPT attention
#if !USE_DML
TEST(CAPITests, GreedySearchLlamaFp32CAPI) {
  std::array<int64_t, 2> input_ids_shape{2, 4};
  std::array<int32_t, 8> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::array<int32_t, 20> expected_output{
      0, 0, 0, 52, 12102, 30463, 4666, 17192, 3266, 18061,
      0, 0, 195, 731, 29592, 4877, 18112, 22607, 12936, 997};

  auto batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  auto model = OgaModel::Create(LLAMA_FP32_PATH);
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
    const auto sequence_length = generator->GetSequenceCount(i);
    const auto* sequence_data = generator->GetSequenceData(i);

    ASSERT_LE(sequence_length, max_length);

    const auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
  }
}
#endif

TEST(CAPITests, GetOutputCAPI) {
  std::array<int64_t, 2> input_ids_shape{2, 4};
  std::array<int32_t, 10> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  int batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  auto model = OgaModel::Create(LLAMA_FP32_PATH);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  // check prompt
  // full logits has shape [2, 4, vocab_size]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 4, 5]
  std::array<float, 40> expected_sampled_logits_prompt{-0.0682238f, 0.0405136f, 0.057766f, -0.0431961f, 0.00696388f,
                                                       -0.0153187f, 0.0369705f, 0.0259072f, -0.0189864f, 0.010939f,
                                                       -0.007559f, 0.0976457f, -0.0195211f, -0.0496172f, -0.0826776f,
                                                       -0.061368f, 0.0905409f, 0.0395047f, 0.0156607f, -0.124637f,
                                                       0.0302449f, 0.0105196f, -0.0475081f, 0.18416f, -0.102302f,
                                                       0.0363197f, -0.0178498f, 0.0538303f, -0.15488f, 0.0186949f,
                                                       -0.308369f, -0.150942f, 0.0628686f, 0.121276f, -0.043074f,
                                                       0.0784324f, -0.0752792f, 0.0352388f, -0.0203399f, -0.0446295f};

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
  std::array<float, 10> expected_sampled_logits_token_gen{-0.0966602f, 0.0653766f, -0.0240025f, -0.238864f, 0.0626191f,
                                                          0.0217852f, 0.0282981f, 0.0627022f, -0.0670064f, -0.0286431f};

  auto token_gen_logits_ptr = generator->GetOutput("logits");
  auto token_gen_logits = static_cast<float*>(token_gen_logits_ptr->Data());
  int num_token_gen_outputs_to_check = 10;

  for (int i = 0; i < num_token_gen_outputs_to_check; i++) {
    EXPECT_NEAR(expected_sampled_logits_token_gen[i], token_gen_logits[i * sample_size], tolerance);
  }
}

TEST(CAPITests, GetLogitsCAPI) {
  std::array<int64_t, 2> input_ids_shape{2, 4};
  std::array<int32_t, 8> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  int batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  auto model = OgaModel::Create(LLAMA_FP32_PATH);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  // check prompt generation, GetLogits() returns last token logits
  // full logits has shape [2, 1, x]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
  std::array<float, 10> expected_sampled_logits_prompt{0.0744539723f, -0.161290422f, 0.193074539f, 0.0139961755f, 0.0509244837f,
                                                       0.0579358675f, 0.0879421160f, -0.0777675137f, 0.0515393987f, 0.0376916751f};

  auto prompt_logits_ptr = generator->GetLogits();
  auto vocab_size = prompt_logits_ptr->Shape().back();
  auto prompt_logits = reinterpret_cast<float*>(prompt_logits_ptr->Data());
  int num_prompt_outputs_to_check = 5;
  int sample_size = 200;
  float tolerance = 0.001f;
  // Verify outputs match expected outputs
  auto expected = expected_sampled_logits_prompt.begin();
  for (int b = 0; b < batch_size; b++)
    for (int i = 0; i < num_prompt_outputs_to_check; i++)
      EXPECT_NEAR(*expected++, prompt_logits[i * sample_size + b * vocab_size], tolerance);

  generator->GenerateNextToken();
  // check for the 1st token generation
  // full logits has shape [2, 1, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
  std::array<float, 10> expected_sampled_logits_token_gen{-0.022472526878118515f, -0.039382763206958771f, 0.12129191309213638f, -0.056800510734319687f, 0.059339739382266998f,
                                                          -0.057457718998193741f, 0.065887778997421265f, -0.11505863070487976f, -0.046969950199127197f, 0.049980655312538147f};

  auto token_gen_logits_ptr = generator->GetLogits();
  auto token_gen_logits = reinterpret_cast<float*>(token_gen_logits_ptr->Data());
  int num_token_gen_outputs_to_check = 5;

  expected = expected_sampled_logits_token_gen.begin();
  for (int b = 0; b < batch_size; b++)
    for (int i = 0; i < num_token_gen_outputs_to_check; i++)
      EXPECT_NEAR(*expected++, token_gen_logits[i * sample_size + b * vocab_size], tolerance);
}

TEST(CAPITests, SetLogitsCAPI) {
  std::array<int64_t, 2> input_ids_shape{2, 4};
  std::array<int32_t, 8> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  int batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  auto model = OgaModel::Create(LLAMA_FP32_PATH);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOption("batch_size", batch_size);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);

  std::array<float, 5> expected_sampled_logits_prompt{0.29694548f, 0.00955007f, 0.0430819f, 0.10063869f, 0.0437237f};
  std::vector<float> dummy_logits(2 * 32000, 0.0f);
  for (int i = 0; i < dummy_logits.size(); i++) {
    dummy_logits[i] = expected_sampled_logits_prompt[i % expected_sampled_logits_prompt.size()];
  }
  std::array<int64_t, 3> dummy_logits_shape{2, 1, 32000};
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
    try {
      while (!generator->IsDone()) {
        generator->GenerateNextToken();
      }
    } catch (const std::exception& e) {
      EXPECT_EQ(generator->IsSessionTerminated(), true);
      std::cout << "Session Terminated: " << e.what() << std::endl;
    }
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
    std::cout << "Waiting for threads completion" << std::endl;
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
    params_->SetSearchOption("batch_size", 3);
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
      for (size_t i = 0; i < 3; i++) {
        auto out_string = tokenizer_->Decode(generator->GetSequenceData(i), generator->GetSequenceCount(i));
        std::cout << "Decoded string:" << out_string << std::endl;
      }
    }
  }

  std::unique_ptr<OgaModel> model_;
  std::unique_ptr<OgaTokenizer> tokenizer_;
  std::unique_ptr<OgaSequences> input_sequences_;
  std::unique_ptr<OgaGeneratorParams> params_;
};

TEST(CAPITests, TopKCAPI) {
  Phi2Test test;

  test.params_->SetSearchOptionBool("do_sample", true);
  test.params_->SetSearchOption("top_k", 50);
  test.params_->SetSearchOption("temperature", 0.6f);

  test.Run();
}

TEST(CAPITests, TopPCAPI) {
  Phi2Test test;

  test.params_->SetSearchOptionBool("do_sample", true);
  test.params_->SetSearchOption("top_p", 0.6f);
  test.params_->SetSearchOption("temperature", 0.6f);

  test.Run();
}

TEST(CAPITests, TopKTopPCAPI) {
  Phi2Test test;

  test.params_->SetSearchOptionBool("do_sample", true);
  test.params_->SetSearchOption("top_k", 50);
  test.params_->SetSearchOption("top_p", 0.6f);
  test.params_->SetSearchOption("temperature", 0.6f);

  test.Run();
}

#endif  // TEST_PHI2 && !USE_DML

#if TEST_PHI2
TEST(CAPITests, AdaptersTest) {
#ifdef USE_CUDA
  using OutputType = Ort::Float16_t;
#elif defined(USE_DML)
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
#endif

TEST(CAPITests, AdaptersTestMultipleAdapters) {
#if TEST_PHI2
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
#endif
}

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

TEST(CAPITests, BatchedRewindLlamaFp32CAPI) {
  std::array<int64_t, 2> input_ids_shape{2, 4};
  std::array<int32_t, 8> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::array<int32_t, 20> expected_output{
      0, 0, 0, 52, 12102, 30463, 4666, 17192, 3266, 18061,
      0, 0, 195, 731, 29592, 4877, 18112, 22607, 12936, 997};

  auto batch_size = static_cast<int>(input_ids_shape[0]);
  int max_length = 10;

  auto model = OgaModel::Create(LLAMA_FP32_PATH);
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
    const auto sequence_length = generator->GetSequenceCount(i);
    const auto* sequence_data = generator->GetSequenceData(i);

    ASSERT_LE(sequence_length, max_length);

    const auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
  }

  // Rewind to length 0 and verify same output
  generator->RewindTo(0);

  generator->AppendTokens(input_ids);
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

TEST(CAPITests, RewindLlamaFp32CAPI) {
  std::array<int64_t, 2> input_ids_shape{1, 4};
  std::array<int32_t, 4> input_ids{0, 0, 195, 731};

  std::array<int32_t, 10> expected_output{
      0, 0, 195, 731, 29592, 4877, 18112, 22607, 12936, 997};

  int max_length = 10;

  auto model = OgaModel::Create(LLAMA_FP32_PATH);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids);
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

  generator->AppendTokens(std::array<int32_t, 1>{731});
  generator->AppendTokens(std::array<int32_t, 1>{29592});
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

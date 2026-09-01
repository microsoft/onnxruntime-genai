// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>  // for memcmp
#include <filesystem>
#include <fstream>
#include <numeric>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <regex>
#include <algorithm>
#include "span.h"
#include <list>

#define OGA_USE_SPAN 1
#include "models/onnxruntime_api.h"
#include "ort_genai.h"

#include <gtest/gtest.h>

#include "test_utils.h"

namespace {

struct EngineEventSnapshot {
  OgaEngineEventFlags flags{};
  const OgaRequest* request{};
  uint64_t turn_id{};
  int32_t token{};
  OgaFinishReason finish_reason{};
  OgaErrorCode error_code{};
  uint64_t prompt_tokens{};
  uint64_t generated_tokens{};
  uint64_t cached_prompt_tokens{};
};

EngineEventSnapshot Snapshot(const OgaEngineEvent* event) {
  if (!event) {
    return {};
  }
  const auto& usage = event->Usage();
  const auto request = event->Request();
  return {
      event->Flags(),
      request ? &request->get() : nullptr,
      event->TurnId(),
      event->Token(),
      event->FinishReason(),
      event->ErrorCode(),
      usage.PromptTokens(),
      usage.GeneratedTokens(),
      usage.CachedPromptTokens()};
}

EngineEventSnapshot RunOne(OgaEngine& engine) {
  auto buffer = engine.CreateEventBuffer(1);
  engine.Run(*buffer);
  return Snapshot(buffer->Get(0));
}

}  // namespace

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
  config->SetDecoderProviderOptionsHardwareDeviceType("OpenVINO", "npu");
  config->ClearDecoderProviderOptionsHardwareDeviceType("OpenVINO");
  config->SetDecoderProviderOptionsHardwareDeviceId("OpenVINO", 1);
  config->ClearDecoderProviderOptionsHardwareDeviceId("OpenVINO");
  config->SetDecoderProviderOptionsHardwareVendorId("OpenVINO", 2);
  config->ClearDecoderProviderOptionsHardwareVendorId("OpenVINO");
  config->SetDecoderProviderOptionsHardwareDeviceType("OpenVINO", "cpu");
  config->SetDecoderProviderOptionsHardwareDeviceType("DML", "gpu");
  config->SetDecoderProviderOptionsHardwareDeviceId("DML", 2);
  config->SetDecoderProviderOptionsHardwareVendorId("DML", 1);
#endif
}

// Regression test: appending CPU provider should not throw.
// See https://github.com/microsoft/onnxruntime-genai/pull/2179
TEST(CAPITests, AppendCpuProvider) {
#if TEST_PHI2
  auto config = OgaConfig::Create(PHI2_PATH);
  config->ClearProviders();
  config->AppendProvider("cpu");
  auto model = OgaModel::Create(*config);
  ASSERT_NE(model.get(), nullptr);

  // Also test other case variants
  auto config2 = OgaConfig::Create(PHI2_PATH);
  config2->ClearProviders();
  config2->AppendProvider("CPU");
  auto model2 = OgaModel::Create(*config2);
  ASSERT_NE(model2.get(), nullptr);

  auto config3 = OgaConfig::Create(PHI2_PATH);
  config3->ClearProviders();
  config3->AppendProvider("CPUExecutionProvider");
  auto model3 = OgaModel::Create(*config3);
  ASSERT_NE(model3.get(), nullptr);
#endif
}

TEST(CAPITests, TokenizerCAPI) {
#if TEST_PHI2
  auto config = OgaConfig::Create(PHI2_PATH);
  auto model = OgaModel::Create(*config);
  auto tokenizer = OgaTokenizer::Create(*model);

  auto eos_token_ids = tokenizer->GetEosTokenIds();
  ASSERT_EQ(tokenizer->GetBosTokenId(), 50256);
  ASSERT_EQ(tokenizer->GetPadTokenId(), 50256);
  ASSERT_EQ(eos_token_ids.size(), 1);
  ASSERT_EQ(eos_token_ids[0], 50256);

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
    auto stream = OgaTokenizerStream::Create(*tokenizer);

    auto* sequence = sequences->SequenceData(i);
    std::string stream_result;
    for (size_t j = 0; j < sequences->SequenceCount(i); j++) {
      stream_result += stream->Decode(sequence[j]);
    }
    std::cout << "Stream decoded string:" << stream_result << std::endl;
    if (strcmp(input_strings[i], stream_result.c_str()) != 0)
      throw std::runtime_error("Stream token decoding mismatch");
  }
#endif
}

TEST(CAPITests, TokenizerCreateFromConfigAndPath) {
#if TEST_PHI2
  const char* input_string = "She sells sea shells by the sea shore.";

  auto config = OgaConfig::Create(PHI2_PATH);
  auto tokenizer_from_config = OgaTokenizer::Create(*config);
  auto tokenizer_from_path = OgaTokenizer::Create(PHI2_PATH);

  ASSERT_EQ(tokenizer_from_config->GetBosTokenId(), 50256);
  ASSERT_EQ(tokenizer_from_path->GetBosTokenId(), 50256);

  auto input_sequences = OgaSequences::Create();
  tokenizer_from_config->Encode(input_string, *input_sequences);

  auto out_string = tokenizer_from_path->Decode(input_sequences->SequenceData(0), input_sequences->SequenceCount(0));
  ASSERT_STREQ(input_string, out_string);
#endif
}

TEST(CAPITests, EncodeBatchEmptyInputThrows) {
#if TEST_PHI2
  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);

  // EncodeBatch with zero strings should throw, not crash with SIGFPE
  ASSERT_THROW(tokenizer->EncodeBatch(nullptr, 0), std::runtime_error);

  // Invalid pointers with count > 0 should also be rejected deterministically.
  ASSERT_THROW(tokenizer->EncodeBatch(nullptr, 1), std::runtime_error);
  const char* bad_strings[] = {nullptr};
  ASSERT_THROW(tokenizer->EncodeBatch(bad_strings, 1), std::runtime_error);
#endif
}

TEST(CAPITests, TokenizerUpdateOptions) {
#if TEST_PHI2
  auto config = OgaConfig::Create(PHI2_PATH);
  auto model = OgaModel::Create(*config);
  auto tokenizer = OgaTokenizer::Create(*model);

  // Update tokenizer options
  // Note: This simply tests the UpdateOptions API; these options are already set as default.
  {
    const char* keys[] = {"add_special_tokens", "skip_special_tokens"};
    const char* values[] = {"false", "true"};
    tokenizer->UpdateOptions(keys, values, 2);
  }

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
    auto stream = OgaTokenizerStream::Create(*tokenizer);

    auto* sequence = sequences->SequenceData(i);
    std::string stream_result;
    for (size_t j = 0; j < sequences->SequenceCount(i); j++) {
      stream_result += stream->Decode(sequence[j]);
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

  // Testing phi-4-mini chat template
  const char* messages_json = R"(
    [
      {
        "role": "system",
        "content": "System message",
        "tools": "[{\"name\": \"calculate_sum\", \"description\": \"Calculate the sum of two numbers.\", \"parameters\": {\"a\": {\"type\": \"int\"}, \"b\": {\"type\": \"int\"}}}]"
      },
      {
        "role": "user",
        "content": "Hello, can you call some tools for me?"
      },
      {
        "role": "assistant",
        "content": "Sure, I can calculate the sum for you!"
      }
    ])";

  const char* chat_template = R"({% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %})";

  // From HuggingFace Python output for 'microsoft/Phi-4-mini-instruct'
  const char* expected_output =
      "<|system|>System message<|tool|>[{\"name\": \"calculate_sum\", \"description\": \"Calculate the sum of two numbers.\", \"parameters\": {\"a\": {\"type\": \"int\"}, \"b\": {\"type\": \"int\"}}}]<|/tool|><|end|><|user|>"
      "Hello, can you call some tools for me?<|end|><|assistant|>Sure, I can calculate the sum for you!<|end|><|assistant|>";

  auto out_string = tokenizer->ApplyChatTemplate(chat_template, messages_json, nullptr, true);
  ASSERT_STREQ(expected_output, out_string);

  const char* kwargs_template =
      "{% if enable_thinking is defined and not enable_thinking %}NO_THINK{% else %}THINK{% endif %}"
      "|{{ reasoning_effort }}|{{ level }}";
  const char* template_kwargs =
      R"({"enable_thinking":false,"reasoning_effort":"low","level":2})";
  const char* option_keys[] = {"chat_template_kwargs"};
  const char* option_values[] = {template_kwargs};
  tokenizer->UpdateOptions(option_keys, option_values, 1);
  auto kwargs_output = tokenizer->ApplyChatTemplate(
      kwargs_template, messages_json, nullptr, true);
  ASSERT_STREQ("NO_THINK|low|2", kwargs_output);

  option_values[0] = "{}";
  tokenizer->UpdateOptions(option_keys, option_values, 1);
  auto cleared_output = tokenizer->ApplyChatTemplate(
      "{% if enable_thinking is defined %}SET{% else %}CLEARED{% endif %}",
      messages_json, nullptr, true);
  ASSERT_STREQ("CLEARED", cleared_output);

  option_values[0] = template_kwargs;
  tokenizer->UpdateOptions(option_keys, option_values, 1);
  auto reapplied_output = tokenizer->ApplyChatTemplate(
      kwargs_template, messages_json, nullptr, true);
  ASSERT_STREQ("NO_THINK|low|2", reapplied_output);

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

TEST(CAPITests, SequencesOutOfBoundsAccess) {
  auto sequences = OgaSequences::Create();

  std::vector<int32_t> tokens{100, 200, 300};
  sequences->Append(tokens.data(), tokens.size());

  ASSERT_EQ(sequences->Count(), 1u);
  EXPECT_EQ(sequences->SequenceCount(0), tokens.size());
  EXPECT_NE(sequences->SequenceData(0), nullptr);

  // Indices outside the stored range return empty results.
  EXPECT_EQ(sequences->SequenceCount(1), 0u);
  EXPECT_EQ(sequences->SequenceData(1), nullptr);
  EXPECT_EQ(sequences->SequenceCount(1000), 0u);
  EXPECT_EQ(sequences->SequenceData(1000), nullptr);
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

// DML doesn't support batch_size > 1
// TODO: WebGPU should support batch_size > 1, investigate why it's failing
TEST(CAPITests, EndToEndPhiBatch) {
#if TEST_PHI2
  if (!test_utils::IsEngineTestsEnabled()) {
    GTEST_SKIP() << "Skipping batch test for DML/WebGPU";
  }
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

// Every ORT tensor is batch*beam wide: Run receives a [batch_size, sequence]
// prompt buffer and DefaultInputIDs expands it over beams. These graphs declare
// fixed dimensions so a mis-sized tensor fails to bind.
TEST(CAPITests, MarianBatchIOContract) {
  auto model = OgaModel::Create(MODEL_PATH "marian-batch");

  // One token per row; MarianState::Run appends eos, giving a width of 2.
  const std::array<int32_t, 1> first{5};
  const std::array<int32_t, 1> second{7};
  auto sequences = OgaSequences::Create();
  sequences->Append(first.data(), first.size());
  sequences->Append(second.data(), second.size());

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("batch_size", 2);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*sequences);
  generator->GenerateNextToken();

  EXPECT_EQ(generator->GetSequenceCount(0), generator->GetSequenceCount(1));
}

// The decoder turns each row's attention-mask sum into its selected token.
// Unequal prompt lengths therefore validate both per-row EOS insertion and
// the actual mask values rather than only tensor dimensions.
TEST(CAPITests, MarianBatchAttentionMaskValues) {
  auto model = OgaModel::Create(MODEL_PATH "marian-batch-values");

  const std::array<int32_t, 1> first{5};
  const std::array<int32_t, 2> second{7, 8};
  auto sequences = OgaSequences::Create();
  sequences->Append(first.data(), first.size());
  sequences->Append(second.data(), second.size());

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("batch_size", 2);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*sequences);
  const auto generated_start = generator->TokenCount();
  generator->GenerateNextToken();

  ASSERT_EQ(generated_start, 2U);
  ASSERT_GT(generator->GetSequenceCount(0), generated_start);
  ASSERT_GT(generator->GetSequenceCount(1), generated_start);
  EXPECT_EQ(generator->GetSequenceData(0)[generated_start], 2);
  EXPECT_EQ(generator->GetSequenceData(1)[generated_start], 3);
}

// Beam search makes every graph tensor batch*beam wide. The prompt length is
// load-bearing: at two tokens the correct sequence width is 3 and sizing the
// prompt buffer by batch*beam gives 4, so the fixture rejects the regression.
TEST(CAPITests, MarianBatchWithBeamsIOContract) {
  auto model = OgaModel::Create(MODEL_PATH "marian-batch-beams");

  const std::array<int32_t, 2> first{5, 6};
  const std::array<int32_t, 2> second{7, 8};
  auto sequences = OgaSequences::Create();
  sequences->Append(first.data(), first.size());
  sequences->Append(second.data(), second.size());

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("batch_size", 2);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*sequences);
  generator->GenerateNextToken();
  ASSERT_FALSE(generator->IsDone());
  auto logits = generator->GetLogits();
  EXPECT_EQ(logits->Shape(), (std::vector<int64_t>{4, 1, 32001}));
  generator->GenerateNextToken();

  auto next_tokens = generator->GetNextTokens();
  constexpr std::array<int32_t, 4> expected_tokens{2, 2, 4, 4};
  ASSERT_EQ(next_tokens.size(), expected_tokens.size());
  for (size_t beam = 0; beam < expected_tokens.size(); ++beam)
    EXPECT_EQ(next_tokens[beam], expected_tokens[beam]) << "beam " << beam;
}

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

  ASSERT_EQ(static_cast<int>(params->GetSearchNumber("max_length")), 40);
  ASSERT_EQ(params->GetSearchBool("early_stopping"), true);
  ASSERT_EQ(static_cast<int>(generator->TokenCount()), static_cast<int>(generator->GetSequenceCount(0)));

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
  ASSERT_EQ(static_cast<int>(generator->TokenCount()), static_cast<int>(generator->GetSequenceCount(0)));

  const auto* expected_output_start = &expected_output[0];
  EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
#endif
}

TEST(CAPITests, LoadModelFromMemory) {
#if TEST_PHI2

  std::string model_path = std::string(PHI2_PATH) + "/model.onnx";
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

  auto GenerateOutput = [](OgaGenerator* generator, std::unique_ptr<OgaTokenizerStream> stream) {
    EXPECT_THROW({
      while (!generator->IsDone()) {
        generator->GenerateNextToken();
      } }, std::runtime_error);
  };

  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto stream = OgaTokenizerStream::Create(*tokenizer);

  const char* input_string = "She sells sea shells by the sea shore.";
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode(input_string, *input_sequences);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 40);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokenSequences(*input_sequences);
  EXPECT_EQ(generator->IsSessionTerminated(), false);
  std::vector<std::thread> threads;
  threads.push_back(std::thread(GenerateOutput, generator.get(), std::move(stream)));
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

TEST(CAPITests, EngineRequestTurnAndEventContracts) {
  auto model = OgaModel::Create(MODEL_PATH "engine/synthetic-paged");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);
  auto engine = OgaEngine::Create(*model);
  auto request = engine->CreateRequest(*params);
  const std::array<int32_t, 3> input_tokens{2, 3, 4};

  auto turn_options = request->CreateTurnOptions();
  turn_options->SetMaxGeneratedTokens(1);
  EXPECT_EQ(request->BeginTurn(input_tokens, turn_options.get()), 1u);
  const auto event = RunOne(*engine);
  EXPECT_EQ(event.request, request.get());
  EXPECT_EQ(event.turn_id, 1u);
  EXPECT_EQ(event.flags,
            OgaEngineEventFlag_Token | OgaEngineEventFlag_TurnFinished);
  EXPECT_EQ(event.finish_reason, OgaFinishReason_MaxGeneratedTokens);
  EXPECT_EQ(event.prompt_tokens, input_tokens.size());
  EXPECT_EQ(event.generated_tokens, 1u);
  EXPECT_EQ(event.cached_prompt_tokens, 0u);

  const std::array<int32_t, 1> continuation{5};
  const auto second_turn = request->BeginTurn(continuation);
  EXPECT_EQ(second_turn, 2u);
  EXPECT_TRUE(request->CancelTurn(second_turn));
  EXPECT_FALSE(request->CancelTurn(second_turn));
  const auto cancelled = RunOne(*engine);
  EXPECT_EQ(cancelled.request, request.get());
  EXPECT_EQ(cancelled.turn_id, second_turn);
  EXPECT_EQ(cancelled.flags, OgaEngineEventFlag_TurnFinished);
  EXPECT_EQ(cancelled.finish_reason, OgaFinishReason_Cancelled);

  auto request_options = OgaRequestOptions::Create();
  request_options->SetMaxSessionTokens(8);
  auto options_request = engine->CreateRequest(*params, request_options.get());
  options_request->Close();

  auto excessive_request_options = OgaRequestOptions::Create();
  excessive_request_options->SetMaxSessionTokens(17);
  try {
    static_cast<void>(
        engine->CreateRequest(*params, excessive_request_options.get()));
    FAIL() << "Expected max_session_tokens above max_length to fail.";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string(error.what()).find("max_total_tokens (17)"),
        std::string::npos);
    EXPECT_NE(
        std::string(error.what()).find("max_length (16)"),
        std::string::npos);
  }

  auto owner_thread_request = engine->CreateRequest(*params);
  auto owner_thread_options = owner_thread_request->CreateTurnOptions();
  OgaResult* create_options_result{};
  OgaResult* set_options_result{};
  std::thread off_owner_thread([&] {
    OgaTurnOptions* unused_options{};
    create_options_result =
        OgaRequestCreateTurnOptions(owner_thread_request.get(), &unused_options);
    set_options_result =
        OgaTurnOptionsSetMaxGeneratedTokens(owner_thread_options.get(), 1);
  });
  off_owner_thread.join();
  std::unique_ptr<OgaResult> owned_create_options_result{create_options_result};
  std::unique_ptr<OgaResult> owned_set_options_result{set_options_result};
  ASSERT_NE(owned_create_options_result, nullptr);
  ASSERT_NE(owned_set_options_result, nullptr);
  EXPECT_NE(std::string(owned_create_options_result->GetError()).find("owner thread"),
            std::string::npos);
  EXPECT_NE(std::string(owned_set_options_result->GetError()).find("owner thread"),
            std::string::npos);
  owner_thread_request->Close();

  auto validation_request = engine->CreateRequest(*params);
  auto validation_options = validation_request->CreateTurnOptions();

  std::unique_ptr<OgaResult> null_stop_token_ids_result{
      OgaTurnOptionsSetStopTokenIds(validation_options.get(), nullptr)};
  ASSERT_NE(null_stop_token_ids_result, nullptr);
  EXPECT_NE(
      std::string(null_stop_token_ids_result->GetError()).find("stop_token_ids must not be null"),
      std::string::npos);
  std::unique_ptr<OgaResult> null_stop_strings_result{
      OgaTurnOptionsSetStopStrings(validation_options.get(), nullptr)};
  ASSERT_NE(null_stop_strings_result, nullptr);
  EXPECT_NE(
      std::string(null_stop_strings_result->GetError()).find("stop_strings must not be null"),
      std::string::npos);
  validation_request->Close();

  OgaRequest* abandoned_created{};
  OgaCheckResult(OgaEngineCreateRequest(
      engine.get(), params.get(), nullptr, &abandoned_created));
  ASSERT_NE(abandoned_created, nullptr);
  OgaDestroyRequest(abandoned_created);
  EXPECT_FALSE(engine->HasPendingRequests());

  OgaRequest* abandoned_queued{};
  OgaCheckResult(OgaEngineCreateRequest(
      engine.get(), params.get(), nullptr, &abandoned_queued));
  ASSERT_NE(abandoned_queued, nullptr);
  uint64_t abandoned_turn_id{};
  OgaCheckResult(OgaRequestBeginTurn(
      abandoned_queued, nullptr, input_tokens.data(),
      input_tokens.size(), &abandoned_turn_id));
  EXPECT_EQ(abandoned_turn_id, 1u);
  OgaDestroyRequest(abandoned_queued);
  EXPECT_FALSE(engine->HasPendingRequests());

  auto idle_buffer = engine->CreateEventBuffer(1);
  EXPECT_EQ(engine->Run(*idle_buffer), 0u);
  EXPECT_EQ(idle_buffer->Count(), 0u);
  EXPECT_EQ(idle_buffer->Get(0), nullptr);
}

TEST(CAPITests, EngineBulkRunAndReusableStorage) {
  auto model = OgaModel::Create(MODEL_PATH "engine/synthetic-paged");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);
  auto engine = OgaEngine::Create(*model);
  const std::array<int32_t, 3> input_tokens{2, 3, 4};

  const auto create_one_token_request = [&] {
    auto request = engine->CreateRequest(*params);
    auto turn_options = request->CreateTurnOptions();
    turn_options->SetMaxGeneratedTokens(1);
    request->BeginTurn(input_tokens, turn_options.get());
    return request;
  };

  auto first = create_one_token_request();
  auto second = create_one_token_request();

  std::unique_ptr<OgaResult> null_out_result{
      OgaCreateEngineEventBuffer(engine.get(), 1, nullptr)};
  ASSERT_NE(null_out_result, nullptr);
  EXPECT_NE(
      std::string(null_out_result->GetError()).find("out must not be null"),
      std::string::npos);

  OgaEngineEventBuffer* unused_buffer{};
  std::unique_ptr<OgaResult> null_engine_result{
      OgaCreateEngineEventBuffer(nullptr, 1, &unused_buffer)};
  ASSERT_NE(null_engine_result, nullptr);
  EXPECT_EQ(unused_buffer, nullptr);
  EXPECT_NE(
      std::string(null_engine_result->GetError()).find("engine must not be null"),
      std::string::npos);

  OgaEngineEventBuffer* off_thread_buffer{};
  OgaResult* off_thread_create_result{};
  std::thread create_off_owner([&] {
    off_thread_create_result =
        OgaCreateEngineEventBuffer(engine.get(), 1, &off_thread_buffer);
  });
  create_off_owner.join();
  std::unique_ptr<OgaResult> owned_off_thread_create_result{
      off_thread_create_result};
  ASSERT_NE(owned_off_thread_create_result, nullptr);
  EXPECT_EQ(off_thread_buffer, nullptr);
  EXPECT_NE(
      std::string(owned_off_thread_create_result->GetError()).find("owner thread"),
      std::string::npos);

  auto zero_capacity_buffer = engine->CreateEventBuffer(0);
  EXPECT_EQ(engine->Run(*zero_capacity_buffer), 0u);
  EXPECT_TRUE(engine->HasPendingRequests());

  std::unique_ptr<OgaResult> null_buffer_result{
      OgaEngineRun(engine.get(), nullptr)};
  ASSERT_NE(null_buffer_result, nullptr);
  EXPECT_NE(
      std::string(null_buffer_result->GetError()).find("buffer must not be null"),
      std::string::npos);

  auto buffer = engine->CreateEventBuffer(1);
  OgaResult* off_thread_run_result{};
  std::thread run_off_owner([&] {
    off_thread_run_result = OgaEngineRun(engine.get(), buffer.get());
  });
  run_off_owner.join();
  std::unique_ptr<OgaResult> owned_off_thread_run_result{
      off_thread_run_result};
  ASSERT_NE(owned_off_thread_run_result, nullptr);
  EXPECT_NE(
      std::string(owned_off_thread_run_result->GetError()).find("owner thread"),
      std::string::npos);
  EXPECT_EQ(buffer->Count(), 0u);
  EXPECT_TRUE(engine->HasPendingRequests());

  std::unique_ptr<OgaResult> null_run_engine_result{
      OgaEngineRun(nullptr, buffer.get())};
  ASSERT_NE(null_run_engine_result, nullptr);
  EXPECT_NE(
      std::string(null_run_engine_result->GetError()).find("engine must not be null"),
      std::string::npos);
  EXPECT_TRUE(engine->HasPendingRequests());
  EXPECT_EQ(buffer->Count(), 0u);

  auto other_engine = OgaEngine::Create(*model);
  std::unique_ptr<OgaResult> wrong_engine_result{
      OgaEngineRun(other_engine.get(), buffer.get())};
  ASSERT_NE(wrong_engine_result, nullptr);
  EXPECT_NE(
      std::string(wrong_engine_result->GetError()).find("Engine that created it"),
      std::string::npos);
  EXPECT_EQ(buffer->Count(), 0u);

  EXPECT_TRUE(first->CancelTurn(1));
  EXPECT_TRUE(second->CancelTurn(1));

  OgaCheckResult(OgaEngineRun(engine.get(), buffer.get()));
  ASSERT_EQ(OgaEngineEventBufferGetCount(buffer.get()), 1u);
  EXPECT_EQ(OgaEngineEventBufferGetCount(nullptr), 0u);
  EXPECT_EQ(OgaEngineEventBufferGet(nullptr, 0), nullptr);
  EXPECT_EQ(OgaEngineEventBufferGet(buffer.get(), 1), nullptr);

  const OgaEngineEvent* first_event =
      OgaEngineEventBufferGet(buffer.get(), 0);
  ASSERT_NE(first_event, nullptr);

  const OgaRequest* borrowed_request{};
  OgaEngineEventFlags flags{};
  uint64_t turn_id{};
  int32_t token{};
  OgaFinishReason finish_reason{};
  OgaErrorCode error_code{};
  const OgaTurnUsage* usage{};
  OgaCheckResult(OgaEngineEventGetRequest(first_event, &borrowed_request));
  OgaCheckResult(OgaEngineEventGetFlags(first_event, &flags));
  OgaCheckResult(OgaEngineEventGetTurnId(first_event, &turn_id));
  OgaCheckResult(OgaEngineEventGetToken(first_event, &token));
  OgaCheckResult(OgaEngineEventGetFinishReason(first_event, &finish_reason));
  OgaCheckResult(OgaEngineEventGetErrorCode(first_event, &error_code));
  OgaCheckResult(OgaEngineEventGetUsage(first_event, &usage));
  EXPECT_EQ(borrowed_request, first.get());
  EXPECT_EQ(flags, OgaEngineEventFlag_TurnFinished);
  EXPECT_EQ(turn_id, 1u);
  EXPECT_EQ(finish_reason, OgaFinishReason_Cancelled);
  EXPECT_EQ(error_code, OgaErrorCode_None);
  ASSERT_NE(usage, nullptr);

  uint64_t prompt_tokens{};
  uint64_t generated_tokens{};
  uint64_t cached_prompt_tokens{};
  OgaCheckResult(OgaTurnUsageGetPromptTokens(usage, &prompt_tokens));
  OgaCheckResult(OgaTurnUsageGetGeneratedTokens(usage, &generated_tokens));
  OgaCheckResult(
      OgaTurnUsageGetCachedPromptTokens(usage, &cached_prompt_tokens));
  EXPECT_EQ(prompt_tokens, input_tokens.size());
  EXPECT_EQ(generated_tokens, 0u);
  EXPECT_EQ(cached_prompt_tokens, 0u);

  std::unique_ptr<OgaResult> null_event_result{
      OgaEngineEventGetFlags(nullptr, &flags)};
  ASSERT_NE(null_event_result, nullptr);
  EXPECT_EQ(flags, OgaEngineEventFlag_None);
  std::unique_ptr<OgaResult> null_event_out_result{
      OgaEngineEventGetFlags(first_event, nullptr)};
  ASSERT_NE(null_event_out_result, nullptr);
  std::unique_ptr<OgaResult> null_usage_result{
      OgaTurnUsageGetPromptTokens(nullptr, &prompt_tokens)};
  ASSERT_NE(null_usage_result, nullptr);
  EXPECT_EQ(prompt_tokens, 0u);

  std::unique_ptr<OgaResult> populated_wrong_engine_result{
      OgaEngineRun(other_engine.get(), buffer.get())};
  ASSERT_NE(populated_wrong_engine_result, nullptr);
  EXPECT_EQ(buffer->Count(), 1u);
  EXPECT_EQ(buffer->Get(0), first_event);
  EXPECT_EQ(&buffer->Get(0)->Request()->get(), first.get());

  OgaCheckResult(OgaEngineRun(engine.get(), buffer.get()));
  ASSERT_EQ(OgaEngineEventBufferGetCount(buffer.get()), 1u);
  EXPECT_EQ(&buffer->Get(0)->Request()->get(), second.get());

  first->Close();
  second->Close();

  auto reusable_request = engine->CreateRequest(*params);
  auto reusable_turn_options = reusable_request->CreateTurnOptions();
  reusable_turn_options->SetMaxGeneratedTokens(2);
  reusable_request->BeginTurn(input_tokens, reusable_turn_options.get());
  auto reusable_buffer = engine->CreateEventBuffer(1);

  ASSERT_EQ(engine->Run(*reusable_buffer), 1u);
  const OgaEngineEvent* reusable = reusable_buffer->Get(0);
  ASSERT_NE(reusable, nullptr);
  EXPECT_EQ(&reusable->Request()->get(), reusable_request.get());

  ASSERT_EQ(engine->Run(*reusable_buffer), 1u);
  EXPECT_EQ(&reusable_buffer->Get(0)->Request()->get(), reusable_request.get());
  EXPECT_NE(
      reusable_buffer->Get(0)->Flags() & OgaEngineEventFlag_TurnFinished,
      0u);
  reusable_request->Close();
}

TEST(CAPITests, EngineCppRunReturnsBorrowedBufferViews) {
  auto model = OgaModel::Create(MODEL_PATH "engine/synthetic-paged");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);
  auto engine = OgaEngine::Create(*model);
  const std::array<int32_t, 3> input_tokens{2, 3, 4};

  const auto create_request = [&] {
    auto request = engine->CreateRequest(*params);
    auto turn_options = request->CreateTurnOptions();
    turn_options->SetMaxGeneratedTokens(1);
    request->BeginTurn(input_tokens, turn_options.get());
    return request;
  };
  auto first = create_request();
  auto second = create_request();

  auto buffer = engine->CreateEventBuffer(4);
  ASSERT_EQ(engine->Run(*buffer), 2u);
  ASSERT_EQ(buffer->Count(), 2u);
  ASSERT_NE(buffer->Get(0), nullptr);
  ASSERT_NE(buffer->Get(1), nullptr);
  EXPECT_EQ(&buffer->Get(0)->Request()->get(), first.get());
  EXPECT_EQ(&buffer->Get(1)->Request()->get(), second.get());
  EXPECT_EQ(buffer->Get(2), nullptr);

  first->Close();
  second->Close();
}

TEST(CAPITests, EngineRetainsModelAfterPublicHandleRelease) {
  OgaModel* model{};
  ASSERT_EQ(
      OgaCreateModel(MODEL_PATH "engine/synthetic-paged", &model),
      nullptr);
  ASSERT_NE(model, nullptr);

  OgaEngine* engine{};
  ASSERT_EQ(OgaCreateEngine(model, &engine), nullptr);
  ASSERT_NE(engine, nullptr);

  OgaDestroyModel(model);
  model = nullptr;

  bool has_pending_requests = true;
  EXPECT_EQ(
      OgaEngineHasPendingRequests(engine, &has_pending_requests),
      nullptr);
  EXPECT_FALSE(has_pending_requests);

  OgaEngineEventBuffer* buffer{};
  EXPECT_EQ(
      OgaCreateEngineEventBuffer(engine, 1, &buffer),
      nullptr);
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(OgaEngineRun(engine, buffer), nullptr);
  EXPECT_EQ(OgaEngineEventBufferGetCount(buffer), 0u);

  OgaDestroyEngine(engine);
  EXPECT_EQ(OgaEngineEventBufferGetCount(buffer), 0u);
  OgaDestroyEngineEventBuffer(buffer);
}

// DML doesn't support batch_size > 1
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

    struct OwnedRequest {
      std::unique_ptr<OgaRequest> request;
      std::vector<int32_t> generated_tokens;
    };
    std::vector<OwnedRequest> requests;
    requests.reserve(batch_size_);
    std::unordered_map<const OgaRequest*, OwnedRequest*> requests_by_handle;

    const char* input_strings[] = {
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    };

    for (size_t i = 0; i < batch_size_; i++) {
      auto input_sequence = OgaSequences::Create();
      tokenizer_->Encode(input_strings[i], *input_sequence);
      auto input_tokens = std::span<const int32_t>{
          input_sequence->SequenceData(0), input_sequence->SequenceCount(0)};
      requests.push_back({engine->CreateRequest(*params_),
                          std::vector<int32_t>(input_tokens.begin(), input_tokens.end())});
      auto& owned_request = requests.back();
      requests_by_handle.emplace(owned_request.request.get(), &owned_request);
      owned_request.request->BeginTurn(input_tokens);
    }

    EXPECT_TRUE(engine->HasPendingRequests());
    while (engine->HasPendingRequests()) {
      auto event = RunOne(*engine);
      auto* ready_request = event.request;
      ASSERT_NE(ready_request, nullptr);
      auto it = requests_by_handle.find(ready_request);
      ASSERT_NE(it, requests_by_handle.end());
      EXPECT_EQ(ready_request, it->second->request.get());
      if ((event.flags & OgaEngineEventFlag_Token) != 0) {
        it->second->generated_tokens.push_back(event.token);
      }
    }
    EXPECT_EQ(RunOne(*engine).flags, OgaEngineEventFlag_None);

    for (auto& owned_request : requests) {
      EXPECT_NO_THROW(owned_request.request->Close());
      EXPECT_NO_THROW(owned_request.request->Close());

      auto out_string = tokenizer_->Decode(owned_request.generated_tokens.data(),
                                           owned_request.generated_tokens.size());
      std::cout << "Decoded string:" << out_string << std::endl;
    }
  }

  std::unique_ptr<OgaModel> model_;
  std::unique_ptr<OgaTokenizer> tokenizer_;
  std::unique_ptr<OgaSequences> input_sequences_;
  std::unique_ptr<OgaGeneratorParams> params_;
  const size_t batch_size_ = 3;
};

TEST(CAPITests, EngineRequestCAbiAndRaiiContracts) {
  if (!test_utils::IsEngineTestsEnabled()) {
    GTEST_SKIP() << "Skipping Engine test for DML/WebGPU";
  }

  auto model = OgaModel::Create(PHI2_PATH);
  auto tokenizer = OgaTokenizer::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto engine = OgaEngine::Create(*model);
  auto input_sequences = OgaSequences::Create();
  tokenizer->Encode("This is a test.", *input_sequences);
  auto input_tokens = std::span<const int32_t>{
      input_sequences->SequenceData(0), input_sequences->SequenceCount(0)};
  params->SetSearchOption(
      "max_length", static_cast<int>(input_tokens.size() + 4));

  auto owned_request = engine->CreateRequest(*params);
  auto one_token_turn = owned_request->CreateTurnOptions();
  one_token_turn->SetMaxGeneratedTokens(1);
  EXPECT_EQ(owned_request->BeginTurn(input_tokens, one_token_turn.get()), 1u);
  one_token_turn->SetMaxGeneratedTokens(3);
  ASSERT_TRUE(engine->HasPendingRequests());

  const auto event = RunOne(*engine);
  ASSERT_EQ(event.request, owned_request.get());
  EXPECT_EQ(event.turn_id, 1u);
  EXPECT_NE(event.flags & OgaEngineEventFlag_TurnFinished, 0u);
  EXPECT_EQ(event.finish_reason, OgaFinishReason_MaxGeneratedTokens);

  EXPECT_NO_THROW(owned_request->Close());
  EXPECT_NO_THROW(owned_request->Close());
  EXPECT_FALSE(engine->HasPendingRequests());
  EXPECT_EQ(RunOne(*engine).flags, OgaEngineEventFlag_None);
}

class ParametrizedTopKCAPITestsTests : public ::testing::TestWithParam<bool> {
};

TEST_P(ParametrizedTopKCAPITestsTests, TopKCAPI) {
  if (GetParam() && !test_utils::IsEngineTestsEnabled()) {
    GTEST_SKIP() << "Skipping Engine test for DML/WebGPU";
  }

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
  if (GetParam() && !test_utils::IsEngineTestsEnabled()) {
    GTEST_SKIP() << "Skipping Engine test for DML/WebGPU";
  }

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
  if (GetParam() && !test_utils::IsEngineTestsEnabled()) {
    GTEST_SKIP() << "Skipping Engine test for DML/WebGPU";
  }

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

// A top_k value larger than the model's vocab_size must be rejected during
// generator creation.
TEST(CAPITests, TopKExceedsVocabSizeThrows) {
  Phi2Test test;

  // Chosen to sit well above typical vocabulary sizes (and above Phi-2's known
  // vocab_size) so it reliably exceeds the model's vocab_size.
  constexpr int kTopKAboveVocabSize = 1'000'000;

  test.params_->SetSearchOptionBool("do_sample", true);
  test.params_->SetSearchOption("top_k", kTopKAboveVocabSize);
  test.params_->SetSearchOption("temperature", 0.6f);

  try {
    OgaGenerator::Create(*test.model_, *test.params_);
    FAIL() << "Expected std::runtime_error for top_k > vocab_size";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("vocab_size"), std::string::npos)
        << "Unexpected error message: " << e.what();
  }
}

// Regression test for the combined Top-K + Top-P sampling path.
TEST(CAPITests, TopKTopPExceedsVocabSizeThrows) {
  Phi2Test test;

  constexpr int kTopKAboveVocabSize = 1'000'000;

  test.params_->SetSearchOptionBool("do_sample", true);
  test.params_->SetSearchOption("top_k", kTopKAboveVocabSize);
  test.params_->SetSearchOption("top_p", 0.6f);
  test.params_->SetSearchOption("temperature", 0.6f);

  try {
    OgaGenerator::Create(*test.model_, *test.params_);
    FAIL() << "Expected std::runtime_error for top_k > vocab_size";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("vocab_size"), std::string::npos)
        << "Unexpected error message: " << e.what();
  }
}

// Regression test: a GeneratorParams created from a model must keep that model
// alive, so destroying the model handle before creating the generator does not
// cause a use-after-free (GeneratorParams aliases the model-owned Config, and
// Generator::Generator calls model.shared_from_this()).
TEST(CAPITests, CreateGeneratorAfterDestroyModel) {
  OgaModel* model = nullptr;
  ASSERT_EQ(OgaCreateModel(PHI2_PATH, &model), nullptr);
  ASSERT_NE(model, nullptr);

  OgaGeneratorParams* params = nullptr;
  ASSERT_EQ(OgaCreateGeneratorParams(model, &params), nullptr);
  ASSERT_NE(params, nullptr);

  // Drop the external reference to the model by destroying its handle. Because
  // params co-owns the underlying Model (and its Config) via shared ownership,
  // the object itself stays alive, so dereferencing the raw model pointer below
  // remains valid. This does NOT imply the handle is generally usable after
  // OgaDestroyModel; it is valid here only because another owner keeps it alive.
  OgaDestroyModel(model);

  OgaGenerator* generator = nullptr;
  ASSERT_EQ(OgaCreateGenerator(model, params, &generator), nullptr);
  ASSERT_NE(generator, nullptr);

  OgaDestroyGenerator(generator);
  OgaDestroyGeneratorParams(params);
}

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

// Regression test for the concurrency use-after-free / data race in the
// adapter lifecycle. Prior to serializing Adapters ops with a mutex,
// concurrent LoadAdapter/UnloadAdapter/SetActiveAdapter calls could race on
// Adapter::ref_count_ and on the underlying unordered_map, producing lost
// updates and a TOCTOU window where UnloadAdapter would erase an adapter
// that another thread had just acquired.
//
// This test hammers the Adapters API from multiple threads. It is not
// deterministic about which operations succeed (a concurrent UnloadAdapter
// may legitimately throw "Adapter still in use" or "Adapter not found",
// and a concurrent LoadAdapter of the same name may throw "already loaded")
// but under TSAN/ASAN and in stress mode it reliably catches the pre-fix
// races. Here we simply assert that no thread crashes or leaves the
// Adapters map in an inconsistent state.
TEST(CAPITests, AdaptersConcurrentLoadUnload) {
  auto model = OgaModel::Create(MODEL_PATH "multiple_adapters");
  auto adapters = OgaAdapters::Create(*model);

  constexpr int kIterations = 50;
  constexpr int kThreadsPerRole = 4;

  const char* adapter_path_a = MODEL_PATH "multiple_adapters/adapter_0.onnx_adapter";
  const char* adapter_path_b = MODEL_PATH "multiple_adapters/adapter_1.onnx_adapter";

  auto swallow = [](auto&& fn) {
    try {
      fn();
    } catch (const std::exception&) {
      // Concurrent load/unload can legitimately throw (already loaded /
      // not found / still in use). We only care that state stays consistent.
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kThreadsPerRole * 2);

  for (int t = 0; t < kThreadsPerRole; ++t) {
    threads.emplace_back([&] {
      for (int i = 0; i < kIterations; ++i) {
        swallow([&] { adapters->LoadAdapter(adapter_path_a, "adapter_a"); });
        swallow([&] { adapters->LoadAdapter(adapter_path_b, "adapter_b"); });
      }
    });
    threads.emplace_back([&] {
      for (int i = 0; i < kIterations; ++i) {
        swallow([&] { adapters->UnloadAdapter("adapter_a"); });
        swallow([&] { adapters->UnloadAdapter("adapter_b"); });
      }
    });
  }

  for (auto& th : threads) th.join();

  // Drain any adapters left loaded so we end in a known state. These may
  // throw "not found" depending on which thread won the last unload; that's
  // fine, we just want to prove the API remains usable and consistent.
  swallow([&] { adapters->UnloadAdapter("adapter_a"); });
  swallow([&] { adapters->UnloadAdapter("adapter_b"); });

  // After draining, a fresh load/unload cycle must still succeed cleanly.
  adapters->LoadAdapter(adapter_path_a, "adapter_a");
  adapters->UnloadAdapter("adapter_a");
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

TEST(CAPITests, GreedySearchLfm2Fp32CAPI) {
  std::vector<int64_t> input_ids_shape{1, 4};
  std::vector<int32_t> input_ids{0, 0, 195, 731};

  int max_length = 10;

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-lfm2-fp32");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Verify generation completed and produced output
  const auto sequence_length = generator->GetSequenceCount(0);
  ASSERT_GT(sequence_length, static_cast<size_t>(input_ids_shape[1]));
  ASSERT_LE(sequence_length, max_length);
}

TEST(CAPITests, RewindLfm2Fp32ThrowsCAPI) {
  std::vector<int32_t> input_ids{0, 0, 195, 731};

  int max_length = 10;

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-lfm2-fp32");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());

  // Generate a few tokens
  generator->GenerateNextToken();

  // RewindTo should throw for LFM2 because conv state cannot be rewound
  EXPECT_THROW(generator->RewindTo(0), std::runtime_error);
}
#endif

// Test RewindTo with static mask handling via NvTensorRtRtx past-present share buffer.
// Skipped when the phi3-fp16-nvtrt model is not available (CI-only model).
TEST(CAPITests, RewindGraphCaptureNvTensorRtRtxCAPI) {
  std::string nvtrt_path = MODEL_PATH "hf-internal-testing/phi3-fp16-nvtrt";
  if (!std::filesystem::exists(nvtrt_path)) {
    GTEST_SKIP() << "NvTensorRtRtx model not available at " << nvtrt_path;
  }

  auto config = OgaConfig::Create(nvtrt_path.c_str());
  config->ClearProviders();
  config->AppendProvider("NvTensorRtRtx");

  int max_length = 20;

  auto model = OgaModel::Create(*config);
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);

  std::vector<int32_t> input_ids{1, 15043, 29892, 920};

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto seq_len = generator->GetSequenceCount(0);
  std::vector<int32_t> first_output(seq_len);
  std::memcpy(first_output.data(), generator->GetSequenceData(0), seq_len * sizeof(int32_t));

  generator->RewindTo(0);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto seq_len2 = generator->GetSequenceCount(0);
  ASSERT_EQ(seq_len2, seq_len);
  EXPECT_TRUE(0 == std::memcmp(first_output.data(), generator->GetSequenceData(0), seq_len * sizeof(int32_t)));

  generator->RewindTo(6);
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  seq_len2 = generator->GetSequenceCount(0);
  ASSERT_EQ(seq_len2, seq_len);
  EXPECT_TRUE(0 == std::memcmp(first_output.data(), generator->GetSequenceData(0), seq_len * sizeof(int32_t)));
}

// Test RewindTo with the qwen-2.5 model. Exercises the static mask rewind path if
// the EP supports it (DML by default, WebGPU/CUDA when graph capture is enabled
// in model generation via _test_utils.py), otherwise falls back to the dynamic mask path.
// Skipped when qwen-2.5 model is not available.
//
// CUDA is explicitly disabled: RewindTo(seq_len - 3) — a deep partial rewind near
// the end of a completed sequence — produces incorrect output on CUDA. This is a
// pre-existing runtime bug (not model-specific): it reproduces with both
// qwen-2.5-0.5b-graph and tiny-qwen35-cuda models, and with both static-mask
// (graph-capture) and dynamic-mask (baseline) code paths. Full rewind (RewindTo(0))
// and shallow partial rewind (e.g. RewindTo(input_ids.size()-1)) work correctly.
// TODO: Remove !USE_CUDA once the CUDA partial rewind bug is fixed.
//
// DML is explicitly disabled: The Qwen-2.5 graph capture model seems to have a node
// that is not placed on either the CPU EP or DML EP, which causes a runtime error when
// the model is loaded. This is a pre-existing runtime issue.
// TODO: Remove !USE_DML once the Qwen-2.5 graph capture model is fixed to place all nodes
// on a valid EP.
#if TEST_QWEN_2_5 && !USE_CUDA && !USE_DML
TEST(CAPITests, RewindQwen25CAPI) {
  // Prefer graph-capture variant (exercises static mask rewind on CUDA/WebGPU/DML),
  // fall back to baseline model when it is not available.
  std::string model_path = QWEN_2_5_GRAPH_PATH;
  if (!std::filesystem::exists(model_path)) {
    model_path = QWEN_2_5_PATH;
  }
  if (!std::filesystem::exists(model_path)) {
    GTEST_SKIP() << "qwen-2.5 model not available at " << model_path;
  }

  int max_length = 50;
  std::vector<int32_t> input_ids{1, 2, 3, 4, 5};

  auto model = OgaModel::Create(model_path.c_str());
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetSearchOptionBool("do_sample", false);

  auto generator = OgaGenerator::Create(*model, *params);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  // Save first-run output
  auto seq_len = generator->GetSequenceCount(0);
  std::vector<int32_t> first_output(seq_len);
  std::copy(generator->GetSequenceData(0), generator->GetSequenceData(0) + seq_len, first_output.begin());

  // RewindTo(0) - full rewind
  generator->RewindTo(0);
  generator->AppendTokens(input_ids.data(), input_ids.size());
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
  }

  auto seq_len2 = generator->GetSequenceCount(0);
  ASSERT_EQ(seq_len2, seq_len);
  EXPECT_TRUE(0 == std::memcmp(first_output.data(), generator->GetSequenceData(0), seq_len * sizeof(int32_t)));

  // Partial rewind
  if (seq_len > 7) {
    generator->RewindTo(seq_len - 3);
    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }

    seq_len2 = generator->GetSequenceCount(0);
    ASSERT_EQ(seq_len2, seq_len);
    EXPECT_TRUE(0 == std::memcmp(first_output.data(), generator->GetSequenceData(0), seq_len * sizeof(int32_t)));
  }
}
#endif  // TEST_QWEN_2_5

// Streaming ASR models exercised by the StreamingASR* tests below, each paired
// with its chunk size (samples per Process() call).
struct StreamingASRModel {
  std::string subdir;
  size_t chunk_samples;
};

// Value-parameterized fixture so each StreamingASR test runs once per model.
// This must be a distinct fixture from the non-parameterized CAPITests tests:
// gtest requires all tests in a suite to share the same fixture class, so we
// use a separate suite name here. Instantiated test names look like
// "StreamingASR/StreamingASRParamTests.StreamingASRCreate/nemotron_speech_streaming".
class StreamingASRParamTests : public ::testing::TestWithParam<StreamingASRModel> {
 protected:
  std::string ModelPath() const { return std::string(MODEL_PATH) + GetParam().subdir; }
  size_t ChunkSamples() const { return GetParam().chunk_samples; }
};

// Helper: a streaming ASR model has VAD enabled by default only when its genai_config.json
// declares a "vad" section AND the referenced silero_vad.onnx file is present in the model dir.
static bool ModelHasVad(const std::string& model_path) {
  const auto vad_path = std::filesystem::path(model_path) / "silero_vad.onnx";
  if (!std::filesystem::exists(vad_path))
    return false;
  std::ifstream config_file(std::filesystem::path(model_path) / "genai_config.json");
  if (!config_file)
    return false;
  const std::string config_json((std::istreambuf_iterator<char>(config_file)),
                                std::istreambuf_iterator<char>());
  return std::regex_search(config_json, std::regex(R"("vad"\s*:\s*\{)"));
}

// Helper: if mel is not null, set inputs and run the decode loop
static void DecodeInputs(OgaGenerator& generator, OgaNamedTensors* mel) {
  if (mel) {
    generator.SetInputs(*mel);
    while (!generator.IsDone()) {
      generator.GenerateNextToken();
    }
  }
}

// Helper: generate a synthetic sine wave of num_samples at the given frequency and sample rate.
// Uses frequency and sample_rate to produce the minimum length of speech detectable by the model.
static std::vector<float> GenerateSineWave(size_t num_samples, float frequency, float sample_rate) {
  std::vector<float> samples(num_samples);
  for (size_t i = 0; i < num_samples; ++i)
    samples[i] = 0.5f * std::sin(2.0f * 3.14159265f * frequency * static_cast<float>(i) / sample_rate);
  return samples;
}

#ifndef STREAMING_ASR_PATH
#define STREAMING_ASR_PATH MODEL_PATH "nemotron-speech-streaming"
#endif

#ifndef STREAMING_ASR_CHUNK_SAMPLES
constexpr size_t STREAMING_ASR_CHUNK_SAMPLES = 8960;
#endif

// Test creating a Generator + StreamingProcessor from a streaming ASR model
TEST_P(StreamingASRParamTests, StreamingASRCreate) {
  const std::string model_path = ModelPath();
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path;
  auto model = OgaModel::Create(model_path.c_str());
  auto processor = OgaStreamingProcessor::Create(*model);
  ASSERT_NE(processor, nullptr);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  ASSERT_NE(generator, nullptr);
}

// Test that the public StreamingProcessor API produces the expected named mel tensor.
TEST(CAPITests, StreamingASRProcessReturnsAudioFeaturesTensor) {
  if (!std::filesystem::exists(STREAMING_ASR_PATH))
    GTEST_SKIP() << "Streaming ASR model not found at " << STREAMING_ASR_PATH;
  auto model = OgaModel::Create(STREAMING_ASR_PATH);
  auto processor = OgaStreamingProcessor::Create(*model);

  std::vector<float> silence(STREAMING_ASR_CHUNK_SAMPLES, 0.0f);
  auto inputs = processor->Process(silence.data(), silence.size());
  ASSERT_NE(inputs, nullptr);

  auto audio_features = inputs->Get("audio_features");
  ASSERT_NE(audio_features, nullptr);

  const auto type = audio_features->Type();
  EXPECT_TRUE(type == OgaElementType_float32 || type == OgaElementType_float16);

  const auto shape = audio_features->Shape();
  ASSERT_EQ(shape.size(), 3U);
  EXPECT_EQ(shape[0], 1);
  EXPECT_GT(shape[1], 0);
  EXPECT_GT(shape[2], 0);
}

// Test transcribing silence (all zeros) via GenerateNextToken
TEST_P(StreamingASRParamTests, StreamingASRTranscribeSilence) {
  const std::string model_path = ModelPath();
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path;
  auto model = OgaModel::Create(model_path.c_str());
  auto processor = OgaStreamingProcessor::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);

  const size_t chunk_samples = ChunkSamples();
  std::vector<float> silence(chunk_samples, 0.0f);

  auto mel = processor->Process(silence.data(), silence.size());
  DecodeInputs(*generator, mel.get());
  SUCCEED();
}

// Test feeding multiple chunks and decoding via GenerateNextToken
TEST_P(StreamingASRParamTests, StreamingASRMultipleChunks) {
  const std::string model_path = ModelPath();
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path;
  auto model = OgaModel::Create(model_path.c_str());
  auto processor = OgaStreamingProcessor::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);

  const size_t chunk_samples = ChunkSamples();
  std::vector<float> silence(chunk_samples, 0.0f);

  for (int i = 0; i < 5; ++i) {
    auto mel = processor->Process(silence.data(), silence.size());
    DecodeInputs(*generator, mel.get());
  }
  SUCCEED();
}

// Test flush processes remaining buffered audio
TEST_P(StreamingASRParamTests, StreamingASRFlush) {
  const std::string model_path = ModelPath();
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path;
  auto model = OgaModel::Create(model_path.c_str());
  auto processor = OgaStreamingProcessor::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);

  const size_t chunk_samples = ChunkSamples();
  std::vector<float> silence(chunk_samples, 0.0f);
  processor->Process(silence.data(), silence.size());

  auto mel = processor->Flush();
  DecodeInputs(*generator, mel.get());
  SUCCEED();
}

// Test transcribing a synthetic sine wave via GenerateNextToken
TEST_P(StreamingASRParamTests, StreamingASRSineWave) {
  const std::string model_path = ModelPath();
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path;
  auto model = OgaModel::Create(model_path.c_str());
  auto processor = OgaStreamingProcessor::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);

  // Silero VAD does not recognize a synthetic tone as speech, so with VAD enabled the tone would
  // be dropped as silence. Disable VAD to exercise the audio->tensor->decode pipeline itself
  // (VAD behavior is covered by the dedicated StreamingASRVad* tests).
  processor->SetOption("use_vad", "false");

  const size_t chunk_samples = ChunkSamples();
  constexpr float sample_rate = 16000.0f;
  constexpr float frequency = 440.0f;

  std::vector<float> audio = GenerateSineWave(chunk_samples, frequency, sample_rate);

  for (int i = 0; i < 4; ++i) {
    auto mel = processor->Process(audio.data(), audio.size());
    ASSERT_NE(mel, nullptr);
    DecodeInputs(*generator, mel.get());
  }

  auto flush_mel = processor->Flush();
  DecodeInputs(*generator, flush_mel.get());
  SUCCEED();
}

// Test raw C API for StreamingProcessor + Generator
TEST_P(StreamingASRParamTests, StreamingASRRawCAPI) {
  const std::string model_path = ModelPath();
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path;
  OgaModel* model = nullptr;
  ASSERT_EQ(OgaCreateModel(model_path.c_str(), &model), nullptr);
  ASSERT_NE(model, nullptr);

  OgaStreamingProcessor* processor = nullptr;
  ASSERT_EQ(OgaCreateStreamingProcessor(model, &processor), nullptr);
  ASSERT_NE(processor, nullptr);

  // Disable VAD so the synthetic silence flows through the pipeline instead of being dropped
  // (Silero VAD treats it as non-speech). VAD behavior is covered by the StreamingASRVad* tests.
  ASSERT_EQ(OgaStreamingProcessorSetOption(processor, "use_vad", "false"), nullptr);

  OgaGeneratorParams* params = nullptr;
  ASSERT_EQ(OgaCreateGeneratorParams(model, &params), nullptr);
  OgaGenerator* generator = nullptr;
  ASSERT_EQ(OgaCreateGenerator(model, params, &generator), nullptr);
  ASSERT_NE(generator, nullptr);

  const size_t chunk_samples = ChunkSamples();
  std::vector<float> silence(chunk_samples, 0.0f);

  OgaNamedTensors* inputs = nullptr;
  ASSERT_EQ(OgaStreamingProcessorProcess(processor, silence.data(), silence.size(), &inputs), nullptr);
  ASSERT_NE(inputs, nullptr);
  ASSERT_EQ(OgaGenerator_SetInputs(generator, inputs), nullptr);
  while (!OgaGenerator_IsDone(generator)) {
    ASSERT_EQ(OgaGenerator_GenerateNextToken(generator), nullptr);
  }
  OgaDestroyNamedTensors(inputs);

  OgaDestroyGenerator(generator);
  OgaDestroyGeneratorParams(params);
  OgaDestroyStreamingProcessor(processor);
  OgaDestroyModel(model);
}

// Test VAD set_option/get_option on StreamingProcessor
TEST_P(StreamingASRParamTests, StreamingASRVadSetGetOption) {
  const std::string model_path = ModelPath();
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Streaming ASR model not found at " << model_path;
  auto model = OgaModel::Create(model_path.c_str());
  auto processor = OgaStreamingProcessor::Create(*model);

  // Checking whether or not the model has the components necessary to run VAD. If not, skip the test.
  // VAD requires having a "vad" section in the genai_config.json and the silero_vad.onnx file in the model directory.
  // For VAD-capable models, InitVadFromConfig automatically enables VAD (via EnableVadFromModel())
  // when the config's vad.filename is non-empty, so use_vad starts out "true".
  const bool model_has_vad = ModelHasVad(model_path);

  if (!model_has_vad) {
    GTEST_SKIP() << "Streaming ASR model is not set up for VAD, skipping test";
  }

  // VAD is enabled by default when possible from InitializeVadFromConfig
  ASSERT_EQ(std::string(processor->GetOption("use_vad")), "true");

  // silence_duration_ms round-trips exactly only when it is an integer multiple of the model's
  // chunk duration (chunk_samples / sample_rate * 1000). Use two chunks' worth: 1000 ms for the
  // 8000-sample moonshine models (500 ms/chunk) and 1120 ms for the 8960-sample nemotron model
  // (560 ms/chunk).
  const char* silence_duration_ms = (GetParam().subdir == "nemotron-speech-streaming") ? "1120" : "1000";
  processor->SetOption("silence_duration_ms", silence_duration_ms);
  ASSERT_EQ(std::string(processor->GetOption("silence_duration_ms")), silence_duration_ms);

  // Enable VAD if the model declares a VAD config
  if (model_has_vad) {
    processor->SetOption("use_vad", "true");
    ASSERT_EQ(std::string(processor->GetOption("use_vad")), "true");

    processor->SetOption("vad_threshold", "0.8");
    ASSERT_EQ(std::string(processor->GetOption("use_vad")), "true");

    // Disable
    processor->SetOption("use_vad", "false");
    ASSERT_EQ(std::string(processor->GetOption("use_vad")), "false");
  }
  SUCCEED();
}

// Test consecutive silence logic for nemotron: VAD keeps silence chunks for context until the
// consecutive-silence threshold is reached, then drops. This is nemotron-specific; moonshine
// uses VAD for utterance segmentation instead (see MoonshineVadSegmentation).
TEST(StreamingASRTests, VadConsecutiveSilence) {
  const std::string model_path = std::string(MODEL_PATH) + "nemotron-speech-streaming";
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Nemotron streaming model not found at " << model_path;

  // VAD requires a "vad" section in genai_config.json and the silero_vad.onnx file. Skip if absent.
  if (!ModelHasVad(model_path))
    GTEST_SKIP() << "Nemotron model is not set up for VAD, skipping test";

  auto model = OgaModel::Create(model_path.c_str());
  auto processor = OgaStreamingProcessor::Create(*model);
  processor->SetOption("use_vad", "true");
  // silence_duration_ms must be an integer multiple of the chunk duration; 1680 ms / 560 ms = 3
  // chunks, so the first 2 silence chunks are kept and the 3rd dropped.
  processor->SetOption("silence_duration_ms", "1680");

  const size_t chunk_samples = 8960;  // nemotron chunk size
  std::vector<float> silence(chunk_samples, 0.0f);

  // First 2 silence chunks are preserved (within the tolerance), the 3rd is dropped.
  auto mel1 = processor->Process(silence.data(), silence.size());
  ASSERT_NE(mel1, nullptr);  // Chunk 1: processed

  auto mel2 = processor->Process(silence.data(), silence.size());
  ASSERT_NE(mel2, nullptr);  // Chunk 2: processed

  auto mel3 = processor->Process(silence.data(), silence.size());
  ASSERT_EQ(mel3, nullptr);  // Chunk 3: dropped (>= threshold of 3)
  SUCCEED();
}

// Helper: read a numeric field ("key": <number>) from a model's genai_config.json. Returns the
// fallback if the file can't be read or the key is not found.
static double ReadConfigNumber(const std::string& model_path, const std::string& key, double fallback) {
  std::ifstream config_file(std::filesystem::path(model_path) / "genai_config.json");
  if (!config_file)
    return fallback;
  const std::string config_json((std::istreambuf_iterator<char>(config_file)),
                                std::istreambuf_iterator<char>());
  std::smatch match;
  if (std::regex_search(config_json, match, std::regex("\"" + key + R"("\s*:\s*([0-9.eE+-]+))")))
    return std::stod(match[1].str());
  return fallback;
}

// Test moonshine's VAD-driven utterance segmentation under our thin-processor architecture.
//
// The older "fat processor" accumulated encoder memory inside Process() and returned nullptr to
// drop pre-utterance silence, so segmentation was observable at the processor boundary. Our
// processor is thin: for every full chunk it ALWAYS emits a NamedTensors bundle
// (audio_chunk + is_silent + is_final) and never drops. The per-chunk VAD verdict rides in the
// is_silent flag, and ALL segmentation (accumulate / flush past min_segment / reset / drop
// pre-utterance silence) happens downstream in the State's RunPipeline. This test therefore:
//   * asserts the thin-processor contract  (a full chunk is never dropped -> Process() != nullptr),
//   * checks the is_silent verdict that drives segmentation (speech -> 0, silence -> 1), and
//   * drives the speech -> silence -> speech pattern through the Generator so the State's
//     segmentation routing actually runs end to end (flush + reset + restart).
// Silero VAD only fires on real human speech, so "speech" uses a synthetic tone with VAD disabled
// (forces is_silent=0, routing through the same accumulation path real speech would) while VAD is
// enabled for the silence that drives the segment boundary.
TEST(StreamingASRTests, MoonshineVadSegmentation) {
  const std::string model_path = std::string(MODEL_PATH) + "moonshine-streaming-small";
  if (!std::filesystem::exists(model_path))
    GTEST_SKIP() << "Moonshine streaming model not found at " << model_path;
  if (!ModelHasVad(model_path))
    GTEST_SKIP() << "Moonshine model is not set up for VAD, skipping test";

  auto model = OgaModel::Create(model_path.c_str());
  auto processor = OgaStreamingProcessor::Create(*model);
  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);

  // Read a scalar int64 flag ("is_silent" / "is_final") emitted alongside the audio chunk.
  auto read_flag = [](OgaNamedTensors& tensors, const char* name) {
    return *static_cast<const int64_t*>(tensors.Get(name)->Data());
  };

  // Read the chunking / segmentation parameters straight from the model config.
  const size_t chunk_samples = static_cast<size_t>(ReadConfigNumber(model_path, "chunk_samples", 8000));
  const float sample_rate = static_cast<float>(ReadConfigNumber(model_path, "sample_rate", 16000));
  const double seconds_per_memory_frame = ReadConfigNumber(model_path, "seconds_per_memory_frame", 0.02);
  const int min_segment_memory_frames =
      static_cast<int>(ReadConfigNumber(model_path, "min_segment_memory_frames", 250));

  constexpr float frequency = 440.0f;
  std::vector<float> speech = GenerateSineWave(chunk_samples, frequency, sample_rate);
  std::vector<float> silence(chunk_samples, 0.0f);

  // 1) Initial speech (VAD disabled): the tone routes through the same accumulation path real
  //    speech would. Every full chunk is emitted (never dropped), tagged non-silent + non-final,
  //    and the State accumulates it into the in-progress segment.
  processor->SetOption("use_vad", "false");
  for (int i = 0; i < 4; ++i) {
    auto chunk = processor->Process(speech.data(), speech.size());
    ASSERT_NE(chunk, nullptr) << "Speech chunk " << i << " must always be emitted (thin processor)";
    EXPECT_EQ(read_flag(*chunk, "is_silent"), 0) << "VAD-off speech chunk should not be silent";
    EXPECT_EQ(read_flag(*chunk, "is_final"), 0) << "Streaming chunks are never is_final";
    DecodeInputs(*generator, chunk.get());
  }

  // 2) Silence (VAD enabled): the processor still emits EVERY chunk (never nullptr), now tagged
  //    silent. Downstream, the State encodes mid-segment silence until the accumulated frames pass
  //    min_segment_memory_frames, then flushes the utterance and resets; after the reset further
  //    silence is pre-utterance and produces no tokens. On the old fat processor these surfaced as
  //    Process()==nullptr; here they are is_silent=1 chunks whose segmentation lives in the State.
  processor->SetOption("use_vad", "true");

  // Each chunk contributes (chunk_samples / sample_rate) / seconds_per_memory_frame memory frames.
  // Feeding min_segment_memory_frames / frames_per_chunk + 2 chunks accumulates past the threshold
  // (flush + reset) and then drives at least one post-reset pre-utterance chunk.
  const int frames_per_chunk =
      static_cast<int>((static_cast<double>(chunk_samples) / sample_rate) / seconds_per_memory_frame);
  const int silence_chunks = min_segment_memory_frames / frames_per_chunk + 2;
  for (int i = 0; i < silence_chunks; ++i) {
    auto chunk = processor->Process(silence.data(), silence.size());
    ASSERT_NE(chunk, nullptr) << "Silence chunk " << i << " is still emitted (never dropped here)";
    EXPECT_EQ(read_flag(*chunk, "is_silent"), 1) << "VAD-on silence chunk should be tagged silent";
    DecodeInputs(*generator, chunk.get());
  }

  // 3) New speech after the reset (VAD disabled): the State rebuilds memory and the chunk is
  //    emitted + tagged non-silent again, confirming the pipeline recovered from the flush/reset.
  processor->SetOption("use_vad", "false");
  auto resumed = processor->Process(speech.data(), speech.size());
  ASSERT_NE(resumed, nullptr) << "New speech after the reset should be emitted";
  EXPECT_EQ(read_flag(*resumed, "is_silent"), 0) << "Resumed speech should not be silent";
  DecodeInputs(*generator, resumed.get());

  // Flush drains the tail with is_final=1 so the State releases held-back lookahead and commits
  // the final segment.
  auto tail = processor->Flush();
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(read_flag(*tail, "is_final"), 1) << "Flush tail must be is_final";
  DecodeInputs(*generator, tail.get());
  SUCCEED();
}

// Run every StreamingASR* test above once per streaming ASR model. The
// instantiation prefix "StreamingASR" combined with the StreamingASRParamTests
// fixture yields test names like
// "StreamingASR/StreamingASRParamTests.StreamingASRCreate/nemotron_speech_streaming",
// with the model appended as the parameter suffix.
INSTANTIATE_TEST_SUITE_P(
    StreamingASR, StreamingASRParamTests,
    ::testing::Values(
        StreamingASRModel{"nemotron-speech-streaming", 8960},
        StreamingASRModel{"moonshine-streaming-small", 8000},
        StreamingASRModel{"moonshine-streaming-tiny", 8000}),
    [](const ::testing::TestParamInfo<StreamingASRModel>& info) {
      std::string name = info.param.subdir;
      std::replace(name.begin(), name.end(), '-', '_');
      return name;
    });

#ifndef PARAKEET_TDT_PATH
#define PARAKEET_TDT_PATH MODEL_PATH "parakeet-tdt"
#endif

#ifndef PARAKEET_TDT_AUDIO_JFK
#define PARAKEET_TDT_AUDIO_JFK MODEL_PATH "audios/jfk.flac"
#endif

#ifndef PARAKEET_TDT_AUDIO_TEDLIUM
#define PARAKEET_TDT_AUDIO_TEDLIUM MODEL_PATH "audios/tedlium_long_120s.flac"
#endif

// Test that the Parakeet TDT model + processor + generator construct correctly.
TEST(CAPITests, ParakeetTdtCreate) {
  if (!std::filesystem::exists(PARAKEET_TDT_PATH))
    GTEST_SKIP() << "Parakeet TDT model not found at " << PARAKEET_TDT_PATH;
  auto model = OgaModel::Create(PARAKEET_TDT_PATH);
  auto processor = OgaMultiModalProcessor::Create(*model);
  ASSERT_NE(processor, nullptr);

  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  ASSERT_NE(generator, nullptr);
}

namespace {
std::string RunParakeetTdt(const std::string& audio_path) {
  auto model = OgaModel::Create(PARAKEET_TDT_PATH);
  auto processor = OgaMultiModalProcessor::Create(*model);
  auto tokenizer_stream = OgaTokenizerStream::Create(*processor);

  std::vector<const char*> paths{audio_path.c_str()};
  auto audios = OgaAudios::Load(paths);
  auto inputs = processor->ProcessAudios("", audios.get());

  auto params = OgaGeneratorParams::Create(*model);
  auto generator = OgaGenerator::Create(*model, *params);
  generator->SetInputs(*inputs);

  std::string transcription;
  while (!generator->IsDone()) {
    generator->GenerateNextToken();
    auto count = generator->GetSequenceCount(0);
    if (count == 0) continue;
    auto last = generator->GetSequenceData(0)[count - 1];
    if (auto piece = tokenizer_stream->Decode(last); piece && *piece) {
      transcription += piece;
    }
  }
  return transcription;
}
}  // namespace

// Transcribe the bundled JFK clip and check the output is non-empty.
TEST(CAPITests, ParakeetTdtTranscribeJfk) {
  if (!std::filesystem::exists(PARAKEET_TDT_PATH))
    GTEST_SKIP() << "Parakeet TDT model not found at " << PARAKEET_TDT_PATH;
  if (!std::filesystem::exists(PARAKEET_TDT_AUDIO_JFK))
    GTEST_SKIP() << "Audio not found: " << PARAKEET_TDT_AUDIO_JFK;

  auto transcription = RunParakeetTdt(PARAKEET_TDT_AUDIO_JFK);
  EXPECT_FALSE(transcription.empty());
}

// Transcribe a 120s TED clip and check the output is non-empty.
TEST(CAPITests, ParakeetTdtTranscribeLong) {
  if (!std::filesystem::exists(PARAKEET_TDT_PATH))
    GTEST_SKIP() << "Parakeet TDT model not found at " << PARAKEET_TDT_PATH;
  if (!std::filesystem::exists(PARAKEET_TDT_AUDIO_TEDLIUM))
    GTEST_SKIP() << "Audio not found: " << PARAKEET_TDT_AUDIO_TEDLIUM;

  auto transcription = RunParakeetTdt(PARAKEET_TDT_AUDIO_TEDLIUM);
  EXPECT_FALSE(transcription.empty());
}

// Test that bot/eot/bor/eor throw for models without these tokens configured
TEST(CAPITests, TokenId_Unsupported) {
  // tiny-random-gpt2 model has type "gpt2" which is NOT in the fallback map → throws
  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  auto tokenizer = OgaTokenizer::Create(*model);

  EXPECT_THROW(tokenizer->GetBotTokenId(), std::runtime_error);
  EXPECT_THROW(tokenizer->GetEotTokenId(), std::runtime_error);
  EXPECT_THROW(tokenizer->GetBorTokenId(), std::runtime_error);
  EXPECT_THROW(tokenizer->GetEorTokenId(), std::runtime_error);
}

TEST(CAPITests, TokenId_FromConfig) {
  // Create a temporary model directory with bot/eot/bor/eor token IDs in model section
  auto temp_dir = std::filesystem::temp_directory_path() / "oga_test_tool_tags";
  std::filesystem::remove_all(temp_dir);  // Clean up any leftover from a previous failed run
  std::filesystem::create_directories(temp_dir);

  // Copy minimal model files from tiny-random-gpt2
  std::string src_dir = MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32";
  for (const auto& entry : std::filesystem::directory_iterator(src_dir)) {
    if (entry.path().filename() != "genai_config.json") {
      std::filesystem::copy_file(entry.path(), temp_dir / entry.path().filename(),
                                 std::filesystem::copy_options::overwrite_existing);
    }
  }

  // Write genai_config.json with token IDs in model section
  {
    std::ofstream f((temp_dir / "genai_config.json").string());
    f << R"({
  "model": {
    "type": "gpt2",
    "pad_token_id": 98,
    "bos_token_id": 98,
    "eos_token_id": 98,
    "vocab_size": 1000,
    "context_length": 512,
    "bot_token_id": 151657,
    "eot_token_id": 151658,
    "bor_token_id": 151659,
    "eor_token_id": 151660,
    "decoder": {
      "session_options": { "provider_options": [] },
      "filename": "past.onnx",
      "num_key_value_heads": 4,
      "head_size": 8,
      "num_hidden_layers": 5,
      "inputs": { "past_names": "past_%d" },
      "outputs": { "present_names": "present_%d" }
    }
  }
})";
  }

  auto model = OgaModel::Create(temp_dir.string().c_str());
  auto tokenizer = OgaTokenizer::Create(*model);

  // Tokenizer returns configured IDs from model section
  EXPECT_EQ(tokenizer->GetBotTokenId(), 151657);
  EXPECT_EQ(tokenizer->GetEotTokenId(), 151658);
  EXPECT_EQ(tokenizer->GetBorTokenId(), 151659);
  EXPECT_EQ(tokenizer->GetEorTokenId(), 151660);

  // Cleanup
  std::filesystem::remove_all(temp_dir);
}

// Regression test for MSRC: malformed audio buffers smaller than the minimum valid
// audio header size must be rejected with an error, not cause a crash.
TEST(CAPITests, LoadAudiosFromBuffersRejectsEmptyBuffer) {
  const void* data_ptr = nullptr;
  size_t data_size = 0;
  OgaAudios* audios = nullptr;
  OgaResult* result = OgaLoadAudiosFromBuffers(&data_ptr, &data_size, 1, &audios);

  // Should return an error for empty buffers.
  ASSERT_NE(result, nullptr);
  EXPECT_NE(std::string(OgaResultGetError(result)).find("empty"), std::string::npos);
  OgaDestroyResult(result);
  // audios should not have been created
  EXPECT_EQ(audios, nullptr);
}

TEST(CAPITests, RequestSetDraftTokensRejectsNullArguments) {
  OgaResult* result = OgaRequestSetDraftTokens(nullptr, nullptr);

  ASSERT_NE(result, nullptr);
  EXPECT_NE(std::string(OgaResultGetError(result)).find("must not be null"),
            std::string::npos);
  OgaDestroyResult(result);
}

TEST(CAPITests, RequestSetDraftTokensRunsProposal) {
  auto model = OgaModel::Create(
      MODEL_PATH "engine/synthetic-paged-per-token");
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 16);
  auto engine = OgaEngine::Create(*model);

  size_t max_drafts{};
  OgaCheckResult(OgaEngineMaxDraftTokensPerProposal(
      engine.get(), &max_drafts));
  ASSERT_GE(max_drafts, 2u);

  OgaResult* off_thread_result{};
  std::thread off_owner_thread([&] {
    size_t unused{};
    off_thread_result = OgaEngineMaxDraftTokensPerProposal(
        engine.get(), &unused);
  });
  off_owner_thread.join();
  std::unique_ptr<OgaResult> owned_off_thread_result{off_thread_result};
  ASSERT_NE(owned_off_thread_result, nullptr);
  EXPECT_NE(
      std::string(owned_off_thread_result->GetError()).find("owner thread"),
      std::string::npos);

  auto request = engine->CreateRequest(*params);
  auto turn_options = request->CreateTurnOptions();
  turn_options->SetMaxGeneratedTokens(4);
  const std::array<int32_t, 3> prompt{2, 3, 4};
  request->BeginTurn(prompt, turn_options.get());
  EXPECT_EQ(RunOne(*engine).token, 9);

  const std::array<int32_t, 2> proposed_tokens{15, 22};
  auto proposal = OgaSequences::Create();
  proposal->Append(proposed_tokens.data(), proposed_tokens.size());
  OgaCheckResult(OgaRequestSetDraftTokens(
      request.get(), proposal.get()));

  auto buffer = engine->CreateEventBuffer(3);
  ASSERT_EQ(engine->Run(*buffer), 3u);
  const std::array<int32_t, 3> expected_tokens{15, 22, 30};
  for (size_t index = 0; index < expected_tokens.size(); ++index) {
    const auto* event = buffer->Get(index);
    ASSERT_NE(event, nullptr);
    EXPECT_EQ(event->Token(), expected_tokens[index]);
    EXPECT_NE(event->Flags() & OgaEngineEventFlag_Token, 0u);
  }
  EXPECT_NE(
      buffer->Get(2)->Flags() & OgaEngineEventFlag_TurnFinished,
      0u);
  request->Close();
}

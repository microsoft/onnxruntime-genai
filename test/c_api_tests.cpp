#include <gtest/gtest.h>
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <iostream>
#include <ort_genai.h>
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif
TEST(CAPITests, TokenizerCAPI) {
#if TEST_PHI2
  auto model = OgaModel::Create(MODEL_PATH "phi-2");
  auto tokenizer = OgaTokenizer::Create(*model);

  // Encode single decode single
  {
    const char* input_string = "She sells sea shells by the sea shore.";
    auto input_sequences = OgaSequences::Create();
    tokenizer->Encode(input_string, *input_sequences);

    auto out_string = tokenizer->Decode(input_sequences->Get(0));
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
    for (auto& string : input_strings) tokenizer->Encode(string, *sequences);
  }

  // Decode one at a time
  for (size_t i = 0; i < sequences->Count(); i++) {
    auto out_string = tokenizer->Decode(sequences->Get(i));
    std::cout << "Decoded string:" << out_string << std::endl;
    if (strcmp(input_strings[i], out_string) != 0) throw std::runtime_error("Token decoding mismatch");
  }

  // Stream Decode one at a time
  for (size_t i = 0; i < sequences->Count(); i++) {
    auto tokenizer_stream = OgaTokenizerStream::Create(*tokenizer);

    std::span<const int32_t> sequence = sequences->Get(i);
    std::string stream_result;
    for (auto& token : sequence) {
      stream_result += tokenizer_stream->Decode(token);
    }
    std::cout << "Stream decoded string:" << stream_result << std::endl;
    if (strcmp(input_strings[i], stream_result.c_str()) != 0)
      throw std::runtime_error("Stream token decoding mismatch");
  }
#endif
}

TEST(CAPITests, EndToEndPhiBatch) {
#if TEST_PHI2
  auto model = OgaModel::Create(MODEL_PATH "phi-2");
  auto tokenizer = OgaTokenizer::Create(*model);

  const char* input_strings[] = {
      "This is a test.",
      "Rats are awesome pets!",
      "The quick brown fox jumps over the lazy dog.",
  };

  auto input_sequences = OgaSequences::Create();
  for (auto& string : input_strings) tokenizer->Encode(string, *input_sequences);

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", 20);
  params->SetInputSequences(*input_sequences);

  auto output_sequences = model->Generate(*params);

  // Decode The Batch
  for (size_t i = 0; i < output_sequences->Count(); i++) {
    auto out_string = tokenizer->Decode(output_sequences->Get(i));
    std::cout << "Decoded string:" << out_string << std::endl;
  }
#endif
}

TEST(CAPITests, Tensor_And_AddExtraInput) {
  // Create a [3 4] shaped tensor
  std::array<float, 12> data{0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23};
  std::vector<int64_t> shape{3, 4};  // Use vector so we can easily compare for equality later

  auto tensor = OgaTensor::Create(data.data(), shape.data(), shape.size(), OgaElementType_float32);

  EXPECT_EQ(tensor->Data(), data.data());
  EXPECT_EQ(tensor->Shape(), shape);
  EXPECT_EQ(tensor->Type(), OgaElementType_float32);

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = OgaGeneratorParams::Create(*model);
  params->SetModelInput("test_input", *tensor);
}

TEST(CAPITests, LoraManagement) {
  const std::string adapter_name_1 = "adapter_1";
  const std::string adapter_name_2 = "adapter_2";

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  constexpr std::array<int64_t, 2> input_ids_shape{2, 4};
  constexpr std::array<int32_t, 8U> input_ids{0, 0, 0, 52, 0, 0, 195, 731};
  const auto batch_size = input_ids_shape[0];
  const auto input_sequence_length = input_ids_shape[1];
  constexpr int max_length = 10;

  const char* const adapter[] = {adapter_name_1.c_str()};
  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetInputIDs(input_ids.data(), input_ids.size(), input_sequence_length, batch_size);

  // Set active non-existing adapters show throw
  ASSERT_THROW(params->SetActiveAdapterNames(adapter), std::runtime_error);

  auto lora_manager = model->GetLoraManager();

  lora_manager.CreateAdapter(adapter_name_1);

  // Creating a duplicate name should throw
  ASSERT_THROW(lora_manager.CreateAdapter(adapter_name_1), std::runtime_error);

  lora_manager.CreateAdapter(adapter_name_2);

  // Now we can activate it
  ASSERT_NO_THROW(params->SetActiveAdapterNames(adapter));

  // Two shapes with different lora_r placements
  const std::array<int64_t, 2> lora_param_shape_1 = {4, 2};
  const std::array<int64_t, 2> lora_param_shape_2 = {2, 4};

  // Lora parameter data
  std::array<float, 8> lora_param = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  auto param_1 = OgaTensor::Create(lora_param.data(), lora_param_shape_1.data(), lora_param_shape_1.size(),
                                   OgaElementType_float32);

  lora_manager.AddLoraAdapterParameter(adapter_name_1, "lora_param_1", *param_1);

  auto param_2 = OgaTensor::Create(lora_param.data(), lora_param_shape_2.data(), lora_param_shape_2.size(),
                                   OgaElementType_float32);

  lora_manager.AddLoraAdapterParameter(adapter_name_2, "lora_param_2", *param_2);

  // At this point, all lora_parameters should be copied to the created state.
  auto generator = OgaGenerator::Create(*model, *params);
  // Test hack. Verify that the created state has lora params in it.
  Generators::Generator* gen = reinterpret_cast<Generators::Generator*>(generator.get());
  auto& input_names = gen->state_->input_names_;
  ASSERT_EQ(input_names.size(), gen->state_->inputs_.size());
  auto hit = std::find_if(input_names.begin(), input_names.end(), [&](const std::string& name) {
    return name == "lora_param_1";
  });

  ASSERT_NE(input_names.end(), hit);
  
  hit = std::find_if(input_names.begin(), input_names.end(),
                          [&](const std::string& name) { return name == "lora_param_2"; });
  ASSERT_NE(input_names.end(), hit);

  // Removing adapters while is in use is also OK
  lora_manager.RemoveAdapter(adapter_name_1);
  lora_manager.RemoveAdapter(adapter_name_2);
}

TEST(CAPITests, Logging) {
  // Trivial test to ensure the API builds properly
  Oga::SetLogBool("enabled", true);
  Oga::SetLogString(
      "filename", nullptr);  // If we had a filename set, this would stop logging to the file and go back to the console
  Oga::SetLogBool("enabled", false);
}

// DML doesn't support GPT attention
#if !USE_DML
TEST(CAPITests, GreedySearchGptFp32CAPI) {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{0, 0, 0,   52,  204, 204, 204, 204, 204, 204,
                                       0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto input_sequence_length = input_ids_shape[1];
  auto batch_size = input_ids_shape[0];
  int max_length = 10;

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output
  // tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20 And copy the resulting gpt2_init_past_fp32.onnx file
  // into these two files (as it's the same for gpt2)

  auto model = OgaModel::Create(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  auto params = OgaGeneratorParams::Create(*model);
  params->SetSearchOption("max_length", max_length);
  params->SetInputIDs(input_ids.data(), input_ids.size(), input_sequence_length, batch_size);

  auto generator = OgaGenerator::Create(*model, *params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();
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

  // Test high level API
  auto sequences = model->Generate(*params);

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    const auto sequence_length = sequences->SequenceCount(i);
    const auto* sequence_data = sequences->SequenceData(i);

    ASSERT_LE(sequence_length, max_length);

    const auto* expected_output_start = &expected_output[i * max_length];
    EXPECT_TRUE(0 == std::memcmp(expected_output_start, sequence_data, sequence_length * sizeof(int32_t)));
  }
}
#endif

#if TEST_PHI2

struct Phi2Test {
  Phi2Test() {
    model_ = OgaModel::Create(MODEL_PATH "phi-2");
    tokenizer_ = OgaTokenizer::Create(*model_);

    input_sequences_ = OgaSequences::Create();

    const char* input_strings[] = {
        "This is a test.",
        "Rats are awesome pets!",
        "The quick brown fox jumps over the lazy dog.",
    };

    for (auto& string : input_strings) tokenizer_->Encode(string, *input_sequences_);

    params_ = OgaGeneratorParams::Create(*model_);
    params_->SetInputSequences(*input_sequences_);
    params_->SetSearchOption("max_length", 40);
  }

  void Run() {
    // Low level loop
    {
      auto generator = OgaGenerator::Create(*model_, *params_);

      while (!generator->IsDone()) {
        generator->ComputeLogits();
        generator->GenerateNextToken();
      }

      // Decode One at a time
      for (size_t i = 0; i < 3; i++) {
        auto out_string = tokenizer_->Decode(generator->GetSequence(i));
        std::cout << "Decoded string:" << out_string << std::endl;
      }
    }

    // High level
    {
      auto output_sequences = model_->Generate(*params_);

      // Decode The Batch
      for (size_t i = 0; i < output_sequences->Count(); i++) {
        auto out_string = tokenizer_->Decode(output_sequences->Get(i));
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

#endif  // TEST_PHI2

void CheckResult(OgaResult* result) {
  if (result) {
    std::string string = OgaResultGetError(result);
    OgaDestroyResult(result);
    throw std::runtime_error(string);
  }
}

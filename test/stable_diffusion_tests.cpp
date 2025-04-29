// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <span>
#include <cstring>  // for memcmp
#include <numeric>
#include <random>
#include "span.h"
#define OGA_USE_SPAN 1
#include <ort_genai.h>
#include <gtest/gtest.h>
#include "models/onnxruntime_api.h"
//#include "onnxruntime_c_api.h"

// Our working directory is generators/build so one up puts us in the root directory:
#ifndef MODEL_PATH
#define MODEL_PATH "../../test/test_models/"
#endif

TEST(StableDiffusionTests, ClipTokenizer) {
  auto config = OgaConfig::Create(MODEL_PATH "sd");

  auto model = OgaModel::Create(*config);
  auto tokenizer = OgaTokenizer::Create(*model);

  auto sequences = OgaSequences::Create();
  tokenizer->Encode("Capybara with his mom and dad in a beautiful stream", *sequences);

  std::vector<int32_t> expected_tokens{49406, 1289, 88, 19345, 593, 787, 2543, 537, 2639, 530, 320, 1215, 3322, 49407};

  EXPECT_TRUE(1 == sequences->Count());
  EXPECT_TRUE(sequences->SequenceCount(0) == expected_tokens.size());
  EXPECT_TRUE(0 == std::memcmp(sequences->SequenceData(0), expected_tokens.data(), expected_tokens.size() * sizeof(int32_t)));
}

TEST(StableDiffusionTests, TextEmbeddings) {
  auto config = OgaConfig::Create(MODEL_PATH "sd");

  auto model = OgaModel::Create(*config);
  auto tokenizer = OgaTokenizer::Create(*model);

  auto sequences = OgaSequences::Create();
  tokenizer->Encode("Capybara with his mom and dad in a beautiful stream", *sequences);

  // create the OrtSession
  // Ort::InitApi();
  std::unique_ptr<OrtEnv> p_env = OrtEnv::Create(ORT_LOGGING_LEVEL_WARNING, "test");

  std::unique_ptr<OrtSessionOptions> session_options = OrtSessionOptions::Create();

  std::unique_ptr<OrtSession> p_session_ = OrtSession::Create(
      *p_env,
      MODEL_PATH "text_encoder/model.onnx",
      session_options.get());

  std::unique_ptr<OrtMemoryInfo> p_memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  auto allocator = Ort::Allocator::Create(*p_session_, *p_memory_info);

  // enable_cuda_graph is false in first version
  // create input_ids tensor
  int32_t batch_size = 1;
  int32_t max_sequence_length = 77;
  std::vector<int64_t> input_ids_shape{batch_size, max_sequence_length};

  std::unique_ptr<OrtValue> p_input_tensor = OrtValue::CreateTensor<int32_t>(*allocator, std::span{input_ids_shape});
  int32_t* input_ids_data = p_input_tensor->GetTensorMutableData<int32_t>();

  // if the length of input_ids is larger than max_sequence_length, we need to truncate it
  if (sequences->SequenceCount(0) > max_sequence_length) {
    std::copy(sequences->SequenceData(0), sequences->SequenceData(0) + max_sequence_length, input_ids_data);
  }

  std::copy(sequences->SequenceData(0), sequences->SequenceData(0) + sequences->SequenceCount(0), input_ids_data);

  // if the length of input_ids is smaller than max_sequence_length, we need to pad it
  if (sequences->SequenceCount(0) < max_sequence_length) {
    std::fill(input_ids_data + sequences->SequenceCount(0), input_ids_data + max_sequence_length, 0);
  }
  // Bind input tensors and run inference
  auto io_binding = OrtIoBinding::Create(*p_session_);
  io_binding->BindInput("input_ids", *p_input_tensor);

  // Bind output text_embeddings tensor
  int32_t hidden_size = 1024;
  std::vector<int64_t> output_embeddings_shape{batch_size, max_sequence_length, hidden_size};
  std::unique_ptr<OrtValue> p_output_tensor = OrtValue::CreateTensor<float>(*allocator, std::span{output_embeddings_shape});
  io_binding->BindOutput("output_embeddings", *p_output_tensor);

  std::unique_ptr<OrtRunOptions> run_options = OrtRunOptions::Create();

  io_binding->SynchronizeInputs();
  p_session_->Run(run_options.get(), *io_binding);
  io_binding->SynchronizeOutputs();

  // Get output tensor
  auto output_tensor = io_binding->GetOutputValues();

  auto output_tensor_data = output_tensor[0]->GetTensorMutableData<float>();


  std::vector<int32_t>
      expected_tokens{49406, 1289, 88, 19345, 593, 787, 2543, 537, 2639, 530, 320, 1215, 3322, 49407};

  EXPECT_TRUE(1 == sequences->Count());
  EXPECT_TRUE(sequences->SequenceCount(0) == expected_tokens.size());
  EXPECT_TRUE(0 == std::memcmp(sequences->SequenceData(0), expected_tokens.data(), expected_tokens.size() * sizeof(int32_t)));
}

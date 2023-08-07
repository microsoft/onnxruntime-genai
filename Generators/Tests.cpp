#include "Generators.h"
#include "onnxruntime_cxx_api_2.h"
#include <assert.h>
#include <iostream>

#include "beam_search_scorer.h"
#include "Sequences.h"
#include "generation_device_helper.h"
#include "logits_processor.h"
#include "greedy_search_impl_base.h"
#include "feeds_fetches_manager.h"
#include "greedy_search_impl_gpt.h"

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_TRUE(a) assert(a)

void Test_BeamSearchTest_GptBeamSearchFp32() {
  auto ort_env = OrtEnv::Create();

  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};
  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  auto info = OrtMemoryInfo::Create("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = OrtValue::CreateTensor(
      *info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = OrtValue::CreateTensor(
      *info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = OrtValue::CreateTensor(
      *info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = OrtValue::CreateTensor(
      *info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = OrtValue::CreateTensor(
      *info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = OrtValue::CreateTensor(
      *info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = OrtValue::CreateTensor(
      *info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<OrtValue*> ort_inputs;
  ort_inputs.push_back(input_ids_tensor.get());
  ort_inputs.push_back(max_length_tensor.get());
  ort_inputs.push_back(min_length_tensor.get());
  ort_inputs.push_back(num_beams_tensor.get());
  ort_inputs.push_back(num_return_sequences_tensor.get());
  ort_inputs.push_back(length_penalty_tensor.get());
  ort_inputs.push_back(repetition_penalty_tensor.get());
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  auto session_options = OrtSessionOptions::Create();
#ifdef USE_CUDA
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
  auto session = OrtSession::Create(*ort_env, ORT_TSTR("C:/code/github/generators/Generators/models/tiny_gpt2_beamsearch.onnx"), session_options.get());
  auto ort_outputs = session->Run(nullptr, input_names, ort_inputs.data(), ort_inputs.size(),
                                  output_names, 1);

  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sequences = ort_outputs[0];
  ASSERT_TRUE(sequences->IsTensor());

  auto result_ts = sequences->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts->GetElementType());

  ASSERT_EQ(expected_output_shape, result_ts->GetShape());
  const auto* result_vals = sequences->GetTensorData<int32_t>();
  auto result_span = gsl::make_span(result_vals, expected_output.size());
  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
  std::cout << "Test_BeamSearchTest_GptBeamSearchFp32 complete\r\n";
}

void Test_Lib_BeamSearchTest_GptBeamSearchFp32() {
  auto ort_env = OrtEnv::Create();

  int32_t max_length{20};
  float length_penalty{1.0f};

  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  auto info = OrtMemoryInfo::Create("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = OrtValue::CreateTensor(
      *info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
  SearchParams params;
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.max_length=max_length;
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.num_beams = 4;

  Gpt gpt(*ort_env,
          ORT_TSTR("C:/code/github/generators/Generators/models/gpt2_fp32.onnx"),
          ORT_TSTR("C:/code/github/generators/Generators/models/gpt2_fp32.onnx"),
          std::move(input_ids_tensor), params);

  Search search{gpt, params};

  while (!search.IsDone()) {
    search.RunModel();

    // Scoring
    Processors::MinLength(search, 1);
    Processors::RepetitionPenalty(search, 1.0f);
    // Processors::LengthPenalty(search, 1.0f);

    // Sampling goes here

    // TODO: Are these steps always the same? If so, merge into one function
    search.NextTokensFromLogits();
    search.CheckForEOS();
    search.AppendNextTokensToSequences();
  }

  std::vector<int32_t> output_sequence(search.params_.batch_size*max_length);
  search.Finalize(1, output_sequence, {});

  // Verify outputs match expected outputs
  for (int i = 0; i < search.params_.batch_size; i++) {
//    auto sequence = search.sequences_.GetSequence(i);
    auto sequence = gsl::make_span<int32_t>(output_sequence.data()+max_length*i, max_length);
    auto* expected_output_start = &expected_output[i * search.params_.max_length];
    ASSERT_TRUE(std::equal(expected_output_start, expected_output_start + search.params_.max_length, sequence.begin(), sequence.end()));
  }

  std::cout << "Test_Lib_BeamSearchTest_GptBeamSearchFp32 complete\r\n";
}

void Test_GreedySearchTest_GptGreedySearchFp32() {
  auto ort_env = OrtEnv::Create();

  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{
      0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{10};
  std::vector<int32_t> min_length{1};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto info = OrtMemoryInfo::Create("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = OrtValue::CreateTensor(
      *info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = OrtValue::CreateTensor(
      *info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = OrtValue::CreateTensor(
      *info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = OrtValue::CreateTensor(
      *info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<OrtValue*> ort_inputs;
  ort_inputs.push_back(input_ids_tensor.get());
  ort_inputs.push_back(max_length_tensor.get());
  ort_inputs.push_back(min_length_tensor.get());
  ort_inputs.push_back(repetition_penalty_tensor.get());
  const char* input_names[] = {"input_ids", "max_length", "min_length", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  auto session_options = OrtSessionOptions::Create();
#ifdef USE_CUDA
  Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

  auto session = OrtSession::Create(*ort_env, ORT_TSTR("C:/code/github/generators/Generators/models/tiny_gpt2_greedysearch_with_init_decoder.onnx"), session_options.get());

  auto ort_outputs = session->Run(nullptr, input_names, ort_inputs.data(), ort_inputs.size(), output_names, 1);

  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sequences = ort_outputs[0];
  ASSERT_TRUE(sequences->IsTensor());

  auto result_ts = sequences->GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts->GetElementType());

  ASSERT_EQ(expected_output_shape, result_ts->GetShape());
  const auto* result_vals = sequences->GetTensorData<int32_t>();
  auto result_span = gsl::make_span(result_vals, expected_output.size());
  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));

  std::cout << "Test_GreedySearchTest_GptGreedySearchFp32 complete\r\n";
}

void Test_Lib_GreedySearchTest_GptGreedySearchFp32() {
  auto ort_env = OrtEnv::Create();

  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  int32_t max_length{10};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto info = OrtMemoryInfo::Create("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = OrtValue::CreateTensor(
      *info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  SearchParams params;
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);

  Gpt gpt(*ort_env,
          ORT_TSTR("C:/code/github/generators/Generators/models/gpt2_fp32.onnx"),
          ORT_TSTR("C:/code/github/generators/Generators/models/gpt2_fp32.onnx"),
          std::move(input_ids_tensor), params);

  Search search{gpt, params};

  while (!search.IsDone()) {
    search.RunModel();

    // Scoring
    Processors::MinLength(search, 1);
    Processors::RepetitionPenalty(search, 1.0f);

    // Sampling goes here

    // TODO: Are these steps always the same? If so, merge into one function
    search.NextTokensFromLogits();
    search.CheckForEOS();
    search.AppendNextTokensToSequences();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < search.params_.batch_size; i++) {
    auto sequence = search.sequences_.GetSequence(i);
    auto* expected_output_start = &expected_output[i * search.params_.max_length];
    ASSERT_TRUE(std::equal(expected_output_start, expected_output_start+search.params_.max_length, sequence.begin(), sequence.end()));
  }

  std::cout << "Test_Lib_GreedySearchTest_GptGreedySearchFp32 complete\r\n";
}

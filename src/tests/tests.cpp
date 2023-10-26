#include "../generators.h"
#include "../search.h"
#include "../models/gpt_cpu.h"
#if USE_CUDA
#include "../search_cuda.h"
#include "../models/gpt_cuda.h"
#endif
#include <iostream>

// Our working directory is generators/build so one up puts us in the root directory:
#define MODEL_PATH "../test_models/"

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_TRUE(a) assert(a)

std::unique_ptr<OrtEnv> g_ort_env;

void Test_BeamSearchTest_GptBeamSearchFp32() {

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

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)

  Generators::Gpt gpt(*g_ort_env, ORT_TSTR_ON_MACRO(MODEL_PATH "hf-internal-testing/tiny-random-gpt2_past_fp32.onnx"));

  Generators::SearchParams params;
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;
  params.max_length = max_length;
  params.vocab_size = gpt.GetVocabSize();
  params.eos_token_id = params.pad_token_id = 98;
  params.num_beams = 4;

  Generators::BeamSearch search{params};
  gpt.CreateInputs(search.sequence_lengths_, params);

  while (!search.IsDone()) {
    gpt.Run(search.GetNextTokens(), search.GetNextIndices(), search.GetSequenceLength());
    search.SetLogits(gpt.GetLogits());

    // Scoring
    Generators::Processors::MinLength(search, 1);
    Generators::Processors::RepetitionPenalty(search, 1.0f);

    search.SelectTop();
  }

  std::vector<int32_t> output_sequence(search.params_.batch_size*max_length);
  search.Finalize(1, output_sequence, {});

  // Verify outputs match expected outputs
  for (int i = 0; i < search.params_.batch_size; i++) {
    auto sequence = std::span<int32_t>(output_sequence.data() + max_length * i, max_length);
    auto* expected_output_start = &expected_output[i * search.params_.max_length];
    ASSERT_TRUE(std::equal(expected_output_start, expected_output_start + search.params_.max_length, sequence.begin(), sequence.end()));
  }

  std::cout << "Test_BeamSearchTest_GptBeamSearchFp32 complete\r\n";
}

void Test_GreedySearchTest_GptGreedySearchFp32() {

  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  Generators::Gpt gpt(*g_ort_env, ORT_TSTR_ON_MACRO(MODEL_PATH "hf-internal-testing/tiny-random-gpt2_past_fp32.onnx"));

  Generators::SearchParams params;
  params.max_length = 10;
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;
  params.vocab_size = gpt.GetVocabSize();
  params.eos_token_id = params.pad_token_id = 98;

  Generators::GreedySearch search{params};
  gpt.CreateInputs(search.sequence_lengths_, params);

  while (!search.IsDone()) {
    gpt.Run(search.GetNextTokens(), {}, search.GetSequenceLength());
    search.SetLogits(gpt.GetLogits());

    // Scoring
    Generators::Processors::MinLength(search, 1);
    Generators::Processors::RepetitionPenalty(search, 1.0f);

    search.SelectTop();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < search.params_.batch_size; i++) {
    auto sequence = search.sequences_.GetSequence(i);
    auto* expected_output_start = &expected_output[i * search.params_.max_length];
    ASSERT_TRUE(std::equal(expected_output_start, expected_output_start+search.params_.max_length, sequence.begin(), sequence.end()));
  }

  std::cout << "Test_GreedySearchTest_GptGreedySearchFp32 complete\r\n";
}

#if USE_CUDA
void Test_GreedySearchTest_GptGreedySearchFp32_Cuda() {
  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  int32_t max_length{10};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  cudaError_t cuda_status = cudaSetDevice(0);
  assert(cuda_status == cudaSuccess);

  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)
  Generators::Gpt_Cuda gpt(*g_ort_env, ORT_TSTR_ON_MACRO(MODEL_PATH "hf-internal-testing/tiny-random-gpt2_past_fp32.onnx"), cuda_stream);

  Generators::SearchParams_Cuda params;
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;
  params.vocab_size = gpt.GetVocabSize();
  params.eos_token_id = params.pad_token_id = 98;
  params.cuda_stream = cuda_stream;

  Generators::GreedySearch_Cuda search{params};
  gpt.CreateInputs(search.sequence_lengths_, params);

  while (!search.IsDone()) {
    gpt.Run(search.GetNextTokens(), {}, search.GetSequenceLength());
    search.SetLogits(gpt.GetLogits());

    // Scoring
    Generators::Processors_Cuda::MinLength(search, 1);

    search.SelectTop();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < search.params_.batch_size; i++) {
    auto sequence_gpu = search.sequences_.GetSequence(i);
    auto sequence = std::make_unique<int32_t[]>(max_length);
    cudaMemcpyAsync(sequence.get(), sequence_gpu.data(), max_length * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    auto* expected_output_start = &expected_output[i * search.params_.max_length];
    ASSERT_TRUE(std::equal(expected_output_start, expected_output_start + search.params_.max_length, sequence.get(), sequence.get()+max_length));
  }

  std::cout << "Test_GreedySearchTest_GptGreedySearchFp32_Cuda complete\r\n";
}

void Test_BeamSearchTest_GptBeamSearchFp32_Cuda() {
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

  cudaError_t cuda_status = cudaSetDevice(0);
  assert(cuda_status == cudaSuccess);

  cudaStream_t cuda_stream;
  cudaStreamCreate(&cuda_stream);

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)

  Generators::Gpt_Cuda gpt(*g_ort_env,
                           ORT_TSTR_ON_MACRO(MODEL_PATH "hf-internal-testing/tiny-random-gpt2_past_fp32.onnx"), cuda_stream);

  Generators::SearchParams_Cuda params;
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;
  params.max_length = max_length;
  params.num_beams = 4;
  params.vocab_size = gpt.GetVocabSize();
  params.eos_token_id = params.pad_token_id = 98;
  params.cuda_stream = cuda_stream;

  Generators::BeamSearch_Cuda search{params};
  gpt.CreateInputs(search.sequence_lengths_, params);

  while (!search.IsDone()) {
    gpt.Run(search.GetNextTokens(), search.GetNextIndices(), search.GetSequenceLength());
    search.SetLogits(gpt.GetLogits());

    // Scoring
    Generators::Processors_Cuda::MinLength(search, 1);
    Generators::Processors_Cuda::RepetitionPenalty(search, 1.0f);

    search.SelectTop();
  }

  size_t sequence_length=search.params_.batch_size*max_length;
  auto output_sequence_cuda = Generators::CudaMallocArray<int32_t>(sequence_length);
  auto output_sequence = std::make_unique<int32_t[]>(sequence_length);

  search.Finalize(1, std::span<int32_t>(output_sequence_cuda.get(), sequence_length), {});
  cudaMemcpyAsync(output_sequence.get(), output_sequence_cuda.get(), sequence_length * sizeof(int32_t), cudaMemcpyDeviceToHost, cuda_stream);
  cudaStreamSynchronize(cuda_stream);

  // Verify outputs match expected outputs
  for (int i = 0; i < search.params_.batch_size; i++) {
    auto sequence = std::span<int32_t>(output_sequence.get() + max_length * i, max_length);
    auto* expected_output_start = &expected_output[i * search.params_.max_length];
    ASSERT_TRUE(std::equal(expected_output_start, expected_output_start + search.params_.max_length, sequence.begin(), sequence.end()));
  }

  std::cout << "Test_BeamSearchTest_GptBeamSearchFp32_Cuda complete\r\n";
}
#endif

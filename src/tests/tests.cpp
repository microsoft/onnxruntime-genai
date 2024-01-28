#include "../generators.h"
#include "../search.h"
#include "../models/model.h"
#include <chrono>
#include <iostream>

// Our working directory is generators/build so one up puts us in the root directory:
#define MODEL_PATH "../test_models/"

std::unique_ptr<OrtEnv> g_ort_env;

// To generate this file:
// python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
// And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)
static const std::pair<const char*, const char*> c_tiny_gpt2_model_paths[] = {
    {MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32", "fp32"},
    {MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp16", "fp16"},
};

void Test_GreedySearch_Gpt_Fp32() {
  std::cout << "Test_GreedySearch_Gpt fp32" << std::flush;

  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  Generators::SearchParams params{*model};
  params.max_length = 10;
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;

  auto search = params.CreateSearch();
  auto state = model->CreateState(search->GetSequenceLengths(), params);

  while (!search->IsDone()) {
    search->SetLogits(state->Run(search->GetSequenceLength(), search->GetNextTokens()));

    // Scoring
    //    Generators::Processors::MinLength(search, 1);
    //    Generators::Processors::RepetitionPenalty(search, 1.0f);

    search->SelectTop();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < params.batch_size; i++) {
    auto sequence = search->GetSequence(i).GetCPU();
    auto* expected_output_start = &expected_output[i * params.max_length];
    if (!std::equal(expected_output_start, expected_output_start + params.max_length, sequence.begin(), sequence.end()))
      throw std::runtime_error("Test Results Mismatch");
  }

  std::cout << " - complete\r\n";
}

void Test_BeamSearch_Gpt_Fp32() {
  std::cout << "Test_BeamSearch_Gpt fp32" << std::flush;

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

  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");

  Generators::SearchParams params{*model};
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;
  params.max_length = 20;
  params.length_penalty = 1.0f;
  params.num_beams = 4;

  Generators::BeamSearch_Cpu search{params};
  auto state = model->CreateState(search.sequence_lengths_, params);

  while (!search.IsDone()) {
    search.SetLogits(state->Run(search.GetSequenceLength(), search.GetNextTokens(), search.GetNextIndices()));

    // Scoring
    Generators::Processors::MinLength(search, 1);
    Generators::Processors::RepetitionPenalty(search, 1.0f);

    search.SelectTop();
  }

  std::vector<int32_t> output_sequence(search.params_.batch_size * search.params_.max_length);
  search.Finalize(1, Generators::cpu_span<int32_t>{output_sequence}, {});

  // Verify outputs match expected outputs
  for (int i = 0; i < search.params_.batch_size; i++) {
    auto sequence = std::span<int32_t>(output_sequence.data() + search.params_.max_length * i, search.params_.max_length);
    auto* expected_output_start = &expected_output[i * search.params_.max_length];
    if (!std::equal(expected_output_start, expected_output_start + search.params_.max_length, sequence.begin(), sequence.end()))
      throw std::runtime_error("Test Results Mismatch");
  }

  std::cout << " - complete\r\n";
}

#if USE_CUDA

void Test_GreedySearch_Gpt_Cuda(const char* model_path, const char* model_label) {
  std::cout << "Test_GreedySearch_Gpt_Cuda " << model_label << std::flush;

  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto provider_options = Generators::GetDefaultProviderOptions(Generators::DeviceType::CUDA);
  auto model = Generators::CreateModel(*g_ort_env, model_path, &provider_options);

  Generators::SearchParams params{*model};
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.max_length = 10;
  params.input_ids = input_ids;

  auto search = params.CreateSearch();
  auto state = model->CreateState(search->GetSequenceLengths(), params);

  while (!search->IsDone()) {
    search->SetLogits(state->Run(search->GetSequenceLength(), search->GetNextTokens()));

    // Scoring
    //    Generators::Processors_Cuda::MinLength(search, 1);

    search->SelectTop();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < params.batch_size; i++) {
    auto sequence_gpu = search->GetSequence(i);
    auto sequence = sequence_gpu.GetCPU();

    auto* expected_output_start = &expected_output[i * params.max_length];
    if (!std::equal(expected_output_start, expected_output_start + params.max_length, sequence.begin(), sequence.end()))
      throw std::runtime_error("Test Results Mismatch");
  }

  std::cout << " - complete\r\n";
}

void Test_GreedySearch_Gpt_Cuda() {
  for (auto model_path : c_tiny_gpt2_model_paths)
    Test_GreedySearch_Gpt_Cuda(model_path.first, model_path.second);
}

void Test_BeamSearch_Gpt_Cuda(const char* model_path, const char* model_label) {
  std::cout << "Test_BeamSearch_Gpt_Cuda " << model_label << std::flush;

  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  auto provider_options = Generators::GetDefaultProviderOptions(Generators::DeviceType::CUDA);

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
  auto model = Generators::CreateModel(*g_ort_env, model_path, &provider_options);

  Generators::SearchParams params{*model};
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;
  params.max_length = 20;
  params.num_beams = 4;
  params.length_penalty = 1.0f;

  auto search = params.CreateSearch();
  auto state = model->CreateState(search->GetSequenceLengths(), params);

  while (!search->IsDone()) {
    search->SetLogits(state->Run(search->GetSequenceLength(), search->GetNextTokens(), search->GetNextIndices()));

    // Scoring
    //    Generators::Processors_Cuda::MinLength(search, 1);
    //    Generators::Processors_Cuda::RepetitionPenalty(search, 1.0f);

    search->SelectTop();
  }

  size_t sequence_length = params.batch_size * params.max_length;
  auto output_sequence_cuda = Generators::CudaMallocArray<int32_t>(sequence_length);
  auto output_sequence_cpu = std::make_unique<int32_t[]>(sequence_length);

  search->Finalize(1, Generators::gpu_span<int32_t>(output_sequence_cuda.get(), sequence_length), {});
  cudaMemcpyAsync(output_sequence_cpu.get(), output_sequence_cuda.get(), sequence_length * sizeof(int32_t), cudaMemcpyDeviceToHost, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);

  // Verify outputs match expected outputs
  for (int i = 0; i < params.batch_size; i++) {
    auto sequence = std::span<int32_t>(output_sequence_cpu.get() + params.max_length * i, params.max_length);
    auto* expected_output_start = &expected_output[i * params.max_length];
    if (!std::equal(expected_output_start, expected_output_start + params.max_length, sequence.begin(), sequence.end()))
      throw std::runtime_error("Test Results Mismatch");
  }

  std::cout << " - complete\r\n";
}

void Test_BeamSearch_Gpt_Cuda() {
  for (auto model_path : c_tiny_gpt2_model_paths)
    Test_BeamSearch_Gpt_Cuda(model_path.first, model_path.second);
}

#include "tests_helper.cuh"
void Test_TopP_Cuda() {
  std::cout << "Test_TopP" << std::flush;

  //////////// Simple Test
  std::vector<int32_t> expected_output{5};
  auto output_span = Generators::cpu_span<int32_t>(expected_output);
  std::vector<float> logits_cpu{0.1, 0.1, 0.1, 0.1, 0.1, 0.5};
  auto logits_gpu = Generators::CudaMallocArray<float>(logits_cpu.size());

  auto provider_options = Generators::GetDefaultProviderOptions(Generators::DeviceType::CUDA);

  Generators::SearchParams params;
  params.max_length = 10;
  params.batch_size = 1;
  params.sequence_length = 1;
  params.vocab_size = 6;
  params.device_type = Generators::DeviceType::CUDA;

  cudaMemcpyAsync(logits_gpu.get(), logits_cpu.data(), logits_cpu.size() * sizeof(float), cudaMemcpyHostToDevice, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  auto search = params.CreateSearch();

  search->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), logits_cpu.size()));
  search->SampleTopP(0.2, 1.0);

  // Verify outputs match expected outputs
  auto next_tokens = search->GetNextTokens().GetCPU();
  if (!std::equal(next_tokens.begin(), next_tokens.end(), output_span.begin(), output_span.end())) 
    throw std::runtime_error("Test Results Mismatch");
  std::cout << " completed simple test\r\n";

  //////////// Batched Test
  expected_output = {1, 2, 3, 4};
  output_span = Generators::cpu_span<int32_t>(expected_output);
  logits_cpu = {0.1, 0.6, 0.1, 0.1, 0.1,
                0.1, 0.1, 0.6, 0.1, 0.1,
                0.1, 0.1, 0.1, 0.6, 0.1,
                0.1, 0.1, 0.1, 0.1, 0.6};
  logits_gpu = Generators::CudaMallocArray<float>(logits_cpu.size());

  params = Generators::SearchParams{};
  params.max_length = 10;
  params.batch_size = 4;
  params.sequence_length = 1;
  params.vocab_size = 5;
  params.device_type = Generators::DeviceType::CUDA;

  cudaMemcpyAsync(logits_gpu.get(), logits_cpu.data(), logits_cpu.size() * sizeof(float), cudaMemcpyHostToDevice, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  search = params.CreateSearch();

  search->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), logits_cpu.size()));
  search->SampleTopP(0.25, 1.0);

  // Verify outputs match expected outputs
  long long total_duration = 0;

  next_tokens = search->GetNextTokens().GetCPU();
  if (!std::equal(next_tokens.begin(), next_tokens.end(), output_span.begin(), output_span.end())) 
    throw std::runtime_error("Test Results Mismatch");
  std::cout << " completed batched test\r\n";

  //////////// Randomized Test
  int vocab_size = 32000; // large number, power of 2, useful for the fisher yates kernel
  int batch_size = 5;

  for (int i = 0; i < 100; i++) {
    logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocHostArray<int>(vocab_size * batch_size);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpyAsync(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost, params.cuda_stream);

    params = Generators::SearchParams{};
    params.max_length = 10;
    params.batch_size = batch_size;
    params.sequence_length = 1;
    params.vocab_size = vocab_size;
    params.device_type = Generators::DeviceType::CUDA;

    search = params.CreateSearch();
    search->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    
    auto start = std::chrono::high_resolution_clock::now();
    search->SampleTopP(0.95, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    total_duration += duration;

    next_tokens = search->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
    // std::cout << "next_tokens: " << next_tokens[0] << ", " << next_tokens[1] << ", " << next_tokens[2] << ", " << next_tokens[3] << ", " << next_tokens[4] << "\r\n";
    // std::cout << "next_token_scores: " << cpu_logits[next_tokens[0]] << ", " << cpu_logits[next_tokens[1]+vocab_size] << ", " << cpu_logits[next_tokens[2]+vocab_size*2] << ", " << cpu_logits[next_tokens[3]+vocab_size*3] << ", " << cpu_logits[next_tokens[4]+vocab_size*4] << "\r\n";

    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = cpu_logits[next_token + vocab_size * b];
      if (next_token_score < 0.0001) {
        std::cout << "next_token_score: " << next_token_score << "\r\n";
        throw std::runtime_error("Test Results Mismatch");
      }
    }
  }

  double averageDuration = static_cast<double>(total_duration) / 100.0;
  std::cout << "Average time taken by top p sampling: "
    << averageDuration << " microseconds" << std::endl;

  std::cout << " completed randomized test\r\n";
  std::cout << " - complete\r\n";
}


void Test_Phi2_Cuda() {
#if TEST_PHI2
  std::cout << "Testing_Phi2\r\n";
#if USE_ORT_EXT

  auto prompt = R"(
def print_prime(n):
'''
Print all primes between 1 and n
'''
)";

  std::cout << "With prompt:" << prompt << "\r\n";

  auto provider_options = Generators::GetDefaultProviderOptions(Generators::DeviceType::CUDA);
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "phi-2", &provider_options);
  auto tokenizer = model->CreateTokenizer();
  auto tokens = tokenizer->Encode(prompt);

  Generators::SearchParams params{*model};
  params.batch_size = 1;
  params.sequence_length = static_cast<int>(tokens.size());
  params.input_ids = tokens;
  params.max_length = 128;

  auto search = params.CreateSearch();
  auto result = model->Generate(params);

  std::cout << tokenizer->Decode(result) << "\r\n";
  std::cout << "Test complete\r\n";
#else
  std::cout << "Test skipped - not built with onnxruntime extensions\r\n";
#endif
#endif
}

#endif
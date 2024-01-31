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

  Generators::GeneratorParams params{*model};
  params.max_length = 10;
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;

  auto generator = Generators::CreateGenerator(*model, params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();
    generator->AppendNextToken_Top();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < params.batch_size; i++) {
    auto sequence = generator->GetSequence(i).GetCPU();
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

  Generators::GeneratorParams params{*model};
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

  Generators::GeneratorParams params{*model};
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.max_length = 10;
  params.input_ids = input_ids;

  auto generator = Generators::CreateGenerator(*model, params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();

    // Scoring
    //    Generators::Processors_Cuda::MinLength(search, 1);

    generator->AppendNextToken_Top();
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < params.batch_size; i++) {
    auto sequence_gpu = generator->GetSequence(i);
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

  Generators::GeneratorParams params{*model};
  params.batch_size = static_cast<int>(input_ids_shape[0]);
  params.sequence_length = static_cast<int>(input_ids_shape[1]);
  params.input_ids = input_ids;
  params.max_length = 20;
  params.num_beams = 4;
  params.length_penalty = 1.0f;

  auto generator = Generators::CreateGenerator(*model, params);

  while (!generator->IsDone()) {
    generator->ComputeLogits();

    // Scoring
    //    Generators::Processors_Cuda::MinLength(search, 1);
    //    Generators::Processors_Cuda::RepetitionPenalty(search, 1.0f);

    generator->AppendNextToken();
  }

  size_t sequence_length = params.batch_size * params.max_length;
  auto output_sequence_cuda = Generators::CudaMallocArray<int32_t>(sequence_length);
  auto output_sequence_cpu = std::make_unique<int32_t[]>(sequence_length);

  generator->search_->Finalize(1, Generators::gpu_span<int32_t>(output_sequence_cuda.get(), sequence_length), {});
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

// TODO: CPU LOGITS AND LOGITS CPU SHOULD NOT BE BOTH THINGS
// LONG FUNCTION, FACTOR OUT THE COMMON PARTS
#include "tests_helper.cuh"

void Batched_Sampling_TopP_Test() {
  // TODO: I don't like that I have to create a model here, but I need to pass it to the generator
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<int32_t> expected_output{1, 2, 3, 4};
  auto output_span = Generators::cpu_span<int32_t>(expected_output);
  std::vector<float> logits_cpu = {0.1, 0.6, 0.1, 0.1, 0.1,
                                   0.1, 0.1, 0.6, 0.1, 0.1,
                                   0.1, 0.1, 0.1, 0.6, 0.1,
                                   0.1, 0.1, 0.1, 0.1, 0.6};
  auto logits_gpu = Generators::CudaMallocArray<float>(logits_cpu.size());
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  cudaMemcpyAsync(logits_gpu.get(), logits_cpu.data(), logits_cpu.size() * sizeof(float), cudaMemcpyHostToDevice, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  auto generator = Generators::CreateGenerator(*model, params);
  generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), logits_cpu.size()));
  // Verify outputs match expected outputs
  generator->search_->SampleTopP(0.25, 1.0);
  auto next_tokens = generator->search_->GetNextTokens().GetCPU();
  if (!std::equal(next_tokens.begin(), next_tokens.end(), output_span.begin(), output_span.end())) 
    throw std::runtime_error("Test Results Mismatch");
  std::cout << " completed Top P batched test\r\n";
}

void Batched_Sampling_TopK_Test() {
  // TODO: I don't like that I have to create a model here, but I need to pass it to the generator
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3};
  std::vector<float> logits_cpu{2.0, 1.5, 1.25, 0.25, 0.25,
                                0.25, 2.0, 1.25, 1.5, 0.25,
                                0.25, 2.0, 0.25, 1.5, 1.25,
                                1.25, 0.25, 1.5, 0.25, 2.0};
  auto logits_gpu = Generators::CudaMallocArray<float>(logits_cpu.size());
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  cudaMemcpyAsync(logits_gpu.get(), logits_cpu.data(), logits_cpu.size() * sizeof(float), cudaMemcpyHostToDevice, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  auto generator = Generators::CreateGenerator(*model, params);
  generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), logits_cpu.size()));
  // Verify outputs match expected outputs
  int k = 2;
  generator->search_->SampleTopK(k, 1.0);
  auto next_tokens = generator->search_->GetNextTokens().GetCPU();
  for (int b = 0; b < batch_size; b++) {
    auto next_token = next_tokens[b];
    auto next_token_score = logits_cpu[next_token + vocab_size * b];
    if (next_token_score < 1.5) {
      std::cout << "next_token_score: " << next_token_score << "\r\n";
      throw std::runtime_error("Test Results Mismatch");
    }
  }
  std::cout << " completed Top K batched test\r\n";
}

void Batched_Subset_TopK_Test() {
  // TODO: I don't like that I have to create a model here, but I need to pass it to the generator
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<float> logits_cpu{2.0, 1.5, 1.25, 0.25, 0.25,
                0.25, 2.0, 1.25, 1.5, 0.25,
                0.25, 2.0, 0.25, 1.5, 1.25,
                1.25, 0.25, 1.5, 0.25, 2.0};
  auto logits_gpu = Generators::CudaMallocArray<float>(logits_cpu.size());

  std::vector<int32_t> input_ids{0, 1, 2, 3};
  int k = 3;
  int vocab_size = 5;
  int batch_size = 4;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;

  cudaMemcpyAsync(logits_gpu.get(), logits_cpu.data(), logits_cpu.size() * sizeof(float), cudaMemcpyHostToDevice, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  auto generator = Generators::CreateGenerator(*model, params);

  generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), logits_cpu.size()));
  auto output_tokens = Generators::CudaMallocArray<int32_t>(batch_size * k);
  std::span<int32_t> output_tokens_span(output_tokens.get(), batch_size * k);
  generator->search_->GetTopKSubset(output_tokens_span.data(), k);

  auto output_tokens_cpu = std::vector<int>(batch_size * k);
  cudaMemcpyAsync(output_tokens_cpu.data(), output_tokens.get(), batch_size * k * sizeof(int32_t), cudaMemcpyDeviceToHost, params.cuda_stream);
  cudaStreamSynchronize(params.cuda_stream);
  // Verify outputs match expected outputs 
  for (int b = 0; b < batch_size; b++) {
    for (int i = 0; i < k; i++) {
      auto next_token = output_tokens_cpu[b * k + i];
      auto next_token_score = logits_cpu[next_token + vocab_size * b];
      if (next_token_score < 1.0) {
        std::cout << "next_token_score: " << next_token_score << "\r\n";
        throw std::runtime_error("Test Results Mismatch");
      }
    }
  }

  std::cout << " completed TopK Subset batched test\r\n";
}

void Randomized_Sampling_TopP_Test() {
  // TODO: I don't like that I have to create a model here, but I need to pass it to the generator
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000; // vocab size of llama
  int batch_size = 5;
  long long total_duration = 0;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  for (int i = 0; i < 100; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocHostArray<int>(vocab_size * batch_size);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpyAsync(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost, params.cuda_stream);

    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopP(0.95, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    total_duration += duration;

    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
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
  std::cout << " completed Top P randomized test\r\n";
}

void Randomized_Sampling_TopK_Test() {
  // TODO: I don't like that I have to create a model here, but I need to pass it to the generator
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  int vocab_size = 32000; // vocab size of llama
  int batch_size = 5;
  long long total_duration = 0;
  int k = 5;
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  for (int i = 0; i < 100; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocHostArray<int>(vocab_size * batch_size);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpyAsync(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost, params.cuda_stream);

    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    
    auto start = std::chrono::high_resolution_clock::now();
    generator->search_->SampleTopK(k, 1.0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    total_duration += duration;

    auto next_tokens = generator->search_->GetNextTokens().GetCPU();
    cudaStreamSynchronize(params.cuda_stream);
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      auto next_token = next_tokens[b];
      auto next_token_score = cpu_logits[next_token + vocab_size * b];
      if (next_token_score < 10.0) {
        std::cout << "next_token_score: " << next_token_score << "\r\n";
        throw std::runtime_error("Test Results Mismatch");
      }
    }
  }
  double averageDuration = static_cast<double>(total_duration) / 100.0;
  std::cout << "Average time taken by top k sampling: "
    << averageDuration << " microseconds" << std::endl;
  std::cout << " completed Top K randomized test\r\n";
}

void Randomized_Subset_TopK_Test() {
  // TODO: I don't like that I have to create a model here, but I need to pass it to the generator
  auto model = Generators::CreateModel(*g_ort_env, MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32");
  std::vector<int32_t> input_ids{0, 1, 2, 3, 4};
  int vocab_size = 32000; // vocab_size for llama model
  int batch_size = 5;
  int k = 5;
  Generators::GeneratorParams params = Generators::GeneratorParams{};
  params.max_length = 10;
  params.batch_size = batch_size;
  params.sequence_length = 1;
  params.vocab_size = vocab_size;
  params.input_ids = input_ids;
  params.device_type = Generators::DeviceType::CUDA;
  for (int i = 0; i < 100; i++) {
    auto logits_gpu = Generators::CudaMallocArray<float>(vocab_size * batch_size);
    auto indices_buffer = Generators::CudaMallocHostArray<int>(vocab_size * batch_size);
    LaunchGeometricDecayKernel(logits_gpu.get(), vocab_size, batch_size, params.cuda_stream);
    LaunchFisherYatesKernel(logits_gpu.get(), indices_buffer.get(), vocab_size, batch_size, params.cuda_stream);
    float* cpu_logits = new float[vocab_size * batch_size];
    cudaMemcpyAsync(cpu_logits, logits_gpu.get(), vocab_size * batch_size * sizeof(float), cudaMemcpyDeviceToHost, params.cuda_stream);

    Generators::GeneratorParams params = Generators::GeneratorParams{};
    params.max_length = 10;
    params.batch_size = batch_size;
    params.sequence_length = 1;
    params.vocab_size = vocab_size;
    params.input_ids = input_ids;
    params.device_type = Generators::DeviceType::CUDA;

    auto generator = Generators::CreateGenerator(*model, params);
    generator->search_->SetLogits(Generators::gpu_span<float>(logits_gpu.get(), vocab_size * batch_size));
    auto output_tokens = Generators::CudaMallocArray<int32_t>(batch_size * k);
    std::span<int32_t> output_tokens_span(output_tokens.get(), batch_size * k);
    generator->search_->GetTopKSubset(output_tokens_span.data(), k);

    auto output_tokens_cpu = std::vector<int>(batch_size * k);
    cudaMemcpyAsync(output_tokens_cpu.data(), output_tokens.get(), batch_size * k * sizeof(int32_t), cudaMemcpyDeviceToHost, params.cuda_stream);
    cudaStreamSynchronize(params.cuda_stream);
    // Verify outputs match expected outputs
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < k; i++) {
        auto next_token = output_tokens_cpu[b * k + i];
        auto next_token_score = cpu_logits[next_token + vocab_size * b];
        if (next_token_score < 1.0) {
          std::cout << "next_token_score: " << next_token_score << "\r\n";
          throw std::runtime_error("Test Results Mismatch");
        }
      }
    }
  }

  std::cout << " completed Top K Subset randomized test\r\n";
}

void Test_Sampling_Cuda() {
  Batched_Sampling_TopP_Test();
  Batched_Sampling_TopK_Test();
  Batched_Subset_TopK_Test();
  Randomized_Sampling_TopP_Test();
  Randomized_Sampling_TopK_Test();
  Randomized_Subset_TopK_Test();
}

void Test_Phi2_Cuda() {
#if TEST_PHI2
  std::cout << "Testing_Phi2\r\n";
#if USE_TOKENIZER

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

  Generators::GeneratorParams params{*model};
  params.batch_size = 1;
  params.sequence_length = static_cast<int>(tokens.size());
  params.input_ids = tokens;
  params.max_length = 128;

  // Original version
  auto search = params.CreateSearch();
  auto state = model->CreateState(search->GetSequenceLengths(), params);

  while (!search->IsDone()) {
    search->SetLogits(state->Run(search->GetSequenceLength(), search->GetNextTokens()));
    search->SelectTop();
  }

  auto result = search->GetSequence(0);

  // Generator version
  auto generator = model->CreateGenerator();
  while (!generator->IsDone()) {
    auto logits = generator->RunStep();

    generator->SelectTop();
  }

  auto result = generator->GetSequence(0);

  // High level version
  auto result = model->Generate(params);

  std::cout << tokenizer->Decode(result) << "\r\n";
  std::cout << "Test complete\r\n";
#else
  std::cout << "Test skipped - not built with onnxruntime extensions\r\n";
#endif
#endif
}

#endif
#include <generators.h>
#include <search.h>
#include <models/model.h>
#include <iostream>

// Our working directory is generators/build so one up puts us in the root directory:
#define MODEL_PATH "../../test_models/"

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
    generator->GenerateNextToken_Top();
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

    generator->GenerateNextToken_Top();
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

    generator->GenerateNextToken();
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

  Generators::SearchParams params{*model};
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

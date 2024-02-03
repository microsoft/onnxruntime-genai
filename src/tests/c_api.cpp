#include "../generators.h"
#include "../search.h"
#include "../models/model.h"
#include <iostream>
#include "../ort_genai_c.h"

// Our working directory is generators/build so one up puts us in the root directory:
#define MODEL_PATH "../test_models/"

struct Deleters {
  void operator()(OgaResult* p) {
    OgaDestroyResult(p);
  }
  void operator()(OgaModel* p) {
    OgaDestroyModel(p);
  }
  void operator()(OgaGeneratorParams* p) {
    OgaDestroyGeneratorParams(p);
  }
  void operator()(OgaGenerator* p) {
    OgaDestroyGenerator(p);
  }
};

using OgaResultPtr = std::unique_ptr<OgaResult, Deleters>;
using OgaModelPtr = std::unique_ptr<OgaModel, Deleters>;
using OgaGeneratorParamsPtr = std::unique_ptr<OgaGeneratorParams, Deleters>;
using OgaGeneratorPtr = std::unique_ptr<OgaGenerator, Deleters>;

void CheckResult(OgaResult* result) {
  if (!result)
    return;

  OgaResultPtr result_ptr{result};
  throw std::runtime_error(OgaResultGetError(result));
}

void Test_GreedySearch_Gpt_Fp32_C_API() {
  std::cout << "Test_GreedySearch_Gpt fp32 C API" << std::flush;

  std::vector<int64_t> input_ids_shape{2, 4};
  std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

  std::vector<int32_t> expected_output{
      0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
      0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

  auto sequence_length = input_ids_shape[1];
  auto batch_size = input_ids_shape[0];
  int max_length = 10;

  // To generate this file:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2 --output tiny_gpt2_greedysearch_fp16.onnx --use_gpu --max_length 20
  // And copy the resulting gpt2_init_past_fp32.onnx file into these two files (as it's the same for gpt2)

  OgaModel* model;
  CheckResult(OgaCreateModel(MODEL_PATH "hf-internal-testing/tiny-random-gpt2-fp32", OgaDeviceTypeCPU, &model));
  OgaModelPtr model_ptr{model};

  OgaGeneratorParams* params;
  CheckResult(OgaCreateGeneratorParams(model, &params));
  OgaGeneratorParamsPtr params_ptr{params};
  CheckResult(OgaGeneratorParamsSetMaxLength(params, max_length));
  CheckResult(OgaGeneratorParamsSetInputIDs(params, input_ids.data(), input_ids.size(), sequence_length, batch_size));

  OgaGenerator* generator;
  CheckResult(OgaCreateGenerator(model, params, &generator));
  OgaGeneratorPtr generator_ptr{generator};

  while (!OgaGenerator_IsDone(generator)) {
    CheckResult(OgaGenerator_ComputeLogits(generator));
    CheckResult(OgaGenerator_GenerateNextToken_Top(generator));
  }

  // Verify outputs match expected outputs
  for (int i = 0; i < batch_size; i++) {
    size_t token_count;
    CheckResult(OgaGenerator_GetSequence(generator, i, nullptr, &token_count));
    std::vector<int32_t> sequence(token_count);
    CheckResult(OgaGenerator_GetSequence(generator, i, sequence.data(), &token_count));

    auto* expected_output_start = &expected_output[i * max_length];
    if (!std::equal(expected_output_start, expected_output_start + max_length, sequence.begin(), sequence.end()))
      throw std::runtime_error("Test Results Mismatch");
  }

  std::cout << " - complete\r\n";
}

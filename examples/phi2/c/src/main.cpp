#include <iostream>
#include "ort_genai_c.h"

struct Deleters {
  void operator()(OgaResult* p) {
    OgaDestroyResult(p);
  }
  void operator()(OgaSequences* p) {
    OgaDestroySequences(p);
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
  void operator()(OgaTokenizer* p) {
    OgaDestroyTokenizer(p);
  }
};

using OgaResultPtr = std::unique_ptr<OgaResult, Deleters>;
using OgaSequencesPtr = std::unique_ptr<OgaSequences, Deleters>;
using OgaModelPtr = std::unique_ptr<OgaModel, Deleters>;
using OgaGeneratorParamsPtr = std::unique_ptr<OgaGeneratorParams, Deleters>;
using OgaGeneratorPtr = std::unique_ptr<OgaGenerator, Deleters>;
using OgaTokenizerPtr = std::unique_ptr<OgaTokenizer, Deleters>;

void CheckResult(OgaResult* result) {
  if (!result)
    return;

  OgaResultPtr result_ptr{result};
  throw std::runtime_error(OgaResultGetError(result));
}

int main() {
  std::cout << "-------------" << std::endl;
  std::cout << "Hello, Phi-2!" << std::endl;
  std::cout << "-------------" << std::endl;

  OgaModel* model;
  CheckResult(OgaCreateModel("phi-2", OgaDeviceTypeCPU, &model));
  OgaModelPtr model_ptr{model};

  OgaTokenizer* tokenizer;
  CheckResult(OgaCreateTokenizer(model, &tokenizer));
  OgaTokenizerPtr tokenizer_ptr{tokenizer};

  const char* prompt = "def is_prime(num):";
  std::cout << "Prompt: " << std::endl << prompt << std::endl;

  OgaSequences* sequences;
  CheckResult(OgaCreateSequences(&sequences));
  OgaSequencesPtr sequences_ptr{sequences};
  CheckResult(OgaTokenizerEncode(tokenizer, prompt, sequences));

  OgaGeneratorParams* params;
  CheckResult(OgaCreateGeneratorParams(model, &params));
  OgaGeneratorParamsPtr params_ptr{params};
  CheckResult(OgaGeneratorParamsSetMaxLength(params, 200));
  CheckResult(OgaGeneratorParamsSetInputSequences(params, sequences));

  OgaSequences* output_sequences;
  CheckResult(OgaGenerate(model, params, &output_sequences));
  OgaSequencesPtr output_sequences_ptr{output_sequences};

  size_t sequence_length = OgaSequencesGetSequenceCount(output_sequences, 0);
  const int32_t* sequence = OgaSequencesGetSequenceData(output_sequences, 0);

  const char* out_string;
  CheckResult(OgaTokenizerDecode(tokenizer, sequence, sequence_length, &out_string));

  std::cout << "Output: " << std::endl << out_string << std::endl;

  return 0;
}